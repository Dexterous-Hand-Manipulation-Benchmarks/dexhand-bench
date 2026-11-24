"""FastTD3's train.py setup on DexMachina, up to building the actor, with fake envs.

The run seed must decide the agent's initial state: building a DexMachina env reseeds the global numpy/torch RNGs
with a fixed seed (curriculum, ADD vertex sampling), and FastTD3 builds two envs (train, eval) before its
actor. The fake env here reseeds in every construction, like the real one.
The run must save the env it trains on next to its checkpoints, for evaluation to rebuild.
The run has W&B off, so train.py's env-config logging is skipped.
"""

import importlib.util
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fakes import FakeDexEnv, FakeDexVecEnv

pytest.importorskip("fast_td3.environments.dexmachina_env", reason="needs the fasttd3 extra")
genesis = pytest.importorskip("genesis")

FASTTD3_DIR = Path(__file__).parents[1] / "algorithms/FastTD3/fast_td3"
ENV_NAME = "dexmachina|allegro_hand|box-30-230"
NUM_ENVS = 2
ACTOR_MODULES = {"fasttd3": "fast_td3", "fasttd3_simbav2": "fast_td3_simbav2"}
# train.py imports these flat from its own directory; the installed fast_td3 package would shadow fast_td3.py.
FLAT_MODULES = ("fast_td3", "fast_td3_simbav2", "fast_td3_utils", "hyperparams", "environments")


class Captured(Exception):
    pass


class ReseedingEnv(FakeDexEnv):
    def __init__(self, seed: int) -> None:
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)


def make_reseeding_envs_builder(dexmachina_env, seed: int):
    """Stands in for build_dexmachina_envs: two envs (train, eval), each reseeding when built."""

    def build(spec, device, num_eval_envs):
        def wrapped():
            return dexmachina_env.GenesisFastTD3Wrapper(FakeDexVecEnv(ReseedingEnv(seed)), rl_device=device)

        train = wrapped()
        # Like DexMachinaVecEnv.from_spec, which records what the train env was built from.
        train.vec_env.spec, train.vec_env.reward_overrides = spec, {}
        return dexmachina_env.DexMachinaEnvs(train=train, evaluation=wrapped())

    return build


@pytest.fixture
def initial_actor_state(monkeypatch, tmp_path):
    # train.py sets env vars at import and pulls its flat modules into sys.modules; undo both afterwards.
    environ_before = dict(os.environ)
    modules_before = set(sys.modules)
    monkeypatch.syspath_prepend(str(FASTTD3_DIR))
    for name in FLAT_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location("fasttd3_train", FASTTD3_DIR / "train.py")
    assert spec is not None and spec.loader is not None
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    monkeypatch.setattr(train, "UniversalLogger", None)
    monkeypatch.setattr(genesis, "init", lambda **_: None)
    dexmachina_env = importlib.import_module("environments.dexmachina_env")

    run_dirs = itertools.count()

    def run(agent: str, run_seed: int, env_reseed: int, extra_args: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
        # train.py writes its run files under models/ in the working directory, named by the second it starts in.
        work_dir = tmp_path / f"run{next(run_dirs)}"
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)
        actor_module = importlib.import_module(ACTOR_MODULES[agent])
        real_actor = actor_module.Actor

        def capture_initial_state(**kwargs):
            actor = real_actor(**kwargs)
            raise Captured({k: v.detach().clone() for k, v in actor.state_dict().items()})

        monkeypatch.setattr(actor_module, "Actor", capture_initial_state)
        build_envs = make_reseeding_envs_builder(dexmachina_env, env_reseed)
        monkeypatch.setattr(dexmachina_env, "build_dexmachina_envs", build_envs)
        monkeypatch.setattr(sys, "argv", [
            "train.py", "--env_name", ENV_NAME, "--agent", agent,
            "--seed", str(run_seed), "--num_envs", str(NUM_ENVS), "--no_cuda", "--no_use_wandb", "--no_compile",
            *extra_args,
        ])
        with pytest.raises(Captured) as captured:
            train.main()
        monkeypatch.setattr(actor_module, "Actor", real_actor)  # the next run() wraps the real class again
        return captured.value.args[0]

    yield run
    os.environ.clear()
    os.environ.update(environ_before)
    for name in set(sys.modules) - modules_before:
        if name.split(".")[0] in FLAT_MODULES:
            del sys.modules[name]


def same_state(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("agent", ACTOR_MODULES)
def test_run_seed_decides_initial_actor_state_despite_env_reseeding(initial_actor_state, agent):
    seed_0 = initial_actor_state(agent, run_seed=0, env_reseed=42)
    assert "noise_scales" in seed_0
    assert not same_state(seed_0, initial_actor_state(agent, run_seed=1, env_reseed=42))
    assert same_state(seed_0, initial_actor_state(agent, run_seed=0, env_reseed=42))
    assert same_state(seed_0, initial_actor_state(agent, run_seed=0, env_reseed=123))


def test_the_run_saves_the_env_spec_it_trains_on_in_its_checkpoint_directory(initial_actor_state, tmp_path):
    from dexmachina.envs.run_spec import RUN_SPEC_FILENAME, current_dexmachina_state, load_run_spec
    from environments.dexmachina_env import benchmark_env_spec

    initial_actor_state("fasttd3", run_seed=3, env_reseed=42)

    saved_files = list(tmp_path.glob(f"run*/models/FastTD3/*/*/{RUN_SPEC_FILENAME}"))
    assert len(saved_files) == 1, f"expected one {RUN_SPEC_FILENAME} under models/FastTD3, found {saved_files}"
    run_spec = load_run_spec(saved_files[0].parent)
    assert run_spec.spec == benchmark_env_spec(ENV_NAME, seed=3, num_envs=NUM_ENVS)
    assert dict(run_spec.reward_overrides) == {}
    assert run_spec.dexmachina_state.sha == current_dexmachina_state().sha


def test_rendering_during_training_is_refused(initial_actor_state):
    with pytest.raises(ValueError, match="--render_interval is 100"):
        initial_actor_state("fasttd3", run_seed=0, env_reseed=42, extra_args=("--render_interval", "100"))

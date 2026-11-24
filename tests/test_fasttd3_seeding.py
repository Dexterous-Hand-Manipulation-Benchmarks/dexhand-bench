"""FastTD3's run seed must decide the agent's initial state on DexMachina.

Building a DexMachina env reseeds the global numpy/torch RNGs with a fixed seed (curriculum, ADD vertex sampling),
and FastTD3 builds three envs (train, eval, render) before its actor. The fake env here reseeds in every
construction, like the real one. The run has W&B off, so train.py's env-config logging is skipped.
"""

import importlib.util
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
    """Stands in for build_dexmachina_envs: three envs (train, eval, render), each reseeding when built."""

    def build(spec, device, num_eval_envs):
        def wrapped():
            return dexmachina_env.GenesisFastTD3Wrapper(FakeDexVecEnv(ReseedingEnv(seed)), rl_device=device)

        return dexmachina_env.DexMachinaEnvs(train=wrapped(), evaluation=wrapped(), render=wrapped())

    return build


@pytest.fixture
def initial_actor_state(monkeypatch):
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

    def run(agent: str, run_seed: int, env_reseed: int) -> dict[str, torch.Tensor]:
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

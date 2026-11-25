"""FastTD3's train.py setup on DexMachina, up to building the actor, with fake envs.

The run seed must decide the agent's initial state: building a DexMachina env reseeds the global numpy/torch RNGs
with a fixed seed (curriculum, ADD vertex sampling), and FastTD3 builds two envs (train, eval) before its
actor. The fake env here reseeds in every construction, like the real one.
The run must save the env it trains on next to its checkpoints, for evaluation to rebuild.
The runs have W&B off (train.py's env-config logging is skipped), except where a test passes a W&B stand-in.
The networks training starts from are pinned by a golden fixture, so refactoring how train.py builds them cannot change
training. Record it deliberately, on purpose-made changes only (a torch bump changes it too), always all cases:
    FASTTD3_RECORD_GOLDEN=1 uv run --exact --extra fasttd3 pytest tests/test_fasttd3_train_setup.py -k golden
"""

import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import CURRENT_PLATFORM, GOLDEN_DIR
from fakes import FakeDexEnv, FakeDexVecEnv

pytest.importorskip("fast_td3.environments.dexmachina_env", reason="needs the fasttd3 extra")
genesis = pytest.importorskip("genesis")

FASTTD3_DIR = Path(__file__).parents[1] / "algorithms/FastTD3/fast_td3"
ENV_NAME = "dexmachina|allegro_hand|box-30-230"
NUM_ENVS = 2
ACTOR_MODULES = {"fasttd3": "fast_td3", "fasttd3_simbav2": "fast_td3_simbav2"}
# train.py imports these flat from its own directory; the installed fast_td3 package would shadow fast_td3.py.
FLAT_MODULES = ("fast_td3", "fast_td3_simbav2", "fast_td3_utils", "hyperparams", "environments", "agent_builders")


INITIAL_NETWORKS_FILE = GOLDEN_DIR / "fasttd3_initial_networks.json"
RECORD_GOLDEN = os.environ.get("FASTTD3_RECORD_GOLDEN") == "1"
# The release, not the build tag (+cu124, +cpu): the CPU init does not depend on the CUDA build.
TORCH_RELEASE = torch.__version__.split("+")[0]
# Run seed and env reseed of every golden case; the case only chooses agent and network options.
GOLDEN_SEEDS = {"run_seed": 0, "env_reseed": 42}
GOLDEN_CASES = {
    "fasttd3": ("fasttd3", ()),
    "fasttd3_simbav2": ("fasttd3_simbav2", ()),
    # SimNorm is fasttd3-only; a non-default seq_len shows the actor's own seq_len reaches it.
    "fasttd3_sim_actor": ("fasttd3", ("--sim_type", "sim_actor", "--actor_seq_len", "4")),
    # Different actor and critic seq_len and a non-default sim_dimension: each network gets its own values.
    "fasttd3_sim_both": (
        "fasttd3",
        ("--sim_type", "sim_both", "--actor_seq_len", "4", "--critic_seq_len", "2", "--sim_dimension", "16"),
    ),
}
# Sums of float32 tensors in float64: equal bits give equal sums; another platform may round the init differently in
# the last bit, which this tolerance absorbs while a different init or draw order is off by far more.
STATS_RTOL = 1e-6
OUTPUT_RTOL = 1e-5
OUTPUT_ATOL = 1e-7


class Captured(Exception):
    pass


class ReseedingEnv(FakeDexEnv):
    def __init__(self, seed: int) -> None:
        super().__init__()
        # train.py logs the reward config to W&B.
        self.reward_module = SimpleNamespace(cfg={})
        np.random.seed(seed)
        torch.manual_seed(seed)


def make_reseeding_envs_builder(dexmachina_env, seed: int):
    """Stands in for build_dexmachina_envs: two envs (train, eval), each reseeding when built."""

    def build(spec, device, num_eval_envs):
        def wrapped(built_from=None):
            vec_env = FakeDexVecEnv(ReseedingEnv(seed))
            # Like DexMachinaVecEnv.from_spec, which records what the train env was built from.
            vec_env.spec, vec_env.reward_overrides = built_from, {}
            # train.py asks for cuda:<rank> even with --no_cuda; the fakes stay on the CPU.
            return dexmachina_env.GenesisFastTD3Wrapper(vec_env, rl_device="cpu")

        return dexmachina_env.DexMachinaEnvs(train=wrapped(built_from=spec), evaluation=wrapped())

    return build


@pytest.fixture
def train_setup(monkeypatch, tmp_path):
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

    def run(
        agent: str, run_seed: int, env_reseed: int, extra_args: tuple[str, ...] = (), stop_at: str = "actor",
        wandb_stand_in=None,
    ):
        """Runs train.main() until it builds the actor (returns its state), until the replay buffer (returns every
        (role, module) the actor and critic classes built, in order), or to the end (returns the working directory)."""
        # train.py writes its run files under models/ in the working directory, named by the second it starts in.
        work_dir = tmp_path / f"run{next(run_dirs)}"
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)
        agent_module = importlib.import_module(ACTOR_MODULES[agent])
        real_actor, real_critic = agent_module.Actor, agent_module.Critic
        built: list[tuple[str, torch.nn.Module]] = []

        def capture_initial_state(**kwargs):
            actor = real_actor(**kwargs)
            raise Captured({k: v.detach().clone() for k, v in actor.state_dict().items()})

        def recording(role, cls):
            def build(**kwargs):
                module = cls(**kwargs)
                built.append((role, module))
                return module

            return build

        def stop_at_replay_buffer(**_):
            raise Captured(built)

        if stop_at == "actor":
            monkeypatch.setattr(agent_module, "Actor", capture_initial_state)
        elif stop_at == "replay_buffer":
            monkeypatch.setattr(agent_module, "Actor", recording("actor", real_actor))
            monkeypatch.setattr(agent_module, "Critic", recording("critic", real_critic))
            monkeypatch.setattr(train, "SimpleReplayBuffer", stop_at_replay_buffer)
        elif stop_at != "end":
            raise ValueError(f"stop_at must be 'actor', 'replay_buffer' or 'end', got {stop_at!r}")
        build_envs = make_reseeding_envs_builder(dexmachina_env, env_reseed)
        monkeypatch.setattr(dexmachina_env, "build_dexmachina_envs", build_envs)
        monkeypatch.setattr(sys, "argv", [
            "train.py", "--env_name", ENV_NAME, "--agent", agent,
            "--seed", str(run_seed), "--num_envs", str(NUM_ENVS), "--no_cuda", "--no_compile",
            *(("--no_use_wandb",) if wandb_stand_in is None else ()), *extra_args,
        ])
        if wandb_stand_in is not None:
            monkeypatch.setattr(train, "wandb", wandb_stand_in)
        if stop_at == "end":
            train.main()
            return work_dir
        with pytest.raises(Captured) as captured:
            train.main()
        # The next run() wraps the real classes again.
        monkeypatch.setattr(agent_module, "Actor", real_actor)
        monkeypatch.setattr(agent_module, "Critic", real_critic)
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
def test_run_seed_decides_initial_actor_state_despite_env_reseeding(train_setup, agent):
    seed_0 = train_setup(agent, run_seed=0, env_reseed=42)
    assert "noise_scales" in seed_0
    assert not same_state(seed_0, train_setup(agent, run_seed=1, env_reseed=42))
    assert same_state(seed_0, train_setup(agent, run_seed=0, env_reseed=42))
    assert same_state(seed_0, train_setup(agent, run_seed=0, env_reseed=123))


def test_the_run_saves_the_env_spec_it_trains_on_in_its_checkpoint_directory(train_setup, tmp_path):
    from dexmachina.envs.run_spec import RUN_SPEC_FILENAME, current_dexmachina_state, load_run_spec
    from environments.dexmachina_env import benchmark_env_spec

    train_setup("fasttd3", run_seed=3, env_reseed=42)

    saved_files = list(tmp_path.glob(f"run*/models/FastTD3/*/*/{RUN_SPEC_FILENAME}"))
    assert len(saved_files) == 1, f"expected one {RUN_SPEC_FILENAME} under models/FastTD3, found {saved_files}"
    run_spec = load_run_spec(saved_files[0].parent)
    assert run_spec.spec == benchmark_env_spec(ENV_NAME, seed=3, num_envs=NUM_ENVS)
    assert dict(run_spec.reward_overrides) == {}
    assert run_spec.dexmachina_state.sha == current_dexmachina_state().sha


def test_rendering_during_training_is_refused(train_setup):
    with pytest.raises(ValueError, match="--render_interval is 100"):
        train_setup("fasttd3", run_seed=0, env_reseed=42, extra_args=("--render_interval", "100"))


def network_fingerprint(built: list[tuple[str, torch.nn.Module]]) -> dict:
    """Roles in construction order; shape, sum and sum of squares of every tensor of each module; outputs of the first
    actor and the first critic on fixed inputs (they also see constants that are not in the state dict, such as
    SimbaV2's c_shift). A different kwarg, init or RNG draw order changes at least one of them."""
    fake = FakeDexEnv()
    obs = torch.linspace(-1.0, 1.0, 4 * fake.obs_dim).reshape(4, -1)
    actions = torch.linspace(-0.5, 0.5, 4 * fake.action_dim).reshape(4, -1)
    (_, actor), (_, critic) = built[0], built[2]
    with torch.no_grad():
        actor_output = actor(obs)
        critic_output = torch.stack(critic(obs, actions))
    return {
        "roles": [role for role, _ in built],
        "stats": [
            {
                key: {"shape": list(t.shape), "sum": float(t.double().sum()), "sumsq": float(t.double().square().sum())}
                for key, t in module.state_dict().items()
            }
            for _, module in built
        ],
        "actor_output": actor_output.tolist(),
        "critic_output": critic_output.tolist(),
    }


@pytest.mark.parametrize("case", GOLDEN_CASES)
def test_golden_initial_networks_of_training(train_setup, case):
    agent, extra_args = GOLDEN_CASES[case]
    built = train_setup(agent, **GOLDEN_SEEDS, extra_args=extra_args, stop_at="replay_buffer")
    fingerprint = network_fingerprint(built)
    if RECORD_GOLDEN:
        recorded = json.loads(INITIAL_NETWORKS_FILE.read_text()) if INITIAL_NETWORKS_FILE.exists() else {"cases": {}}
        recorded.update(recorded_on=CURRENT_PLATFORM, torch_version=TORCH_RELEASE)
        recorded["cases"][case] = fingerprint
        INITIAL_NETWORKS_FILE.write_text(json.dumps(recorded, indent=1) + "\n")
        pytest.skip(f"recorded {case} into {INITIAL_NETWORKS_FILE.name}")

    golden = json.loads(INITIAL_NETWORKS_FILE.read_text())
    assert golden["torch_version"] == TORCH_RELEASE, (
        f"The fixture was recorded with torch {golden['torch_version']}, this is {TORCH_RELEASE}: a torch bump "
        "may change the initial networks. Re-record on purpose (see the module docstring) with a Decisions entry."
    )
    expected = golden["cases"][case]
    where = f"(fixture recorded on {golden['recorded_on']}, running on {CURRENT_PLATFORM})"
    assert fingerprint["roles"] == expected["roles"] == ["actor", "actor", "critic", "critic"]
    for index, (stats, expected_stats) in enumerate(zip(fingerprint["stats"], expected["stats"], strict=True)):
        assert stats.keys() == expected_stats.keys(), f"module {index} has different tensors"
        for key, tensor_stats in stats.items():
            expected_tensor = expected_stats[key]
            assert tensor_stats["shape"] == expected_tensor["shape"], f"module {index} {key} shape"
            assert (tensor_stats["sum"], tensor_stats["sumsq"]) == pytest.approx(
                (expected_tensor["sum"], expected_tensor["sumsq"]), rel=STATS_RTOL
            ), f"module {index} {key} {where}"
    for output in ("actor_output", "critic_output"):
        torch.testing.assert_close(
            torch.tensor(fingerprint[output]), torch.tensor(expected[output]),
            rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL, msg=lambda m, output=output: f"{output} {where}: {m}",
        )


# Saves need global_step > learning_starts and a multiple of save_interval (2 here). With 4 iterations the save at
# loop index 2 is mid-run; with 3 it is the last iteration, whose weights only final.pt holds.
@pytest.mark.parametrize(("total", "expected_step_files"), [(4, {"step3.pt": 3}), (3, {})])
def test_step_checkpoints_are_named_after_and_store_the_iterations_done(
    train_setup, monkeypatch, total, expected_step_files
):
    # The fake env has no curriculum.
    dexmachina_env = importlib.import_module("environments.dexmachina_env")
    monkeypatch.setattr(dexmachina_env, "advance_curriculum", lambda envs, step, total, current_epoch: current_epoch)
    work_dir = train_setup(
        "fasttd3", run_seed=0, env_reseed=42, stop_at="end",
        extra_args=(
            "--total_timesteps", str(total), "--save_interval", "2", "--learning_starts", "0", "--batch_size", "8",
        ),
    )
    (checkpoint_dir,) = work_dir.glob("models/FastTD3/*/*/")
    saved = {path.name: torch.load(path, weights_only=False)["global_step"] for path in checkpoint_dir.glob("*.pt")}
    step_files = {name: step for name, step in saved.items() if name.startswith("step")}
    assert step_files == expected_step_files
    assert saved["final.pt"] == total


def wandb_run(mode: str, project="a-project", entity="an-entity") -> SimpleNamespace:
    """The attributes train.py reads from wandb.run, as W&B sets them in each mode."""
    return SimpleNamespace(
        id="run-id", project=project, entity=entity, offline=mode == "offline", disabled=mode == "disabled"
    )


class WandbStandIn:
    """What train.py uses of the wandb module before it builds the actor."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.run = None
        self.config = SimpleNamespace(update=lambda *args, **kwargs: None)

    def init(self, project, entity=None, **_):
        self.run = wandb_run(self.mode, project, entity)
        return self.run


# Offline and disabled runs pass without the change too: they only guard that no file is written for them.
@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
def test_the_run_records_its_wandb_run_next_to_its_checkpoints_when_online(train_setup, tmp_path, mode):
    train_setup(
        "fasttd3", run_seed=3, env_reseed=42, wandb_stand_in=WandbStandIn(mode),
        extra_args=("--project", "a-project", "--entity", "an-entity"),
    )
    saved_files = list(tmp_path.glob("run*/models/FastTD3/*/*/wandb_run.json"))
    if mode == "online":
        (saved,) = saved_files
        assert json.loads(saved.read_text()) == {"id": "run-id", "project": "a-project", "entity": "an-entity"}
    else:
        assert saved_files == []


def test_a_recorded_wandb_run_is_never_replaced_and_an_unset_entity_is_null(train_setup, tmp_path):
    save_wandb_run = importlib.import_module("fast_td3_utils").save_wandb_run
    save_wandb_run(tmp_path, wandb_run("online", entity=""))
    assert json.loads((tmp_path / "wandb_run.json").read_text())["entity"] is None
    with pytest.raises(FileExistsError, match="another W&B run"):
        save_wandb_run(tmp_path, wandb_run("online"))

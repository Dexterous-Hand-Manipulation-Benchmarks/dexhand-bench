"""FastTD3's benchmark evaluation: it rebuilds actor and env from what the run saved, and refuses what disagrees."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from fasttd3_flat import FASTTD3_DIR, flat_imports, load_script

pytest.importorskip("fast_td3.environments.dexmachina_env", reason="needs the fasttd3 extra")
genesis = pytest.importorskip("genesis")
import dexmachina.envs.evaluation as shared_evaluation  # noqa: E402  (after the skip: needs dexmachina)
import dexmachina.envs.run_spec as run_spec  # noqa: E402
from dexmachina.envs.evaluation import (  # noqa: E402
    EVAL_SEED,
    BenchmarkEvaluation,
    EvalScore,
    load_trajectory,
    rollout_episode,
    score,
    zero_action_floor,
)
from dexmachina.envs.factory import make_eval_env, task_hash  # noqa: E402
from dexmachina.envs.run_spec import DexmachinaState, current_dexmachina_state, save_run_spec  # noqa: E402

ENV_NAME = "dexmachina|allegro_hand|box-30-230"
RUN_SEED = 0
NUM_ENVS = 2
OBS_DIM, ACT_DIM = 7, 3
CPU = "cpu"
AGENTS = ("fasttd3", "fasttd3_simbav2")


class Captured(Exception):
    pass


@pytest.fixture
def td3_eval(monkeypatch):
    with flat_imports(monkeypatch):
        yield load_script("eval_dexmachina")


@pytest.fixture(autouse=True)
def clean_dexmachina(monkeypatch):
    """The tests save and evaluate runs in this checkout of dexmachina, which may have uncommitted changes."""
    clean = DexmachinaState(current_dexmachina_state().sha, False, None)
    monkeypatch.setattr(run_spec, "current_dexmachina_state", lambda: clean)
    monkeypatch.setattr(shared_evaluation, "current_dexmachina_state", lambda: clean)
    return clean


def run_args(agent: str = "fasttd3", **overrides: Any) -> dict[str, Any]:
    """The args a DexMachina training run saves into its checkpoints (train.py's save_params: vars(args))."""
    args = vars(importlib.import_module("hyperparams").DexMachinaArgs())
    args.update({"env_name": ENV_NAME, "agent": agent, "seed": RUN_SEED, "num_envs": NUM_ENVS, **overrides})
    return args


def save_checkpoint(path: Path, args: dict[str, Any], obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, step: int = 3):
    """A fresh actor and normaliser saved the way train.py saves them; the normaliser has seen some observations."""
    builders = importlib.import_module("agent_builders")
    namespace = SimpleNamespace(**args)
    actor = builders.build_actor(namespace, obs_dim, act_dim, args["num_envs"], CPU)
    normalizer = builders.build_obs_normalizer(args["obs_normalization"], obs_dim, CPU)
    if args["obs_normalization"]:
        normalizer(5.0 + 3.0 * torch.randn(64, obs_dim))
    unused = nn.Identity()
    importlib.import_module("fast_td3_utils").save_params(
        step, actor, unused, unused, normalizer, unused, namespace, str(path)
    )
    return actor.eval(), normalizer.eval()


def loaded(td3_eval, path: Path, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM):
    checkpoint = torch.load(path, weights_only=False)
    saved = td3_eval.SavedArgs(**checkpoint["args"])
    return saved, *td3_eval.load_networks(saved, checkpoint, obs_dim, act_dim, CPU)


@pytest.mark.parametrize("agent", AGENTS)
def test_the_loaded_actor_acts_like_the_saved_one(td3_eval, tmp_path, agent):
    actor, normalizer = save_checkpoint(tmp_path / "final.pt", run_args(agent))
    _, loaded_actor, loaded_normalizer = loaded(td3_eval, tmp_path / "final.pt")
    observations = torch.randn(16, OBS_DIM)
    with torch.no_grad():
        assert torch.equal(
            loaded_actor(loaded_normalizer(observations, update=False)), actor(normalizer(observations, update=False))
        )


@pytest.mark.parametrize("agent", AGENTS)
def test_eval_actions_are_the_deterministic_mean_and_leave_the_normalizer_alone(td3_eval, tmp_path, monkeypatch, agent):
    # Wide noise bounds, so exploration noise would show in the actions.
    save_checkpoint(tmp_path / "final.pt", run_args(agent, std_min=0.5, std_max=0.9))
    saved, actor, normalizer = loaded(td3_eval, tmp_path / "final.pt")
    monkeypatch.setattr(type(actor), "explore", lambda *_, **__: pytest.fail("the eval explored"))
    stats_before = {name: buffer.clone() for name, buffer in normalizer.named_buffers()}
    observations = torch.randn(16, OBS_DIM)

    actions = td3_eval.deterministic_policy(actor, normalizer, saved, CPU)(observations)

    assert actions.dtype == torch.float32
    with torch.no_grad():
        assert torch.equal(actions, actor(normalizer(observations, update=False)))
    assert all(torch.equal(buffer, stats_before[name]) for name, buffer in normalizer.named_buffers())


@pytest.mark.parametrize(
    ("trained_with", "evaluated_as", "message"),
    [(True, False, "has a normaliser state although"), (False, True, "has no normaliser state")],
)
def test_a_normaliser_state_that_disagrees_with_the_saved_args_is_an_error(
    td3_eval, tmp_path, trained_with, evaluated_as, message
):
    save_checkpoint(tmp_path / "final.pt", run_args(obs_normalization=trained_with))
    checkpoint = torch.load(tmp_path / "final.pt", weights_only=False)
    checkpoint["args"]["obs_normalization"] = evaluated_as
    with pytest.raises(ValueError, match=message):
        td3_eval.load_networks(td3_eval.SavedArgs(**checkpoint["args"]), checkpoint, OBS_DIM, ACT_DIM, CPU)


def test_a_missing_saved_arg_is_an_error_not_a_default(td3_eval, tmp_path):
    save_checkpoint(tmp_path / "final.pt", run_args())
    checkpoint = torch.load(tmp_path / "final.pt", weights_only=False)
    del checkpoint["args"]["actor_seq_len"]
    with pytest.raises(AttributeError, match="saved args have no 'actor_seq_len'"):
        td3_eval.load_networks(td3_eval.SavedArgs(**checkpoint["args"]), checkpoint, OBS_DIM, ACT_DIM, CPU)


def test_a_missing_actor_weight_is_an_error(td3_eval, tmp_path):
    save_checkpoint(tmp_path / "final.pt", run_args())
    checkpoint = torch.load(tmp_path / "final.pt", weights_only=False)
    del checkpoint["actor_state_dict"]["noise_scales"]
    with pytest.raises(RuntimeError, match="noise_scales"):
        td3_eval.load_networks(td3_eval.SavedArgs(**checkpoint["args"]), checkpoint, OBS_DIM, ACT_DIM, CPU)


def save_run(run_dir: Path, **checkpoint_kwargs: Any) -> Path:
    """A run directory as train.py leaves it: env_spec.json and a checkpoint."""
    spec = importlib.import_module("environments.dexmachina_env").benchmark_env_spec(ENV_NAME, RUN_SEED, NUM_ENVS)
    save_run_spec(run_dir, spec, {})
    save_checkpoint(run_dir / "final.pt", run_args(), **checkpoint_kwargs)
    return run_dir / "final.pt"


def stop_before_the_env(monkeypatch, td3_eval):
    monkeypatch.setattr(td3_eval, "make_eval_env", lambda spec, device: pytest.fail("built the env"))


def test_a_run_without_env_spec_json_cannot_be_scored(td3_eval, tmp_path, monkeypatch):
    stop_before_the_env(monkeypatch, td3_eval)
    save_checkpoint(tmp_path / "final.pt", run_args())
    with pytest.raises(FileNotFoundError, match="env_spec.json does not exist"):
        td3_eval.evaluate(tmp_path / "final.pt", CPU)


def test_an_evaluated_checkpoint_is_refused_before_the_env_is_built(td3_eval, tmp_path, monkeypatch, clean_dexmachina):
    stop_before_the_env(monkeypatch, td3_eval)
    checkpoint = save_run(tmp_path)
    td3_eval.trajectory_path(checkpoint, clean_dexmachina.sha).touch()
    with pytest.raises(FileExistsError, match="Delete the file to evaluate it again"):
        td3_eval.evaluate(checkpoint, CPU)


@pytest.mark.parametrize(
    ("checkpoint_args", "message"),
    [
        ({"num_envs": NUM_ENVS + 1}, f"num_envs={NUM_ENVS + 1} but env_spec.json next to it has num_envs={NUM_ENVS}"),
        ({"seed": 1000}, f"seed=1000 but env_spec.json next to it has seed={RUN_SEED}"),
        ({"env_name": "dexmachina|allegro_hand|mixer-0-150"}, "clip='mixer-0-150"),
    ],
)
def test_a_checkpoint_from_another_run_is_refused(td3_eval, tmp_path, monkeypatch, checkpoint_args, message):
    stop_before_the_env(monkeypatch, td3_eval)
    spec = importlib.import_module("environments.dexmachina_env").benchmark_env_spec(ENV_NAME, RUN_SEED, NUM_ENVS)
    save_run_spec(tmp_path, spec, {})
    save_checkpoint(tmp_path / "final.pt", run_args(**checkpoint_args))
    with pytest.raises(ValueError, match=message):
        td3_eval.evaluate(tmp_path / "final.pt", CPU)


def test_a_dirty_dexmachina_is_refused(td3_eval, tmp_path, monkeypatch, clean_dexmachina):
    stop_before_the_env(monkeypatch, td3_eval)
    checkpoint = save_run(tmp_path)
    dirty = clean_dexmachina._replace(dirty=True, diff_sha256="0" * 64)
    monkeypatch.setattr(shared_evaluation, "current_dexmachina_state", lambda: dirty)
    with pytest.raises(RuntimeError, match="--allow-dirty-dexmachina"):
        td3_eval.evaluate(checkpoint, CPU)


def test_the_eval_seed_is_set_after_the_env_is_built(td3_eval, tmp_path, monkeypatch):
    checkpoint = save_run(tmp_path)
    calls: list[Any] = []
    fake_env = SimpleNamespace(obs_dim=OBS_DIM, action_dim=ACT_DIM)
    monkeypatch.setattr(td3_eval, "make_eval_env", lambda spec, device: calls.append("env") or fake_env)

    def seed(value):
        calls.append(("seed", value))
        raise Captured

    monkeypatch.setattr(td3_eval, "set_random_seed", seed)
    with pytest.raises(Captured):
        td3_eval.evaluate(checkpoint, CPU, eval_seed=7)
    assert calls == ["env", ("seed", 7)]


def stub_evaluation() -> BenchmarkEvaluation:
    result = EvalScore(
        auc_add=0.5, auc_add_of_mean=0.4, auc_add_part={"top": 0.6, "bottom": 0.4}, mean_add=0.01,
        success_2cm=0.3, success_5cm=0.7, nan_rate=0.0, drop_rate=0.5,
    )  # fmt: skip
    state = DexmachinaState("a" * 40, False, None)
    return BenchmarkEvaluation(result, "t", state, state, result, checkpoint_step=3, env_step=6, eval_seed=0)


class WandbStandIn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.summary: dict[str, Any] = {}

    def init(self, **kwargs):
        self.calls.append(("init", kwargs))
        return SimpleNamespace(summary=self.summary)

    def define_metric(self, name, step_metric):
        self.calls.append(("define_metric", (name, step_metric)))

    def log(self, metrics):
        self.calls.append(("log", metrics))

    def finish(self):
        self.calls.append(("finish", None))


IDENTITY = {"id": "run-id", "project": "a-project", "entity": None}


def run_main_with_wandb(td3_eval, tmp_path, monkeypatch, evaluate) -> None:
    (tmp_path / "wandb_run.json").write_text(json.dumps(IDENTITY))
    monkeypatch.setattr(td3_eval.gs, "init", lambda **_: None)
    monkeypatch.setattr(td3_eval, "evaluate", evaluate)
    monkeypatch.setattr(sys, "argv", ["eval_dexmachina.py", "--checkpoint", str(tmp_path / "final.pt"), "--wandb"])
    td3_eval.main()


def test_the_evaluation_is_logged_into_the_recorded_training_run(td3_eval, tmp_path, monkeypatch):
    stand_in = WandbStandIn()
    monkeypatch.setattr(td3_eval, "wandb", stand_in)
    evaluation = stub_evaluation()

    def evaluate(*_, **__):
        stand_in.calls.append(("evaluate", None))
        return evaluation

    run_main_with_wandb(td3_eval, tmp_path, monkeypatch, evaluate)

    assert stand_in.calls == [
        ("init", {**IDENTITY, "resume": "must"}),
        ("define_metric", ("benchmark/*", "benchmark/env_step")),
        ("evaluate", None),
        ("log", evaluation.metrics()),
        ("finish", None),
    ]
    assert stand_in.summary == evaluation.provenance()


def test_a_wandb_failure_stops_before_the_evaluation_writes_anything(td3_eval, tmp_path, monkeypatch):
    stand_in = WandbStandIn()
    stand_in.init = lambda **_: (_ for _ in ()).throw(RuntimeError("W&B is unreachable"))
    monkeypatch.setattr(td3_eval, "wandb", stand_in)
    with pytest.raises(RuntimeError, match="unreachable"):
        run_main_with_wandb(td3_eval, tmp_path, monkeypatch, lambda *_, **__: pytest.fail("evaluated"))


def test_with_wandb_a_run_without_its_recorded_wandb_run_is_refused_before_genesis(td3_eval, tmp_path, monkeypatch):
    checkpoint = save_run(tmp_path)
    monkeypatch.setattr(td3_eval.gs, "init", lambda **_: pytest.fail("initialised Genesis"))
    monkeypatch.setattr(sys, "argv", ["eval_dexmachina.py", "--checkpoint", str(checkpoint), "--wandb"])
    with pytest.raises(FileNotFoundError, match="Evaluate without --wandb"):
        td3_eval.main()


@pytest.mark.parametrize("script", ["eval_dexmachina.py", "train.py"])
def test_the_scripts_start_the_way_train_sh_runs_them(script):
    result = subprocess.run(
        [sys.executable, f"fast_td3/{script}", "--help"], cwd=FASTTD3_DIR.parent, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.sim
def test_a_fresh_actor_is_scored_on_the_saved_spec(
    td3_eval, genesis_cpu, isolated_hand_cfgs, tmp_path, monkeypatch, clean_dexmachina
):
    spec = importlib.import_module("environments.dexmachina_env").benchmark_env_spec(ENV_NAME, RUN_SEED, NUM_ENVS)
    reference_env: Any = make_eval_env(spec, CPU)
    expected_floor = zero_action_floor(reference_env, spec.hand, spec.clip)
    obs_dim, act_dim = int(reference_env.obs_dim), int(reference_env.action_dim)
    checkpoint = save_run(tmp_path, obs_dim=obs_dim, act_dim=act_dim, step=5)
    saved, actor, normalizer = loaded(td3_eval, checkpoint, obs_dim, act_dim)
    expected_policy_score = score(
        rollout_episode(
            reference_env, td3_eval.deterministic_policy(actor, normalizer, saved, CPU), spec.hand, spec.clip,
            task_hash(spec), clean_dexmachina.sha,
        )
    )
    episodes: list[tuple[str, int]] = []
    for name in ("zero_action_floor", "rollout_episode"):
        real = getattr(td3_eval, name)

        def recorded(env, *args, _name=name, _real=real, **kwargs):
            episodes.append((_name, id(env)))
            return _real(env, *args, **kwargs)

        monkeypatch.setattr(td3_eval, name, recorded)

    evaluation = td3_eval.evaluate(checkpoint, CPU)

    assert [name for name, _ in episodes] == ["zero_action_floor", "rollout_episode"]
    assert episodes[0][1] == episodes[1][1], "the floor and the policy ran on different envs"
    assert evaluation.task_hash == task_hash(spec)
    assert evaluation.floor == expected_floor
    assert evaluation.eval_score == expected_policy_score
    assert (evaluation.checkpoint_step, evaluation.env_step) == (5, 5 * NUM_ENVS)
    (trajectory_file,) = tmp_path.glob("benchmark_trajectory_final_*.npz")
    assert score(load_trajectory(trajectory_file)) == evaluation.eval_score
    assert evaluation.eval_seed == EVAL_SEED

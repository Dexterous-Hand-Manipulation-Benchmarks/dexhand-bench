"""FlashSAC's benchmark evaluation: it rebuilds agent and env from what the run saved, and refuses what disagrees."""
import ast
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from conftest import flashsac_env_kwargs, load_golden
from fakes import FakeDexEnv, FakeDexVecEnv

flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
gym = pytest.importorskip("gymnasium")
create_agent = pytest.importorskip("flash_rl.agents").create_agent
from dexmachina.envs.evaluation import (  # noqa: E402  (after the skip: needs dexmachina)
    EvalScore,
    load_trajectory,
    score,
)
from dexmachina.envs.factory import task_hash  # noqa: E402
from dexmachina.envs.run_spec import DexmachinaState, current_dexmachina_state, load_run_spec  # noqa: E402

FLASHSAC_DIR = Path(__file__).parents[1] / "algorithms/FlashSAC"
CONFIG_DIR = str(FLASHSAC_DIR / "configs")
GOLDEN = "flashsac_allegro_box"
CPU = "cpu"
STEP_DIRNAME = "step1"
FLOOR_AUC_ADD = 0.25
# What the training run's Hydra config turns into on the CPU: no AMP, a tiny unused replay buffer.
CPU_AGENT_OVERRIDES = [
    "agent.use_amp=false",
    "agent.device_type=cpu",
    "agent.buffer_device_type=cpu",
    "agent.buffer_max_length=10",
    "agent.buffer_min_length=1",
]
TRAINING_SAMPLES = 3
TRAIN_NUM_ENVS = 4
SAMPLE_BATCH = 4

_module = importlib.util.spec_from_file_location("flashsac_eval_dexmachina", FLASHSAC_DIR / "eval_dexmachina.py")
assert _module is not None and _module.loader is not None
eval_dexmachina = importlib.util.module_from_spec(_module)
sys.modules[_module.name] = eval_dexmachina
_module.loader.exec_module(eval_dexmachina)
PREFIX = eval_dexmachina.BENCHMARK_PREFIX


@pytest.fixture(autouse=True)
def no_dynamo(monkeypatch):
    # The agent compiles helper functions even with use_compile=false; these tests only need its weights and actions.
    monkeypatch.setattr("torch._dynamo.config.disable", True)


def _save_run(run_dir: Path, normalize_observation: bool = True, use_compile: bool = False) -> tuple[Path, Any]:
    """Trains a fresh agent for a few samples and saves it and the run files the way train.py does."""
    meta = load_golden(GOLDEN)[1]
    kwargs = flashsac_env_kwargs(meta["env_kwargs"])
    spec, reward_overrides = flash_dex.dexmachina_spec(TRAIN_NUM_ENVS, kwargs, meta["seed"], kwargs["env_name"])
    vec: Any = FakeDexVecEnv(FakeDexEnv())
    vec.spec, vec.reward_overrides = spec, reward_overrides
    cfg, _ = flash_dex.compose_eval_config(
        CONFIG_DIR,
        "flashSAC_base",
        [
            "env=dexmachina",
            f"agent.normalize_observation={str(normalize_observation).lower()}",
            f"agent.use_compile={str(use_compile).lower()}",
            *CPU_AGENT_OVERRIDES,
        ],
    )
    flash_dex.save_run_files(cfg, flash_dex.GenesisFlashSACWrapper(vec, rl_device=CPU), run_dir)
    agent = create_agent(
        observation_space=gym.spaces.Box(-float("inf"), float("inf"), (meta["obs_dim"],), dtype="float32"),
        action_space=gym.spaces.Box(-1.0, 1.0, (meta["action_dim"],), dtype="float32"),
        env_info={},
        cfg=cfg.agent,
    )
    for _ in range(TRAINING_SAMPLES):  # make the normalizer statistics differ from a fresh one's
        transition = {"next_observation": 5.0 + 3.0 * torch.randn(SAMPLE_BATCH, meta["obs_dim"])}
        agent.sample_actions(interaction_step=0, prev_transition=transition, training=True)
    agent.save(str(run_dir / STEP_DIRNAME))
    return run_dir / STEP_DIRNAME, agent


def _floor_content(run_dir: Path, sha: str, **changes: Any) -> dict[str, Any]:
    spec = load_run_spec(run_dir).spec
    content = {
        "hand": spec.hand, "clip": spec.clip, "task_hash": task_hash(spec), "dexmachina_sha": sha,
        "dexmachina_dirty": False, "policy": "zero", "device": CPU,
        "per_seed_auc_add": [FLOOR_AUC_ADD] * 3, "auc_add": FLOOR_AUC_ADD,
    }  # fmt: skip
    return {**content, **changes}


def _write_floor(run_dir: Path, sha: str, **changes: Any) -> None:
    content = _floor_content(run_dir, sha, **changes)
    name = eval_dexmachina.FLOOR_FILENAME.format(task_hash=task_hash(load_run_spec(run_dir).spec), sha=sha)
    (run_dir / name).write_text(json.dumps(content))


def _emitted_by_the_training_eval() -> set[str]:
    """Every string constant in flash_rl/evaluation.py: the metric names the in-training eval can log."""
    tree = ast.parse((FLASHSAC_DIR / "flash_rl/evaluation.py").read_text())
    return {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}


@pytest.mark.sim
def test_a_fresh_agent_is_scored_on_the_saved_spec(genesis_cpu, isolated_hand_cfgs, tmp_path):
    step_dir, _ = _save_run(tmp_path)
    sha = current_dexmachina_state().sha
    _write_floor(tmp_path, sha)

    evaluation = eval_dexmachina.evaluate_benchmark(step_dir, CPU)

    metrics = evaluation.metrics()
    expected = {
        "auc_add", "auc_add_of_mean", "mean_add", "success_2cm", "success_5cm", "nan_rate", "drop_rate",
        "floor/auc_add", "checkpoint_step", "env_step", "auc_add_part/top", "auc_add_part/bottom",
    }  # fmt: skip
    assert set(metrics) == {f"{PREFIX}/{name}" for name in expected}
    assert all(math.isfinite(value) for value in metrics.values()), metrics
    assert 0.0 <= evaluation.eval_score.auc_add <= 1.0
    assert metrics[f"{PREFIX}/floor/auc_add"] == FLOOR_AUC_ADD
    assert (metrics[f"{PREFIX}/checkpoint_step"], metrics[f"{PREFIX}/env_step"]) == (1.0, float(TRAIN_NUM_ENVS))
    assert evaluation.evaluated_state == DexmachinaState(sha, current_dexmachina_state().dirty)
    short_sha = sha[: eval_dexmachina.SHORT_SHA_LENGTH]
    stored = load_trajectory(step_dir / eval_dexmachina.TRAJECTORY_FILENAME.format(sha=short_sha))
    assert score(stored) == evaluation.eval_score
    assert stored.spec_hash == evaluation.task_hash


def _stub_evaluation() -> Any:
    result = EvalScore(
        auc_add=0.5, auc_add_of_mean=0.5, auc_add_part={"top": 0.5, "bottom": 0.5}, mean_add=0.01,
        success_2cm=0.5, success_5cm=0.5, nan_rate=0.0, drop_rate=0.0,
    )  # fmt: skip
    return eval_dexmachina.BenchmarkEvaluation(
        eval_score=result, task_hash="t", saved_state=DexmachinaState("a", False),
        evaluated_state=DexmachinaState("a", False), floor=eval_dexmachina.Floor(0.1, CPU),
        checkpoint_step=1, env_step=1,
    )


def test_the_logged_keys_stay_clear_of_the_ones_the_training_eval_logs():
    evaluation = _stub_evaluation()
    keys = set(evaluation.metrics()) | set(evaluation.provenance())
    emitted = _emitted_by_the_training_eval()
    assert "eval/success_2cm" in emitted  # the derivation sees what it should avoid
    assert keys.isdisjoint(emitted)
    assert evaluation.provenance()[f"{PREFIX}/floor_device"] == "cpu"
    assert all(key.startswith(f"{PREFIX}/") for key in keys)


def test_a_run_without_env_spec_json_points_at_the_old_protocol(tmp_path):
    step_dir, _ = _save_run(tmp_path)
    (tmp_path / "env_spec.json").unlink()
    with pytest.raises(FileNotFoundError, match="--old-protocol"):
        eval_dexmachina.evaluate_benchmark(step_dir, CPU)


def test_a_missing_floor_is_an_error(tmp_path):
    step_dir, _ = _save_run(tmp_path)
    with pytest.raises(FileNotFoundError, match="floor"):
        eval_dexmachina.evaluate_benchmark(step_dir, CPU)


def test_a_different_dexmachina_commit_is_warned_about(tmp_path, monkeypatch, capsys):
    step_dir, _ = _save_run(tmp_path)
    saved_state = current_dexmachina_state()
    other_commit = "0" * len(saved_state.sha)
    monkeypatch.setattr(
        eval_dexmachina, "current_dexmachina_state", lambda: DexmachinaState(other_commit, saved_state.dirty)
    )
    with pytest.raises(FileNotFoundError, match="floor"):  # the evaluation stops at the missing floor of that commit
        eval_dexmachina.evaluate_benchmark(step_dir, CPU)
    assert f"evaluated with {other_commit[:12]}" in capsys.readouterr().out


def test_an_obs_normalizer_file_with_a_config_that_does_not_normalize_is_an_error(tmp_path):
    step_dir, _ = _save_run(tmp_path, normalize_observation=True)
    config_path = tmp_path / flash_dex.RUN_CONFIG_FILENAME
    config = OmegaConf.load(config_path)
    config.agent.normalize_observation = False
    OmegaConf.save(config, config_path)
    with pytest.raises(ValueError, match="normalize_observation false"):
        eval_dexmachina.evaluate_benchmark(step_dir, CPU)


def test_a_config_that_normalizes_without_an_obs_normalizer_file_is_an_error(tmp_path):
    step_dir, _ = _save_run(tmp_path, normalize_observation=True)
    (step_dir / eval_dexmachina.OBS_NORMALIZER_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="normalize_observation true"):
        eval_dexmachina.evaluate_benchmark(step_dir, CPU)


def test_a_directory_that_is_not_a_step_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a checkpoint directory name"):
        eval_dexmachina.checkpoint_step(tmp_path / "final")
    assert eval_dexmachina.checkpoint_step(tmp_path / "step48828") == 48828


def _eval_agent(run_dir: Path, step_dir: Path) -> Any:
    config = OmegaConf.load(run_dir / flash_dex.RUN_CONFIG_FILENAME)
    meta = load_golden(GOLDEN)[1]
    agent = eval_dexmachina.build_eval_agent(config.agent, meta["obs_dim"], meta["action_dim"], 1.0, CPU)
    agent.load(str(step_dir))
    return agent


def test_eval_actions_are_deterministic_and_leave_the_normalizer_alone(tmp_path):
    step_dir, _ = _save_run(tmp_path)
    agent = _eval_agent(tmp_path, step_dir)
    observations = torch.randn(SAMPLE_BATCH, load_golden(GOLDEN)[1]["obs_dim"])
    before = agent.obs_normalizer.count.clone()
    first = eval_dexmachina.deterministic_actions(agent, observations, CPU)
    second = eval_dexmachina.deterministic_actions(agent, observations, CPU)
    assert torch.equal(first, second)
    assert torch.equal(agent.obs_normalizer.count, before)
    agent.sample_actions(0, {"next_observation": observations}, training=True)
    assert not torch.equal(agent.obs_normalizer.count, before)  # the check above can fail


def test_the_loaded_agent_acts_like_the_saved_one(tmp_path):
    step_dir, saved_agent = _save_run(tmp_path)
    (step_dir / "agent_state.pt").unlink()  # the optimizer state is not loaded for evaluation
    loaded = _eval_agent(tmp_path, step_dir)
    observations = torch.randn(SAMPLE_BATCH, load_golden(GOLDEN)[1]["obs_dim"])
    saved_actions = eval_dexmachina.deterministic_actions(saved_agent, observations, CPU)
    assert torch.equal(eval_dexmachina.deterministic_actions(loaded, observations, CPU), saved_actions)


def test_a_config_with_compile_on_loads(tmp_path):
    step_dir, _ = _save_run(tmp_path, use_compile=True)
    assert _eval_agent(tmp_path, step_dir) is not None


def test_asymmetric_observations_are_refused(tmp_path):
    _save_run(tmp_path)
    config = OmegaConf.load(tmp_path / flash_dex.RUN_CONFIG_FILENAME)
    config.agent.asymmetric_observation = True
    with pytest.raises(ValueError, match="asymmetric_observation"):
        eval_dexmachina.build_eval_agent(config.agent, 4, 2, 1.0, CPU)


FLOOR_SHA = "ab" * 20
FLOOR_TASK = "0123456789ab"
FLOOR = {
    "hand": "allegro_hand", "clip": "box", "task_hash": FLOOR_TASK, "dexmachina_sha": FLOOR_SHA,
    "dexmachina_dirty": False, "policy": "zero", "device": CPU, "per_seed_auc_add": [0.25, 0.25], "auc_add": 0.25,
}  # fmt: skip


def _read_floor(tmp_path: Path, content: Any) -> Any:
    name = eval_dexmachina.FLOOR_FILENAME.format(task_hash=FLOOR_TASK, sha=FLOOR_SHA)
    (tmp_path / name).write_text(json.dumps(content))
    evaluated = DexmachinaState(FLOOR_SHA, True)
    return eval_dexmachina.read_floor(tmp_path, "allegro_hand", "box", FLOOR_TASK, evaluated)


def test_a_valid_floor_is_read(tmp_path):
    assert _read_floor(tmp_path, FLOOR) == eval_dexmachina.Floor(auc_add=0.25, device=CPU)


@pytest.mark.parametrize(
    "changes",
    [
        {"hand": "inspire_hand"},
        {"clip": "mixer"},
        {"task_hash": "ffffffffffff"},
        {"dexmachina_sha": "cd" * 20},
        {"dexmachina_dirty": True},
        {"policy": "uniform"},
        {"per_seed_auc_add": [0.25, 0.26]},
        {"per_seed_auc_add": []},
        {"per_seed_auc_add": "0.25"},
        {"auc_add": 0.3},
        {"auc_add": True, "per_seed_auc_add": [True, True]},
        {"auc_add": float("nan"), "per_seed_auc_add": [float("nan")] * 2},
        {"auc_add": float("inf"), "per_seed_auc_add": [float("inf")] * 2},
        {"auc_add": "0.25"},
    ],
    ids=lambda changes: ",".join(changes),
)
def test_a_floor_that_is_not_this_evaluations_zero_floor_is_refused(tmp_path, changes):
    with pytest.raises(ValueError):
        _read_floor(tmp_path, {**FLOOR, **changes})


@pytest.mark.parametrize("content", [[0.25], {k: v for k, v in FLOOR.items() if k != "device"}, {**FLOOR, "extra": 1}])
def test_a_floor_without_exactly_the_floor_keys_is_refused(tmp_path, content):
    with pytest.raises(ValueError, match="exactly the keys"):
        _read_floor(tmp_path, content)


def test_main_runs_the_protocol_the_flag_names(monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(eval_dexmachina, "run_old_protocol", lambda args: called.append("old"))
    monkeypatch.setattr(eval_dexmachina, "run_benchmark", lambda args: called.append("benchmark"))
    eval_dexmachina.main(["--checkpoint_path", "x"])
    eval_dexmachina.main(["--checkpoint_path", "x", "--old-protocol", "--step", "3"])
    assert called == ["benchmark", "old"]


@pytest.mark.parametrize("flag", [["--step", "3"], ["--num_episodes", "2"], ["--overrides", "a=1"], ["--no-video"]])
def test_old_protocol_arguments_without_the_flag_are_refused(monkeypatch, flag):
    monkeypatch.setattr(eval_dexmachina, "run_benchmark", lambda args: pytest.fail("ran the benchmark"))
    with pytest.raises(SystemExit, match="only apply to --old-protocol"):
        eval_dexmachina.main(["--checkpoint_path", "x", *flag])


def test_the_benchmark_metrics_are_plotted_against_the_env_step(monkeypatch):
    calls: list[tuple[str, Any]] = []
    fake_run = SimpleNamespace(summary=SimpleNamespace(update=lambda values: calls.append(("summary", values))))
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: fake_run,
        define_metric=lambda name, **kwargs: calls.append(("define_metric", (name, kwargs))),
        log=lambda values: calls.append(("log", values)),
        finish=lambda: None,
    )
    monkeypatch.setattr(eval_dexmachina, "wandb", fake_wandb)
    monkeypatch.setattr(eval_dexmachina.gs, "init", lambda **kwargs: None)
    monkeypatch.setattr(eval_dexmachina, "evaluate_benchmark", lambda *args: _stub_evaluation())
    argv = ["--checkpoint_path", "step1", "--wandb_run_id", "r", "--device", CPU]
    args = eval_dexmachina.build_parser().parse_args(argv)
    eval_dexmachina.run_benchmark(args)
    kinds = [kind for kind, _ in calls]
    assert kinds == ["define_metric", "log", "summary"]  # defined before the first log
    assert calls[0][1] == (f"{PREFIX}/*", {"step_metric": f"{PREFIX}/env_step"})
    assert f"{PREFIX}/floor_device" in calls[2][1]

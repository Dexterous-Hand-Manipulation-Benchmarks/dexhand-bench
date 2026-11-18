"""The floor script: its summary and cells without a simulator, its protocol against the old one on it."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from conftest import isolate_hand_cfgs

pytest.importorskip("dexmachina.eval.compute_add", reason="needs genesis and dexmachina")
from dexmachina.envs.add_metric import ADD_AUC_THRESHOLDS, ADD_PARTS  # noqa: E402  (after the skip: needs dexmachina)
from dexmachina.envs.benchmark_task import RunChoices, benchmark_spec  # noqa: E402
from dexmachina.envs.evaluation import ADD_LOG_KEY  # noqa: E402
from dexmachina.envs.factory import make_env  # noqa: E402
from dexmachina.eval.compute_add import compute_add_stats  # noqa: E402

SCRIPT = Path(__file__).parents[1] / "scripts/analysis/floor_add_auc.py"
spec = importlib.util.spec_from_file_location("floor_add_auc", SCRIPT)
assert spec is not None and spec.loader is not None
floor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = floor
spec.loader.exec_module(floor)

NAN = float("nan")


def _row(
    hand: str = "allegro_hand",
    policy: str = "zero",
    seed: int = 0,
    auc: float = 0.5,
    mean_add: float = 0.1,
    error: str = "",
    **overrides: Any,
) -> Any:
    values: dict[str, Any] = dict(
        hand=hand, clip="box", policy=policy, seed=seed, auc_overall=auc, auc_top=auc, auc_bottom=auc,
        mean_add=mean_add, drop_rate=0.0, nan_rate=0.0, wall_seconds=1.0, error=error,
    )  # fmt: skip
    return floor.CellRow(**{**values, **overrides})


def test_summary_gives_mean_and_min_max_over_seeds() -> None:
    text = floor.summarize([_row(seed=0, auc=0.2), _row(seed=1, auc=0.4), _row(seed=2, auc=0.9)])
    assert "| allegro_hand | box | zero | 3 | 0.5000 | 0.2000-0.9000 |" in text


def test_summary_lists_failed_cells_loudly_and_keeps_them_out_of_the_statistics() -> None:
    failed = _row(hand="xhand", seed=1, auc=NAN, mean_add=NAN, error="FileNotFoundError: no clip")
    text = floor.summarize([_row(auc=0.3), failed])
    assert "FAILED CELLS (1 of 2 runs)" in text
    assert "xhand / box / zero / seed 1: FileNotFoundError: no clip" in text
    assert "| allegro_hand | box | zero | 1 | 0.3000 |" in text


def test_summary_gives_the_mean_drop_and_nan_rates() -> None:
    text = floor.summarize([_row(seed=0, drop_rate=1.0, nan_rate=0.0), _row(seed=1, drop_rate=0.0, nan_rate=1.0)])
    assert "| 0.1000 | 0.50 | 0.50 |" in text


def test_run_cell_records_the_error_and_nan_metrics_when_the_env_cannot_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_build(hand: str, clip: str, seed: int) -> None:
        raise FileNotFoundError("no retargeted clip")

    monkeypatch.setattr(floor, "build_env", failing_build)
    row = floor.run_cell("xhand", "box", floor.Policy.ZERO, 1, {})
    assert row.error == "FileNotFoundError: no retargeted clip"
    assert (row.hand, row.clip, row.policy, row.seed) == ("xhand", "box", "zero", 1)
    assert np.isnan(row.auc_overall) and np.isnan(row.mean_add)


def test_run_cell_fills_the_row_from_the_score_of_the_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    episode_score = floor.EvalScore(
        auc_add=0.4, auc_add_of_mean=0.3, auc_add_part={"top": 0.5, "bottom": 0.3}, mean_add=0.07,
        success_2cm=0.0, success_5cm=0.0, nan_rate=1.0, drop_rate=0.5,
    )  # fmt: skip
    monkeypatch.setattr(floor, "build_env", lambda hand, clip, seed: None)
    monkeypatch.setattr(floor, "run_episode", lambda env, policy, seed, hand, clip: episode_score)
    row = floor.run_cell("allegro_hand", "box", floor.Policy.UNIFORM, 0, {})
    assert row.error == ""
    assert (row.auc_overall, row.auc_top, row.auc_bottom, row.mean_add) == (0.4, 0.5, 0.3, 0.07)
    assert (row.drop_rate, row.nan_rate) == (0.5, 1.0)


# Simulator tests: allegro hand, box clip, two zero-action rollouts (about 40 s on the CPU).
HAND = "allegro_hand"
CLIP = floor.BENCHMARK_CLIPS[0]
SEED = 0
OLD_PROTOCOL_NUM_ENVS = 2
# The floor and the old protocol run the same physics (only the env count and the assist gains differ), but the
# result differs between CI runners with identical packages: on one runner 3 (frame, threshold) decisions of the
# falling box flipped (0.11 vs 0.1090625, exactly 3 of the steps below), on another runner and on the Mac none did.
# The smallest step of the old protocol's AUC, one flip at an end threshold:
ONE_THRESHOLD_FLIP = 1 / 3200
# A few flips are runner noise. A real protocol difference moves many more: leaving the assist gains on moved the old
# protocol by about 25 flips (0.10875 to 0.1167 on the Mac). Above this, find a named cause.
THRESHOLD_FLIPS_ALLOWED = 5
FLOOR_PROTOCOL_TOLERANCE = THRESHOLD_FLIPS_ALLOWED * ONE_THRESHOLD_FLIP


def _old_protocol_auc(num_envs: int) -> float:
    """The floor as measured before: no early or fell-off resets, assist gains zeroed, its own scoring.

    The zero action is deterministic, so seeding matters only for the uniform policy.
    """
    spec = benchmark_spec(RunChoices(hand=HAND, clip=CLIP, seed=SEED, num_envs=num_envs))
    env: Any = make_env(replace(spec, early_reset_threshold=0.0), "cpu", {})
    env.object.set_joint_gains(0.0, 0.0, force_range=0.0)
    env.add_metric = env.object.setup_add_metric(num_samples=env.add_num_verts)
    env.table_height = float("-inf")
    env.record_curriculum = False
    frames: dict[str, list[Any]] = {part: [] for part in ADD_PARTS}
    env.reset()
    for _ in range(env.max_episode_length):
        extras = env.step(torch.zeros((env.num_envs, env.action_dim), device=env.device))[4]
        for part, part_frames in frames.items():
            part_frames.append(extras["log"][ADD_LOG_KEY.format(part=part)].detach().cpu().numpy().astype(np.float64))
    add_by_part = {part: np.stack(part_frames, axis=1) for part, part_frames in frames.items()}
    return compute_add_stats(add_by_part, list(ADD_AUC_THRESHOLDS))["auc"]["overall"]


def _floor_score() -> Any:
    env = floor.build_env(HAND, CLIP, SEED)
    return floor.run_episode(env, floor.Policy.ZERO, SEED, HAND, CLIP)


@pytest.mark.sim
def test_the_floor_on_the_eval_env_matches_the_old_floor_protocol(
    genesis_cpu: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_hand_cfgs(monkeypatch)
    pristine_cfgs = floor.snapshot_hand_cfgs()
    old = _old_protocol_auc(OLD_PROTOCOL_NUM_ENVS)
    floor.restore_hand_cfgs(pristine_cfgs)
    floor_score = _floor_score()
    assert floor_score.auc_add == pytest.approx(old, abs=FLOOR_PROTOCOL_TOLERANCE)
    # The hand lets the box fall with a zero action; the record of that is the adapter's dropped flag.
    assert floor_score.drop_rate == 1.0


@pytest.mark.sim
def test_the_uniform_policy_scores_the_same_twice_with_one_seed(
    genesis_cpu: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_hand_cfgs(monkeypatch)
    env = floor.build_env(HAND, CLIP, SEED)
    first = floor.run_episode(env, floor.Policy.UNIFORM, SEED, HAND, CLIP)
    second = floor.run_episode(env, floor.Policy.UNIFORM, SEED, HAND, CLIP)
    assert first == second


def test_appending_to_a_csv_with_another_header_fails_loudly(tmp_path: Path) -> None:
    csv_path = tmp_path / floor.RUNS_CSV
    csv_path.write_text("hand,clip,policy,seed,num_envs,auc_overall,early_resets\nallegro_hand,box,zero,0,8,0.1,0\n")
    with pytest.raises(ValueError, match="use a new --out-dir"):
        floor.append_row(csv_path, _row())


def test_appending_rows_writes_the_header_once(tmp_path: Path) -> None:
    csv_path = tmp_path / floor.RUNS_CSV
    floor.append_row(csv_path, _row(seed=0))
    floor.append_row(csv_path, _row(seed=1))
    assert len(csv_path.read_text().splitlines()) == 3

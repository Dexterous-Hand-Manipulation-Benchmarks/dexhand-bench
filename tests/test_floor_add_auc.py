"""The floor script's aggregation (per-frame ADD to AUC) and its summary; no simulator."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("dexmachina.eval.compute_add", reason="needs genesis and dexmachina")

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
        hand=hand, clip="box", policy=policy, seed=seed, num_envs=2, auc_overall=auc, auc_top=auc, auc_bottom=auc,
        mean_add=mean_add, ep_length=200, early_resets=0, nan_envs=0, wall_seconds=1.0, error=error,
    )  # fmt: skip
    return floor.CellRow(**{**values, **overrides})


def test_auc_is_the_trapezoid_of_the_accuracy_curve_averaged_over_parts() -> None:
    # Thresholds 0.01..0.09 (9 points on [0, 1]). Top part: ADD 0.025 everywhere, so accuracy is 0 at 0.01 and 0.02 and
    # 1 from 0.03 on; the trapezoid over 8 equal intervals is 0.5 / 8 + 6 / 8 = 0.8125 (segment 0.02 -> 0.03 is 0.5).
    # Bottom part: ADD 0.5 everywhere, so accuracy is 0 and the AUC 0.
    frames = 4
    stats = floor.add_stats({"top": np.full((2, frames), 0.025), "bottom": np.full((2, frames), 0.5)})
    assert stats.auc_top == pytest.approx(0.8125)
    assert stats.auc_bottom == 0.0
    assert stats.auc_overall == pytest.approx(0.8125 / 2)
    assert stats.mean_add == pytest.approx((0.025 + 0.5) / 2)


def test_nan_add_counts_as_a_miss() -> None:
    stats = floor.add_stats({"top": np.array([[0.0, NAN]]), "bottom": np.array([[0.0, NAN]])})
    assert stats.auc_overall == pytest.approx(0.5)


def test_summary_gives_mean_and_min_max_over_seeds() -> None:
    text = floor.summarize([_row(seed=0, auc=0.2), _row(seed=1, auc=0.4), _row(seed=2, auc=0.9)])
    assert "| allegro_hand | box | zero | 3 | 0.5000 | 0.2000-0.9000 |" in text


def test_summary_lists_failed_cells_loudly_and_keeps_them_out_of_the_statistics() -> None:
    failed = _row(hand="xhand", seed=1, auc=NAN, mean_add=NAN, error="FileNotFoundError: no clip")
    text = floor.summarize([_row(auc=0.3), failed])
    assert "FAILED CELLS (1 of 2 runs)" in text
    assert "xhand / box / zero / seed 1: FileNotFoundError: no clip" in text
    assert "| allegro_hand | box | zero | 1 | 0.3000 |" in text


class ScriptedEnv:
    """Replays scripted per-step outputs: terminated and ADD per step, NaN flags per step (each shaped by env)."""

    action_dim = 2
    action_clip = 1.0
    device = "cpu"

    def __init__(self, terminated: list[list[bool]], nan_flags: list[list[bool]], add: float = 0.0) -> None:
        self.terminated = torch.tensor(terminated)
        self.nan_flags = torch.tensor(nan_flags)
        self.num_envs = self.terminated.shape[1]
        self.max_episode_length = self.terminated.shape[0]
        self.add = add
        self.nan_envs = torch.zeros(self.num_envs, dtype=torch.bool)
        self.steps_taken = 0

    def reset(self) -> None:
        self.steps_taken = 0

    def step(self, actions: torch.Tensor) -> tuple[None, None, torch.Tensor, None, dict[str, dict[str, torch.Tensor]]]:
        self.nan_envs = self.nan_flags[self.steps_taken]
        terminated = self.terminated[self.steps_taken]
        self.steps_taken += 1
        frame = torch.full((self.num_envs,), self.add)
        return None, None, terminated, None, {"log": {"add_top": frame, "add_bottom": frame}}


def test_early_resets_count_terminations_before_the_last_step_only() -> None:
    # 3 steps, 2 envs: step 0 has one termination, step 1 has two, the last step (a time-out) has two and is ignored.
    env = ScriptedEnv(terminated=[[True, False], [True, True], [True, True]], nan_flags=[[False, False]] * 3)
    result = floor.run_episode(env, floor.Policy.ZERO, seed=0)
    assert result.early_resets == 3
    assert result.ep_length == 3


def test_nan_envs_counts_envs_that_were_nan_at_any_step() -> None:
    # Env 0 is NaN at step 0 only, env 1 at steps 1 and 2, env 2 never: two distinct envs.
    env = ScriptedEnv(
        terminated=[[False] * 3] * 3, nan_flags=[[True, False, False], [False, True, False], [False, True, False]]
    )
    assert floor.run_episode(env, floor.Policy.ZERO, seed=0).nan_envs == 2


def test_run_cell_records_the_error_and_nan_metrics_when_the_env_cannot_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_build(hand: str, clip: str, seed: int, num_envs: int) -> None:
        raise FileNotFoundError("no retargeted clip")

    monkeypatch.setattr(floor, "build_env", failing_build)
    row = floor.run_cell("xhand", "box", floor.Policy.ZERO, 1, 2, {})
    assert row.error == "FileNotFoundError: no retargeted clip"
    assert (row.hand, row.clip, row.policy, row.seed, row.num_envs) == ("xhand", "box", "zero", 1, 2)
    assert np.isnan(row.auc_overall) and np.isnan(row.mean_add)
    assert row.ep_length == 0


def test_run_cell_fills_the_row_from_the_episode_when_the_env_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    # ADD 0 everywhere: every threshold is met, accuracy is 1, so the AUC is 1 and the mean ADD 0.
    env = ScriptedEnv(terminated=[[False, False], [True, False]], nan_flags=[[False, True]] * 2)
    monkeypatch.setattr(floor, "build_env", lambda hand, clip, seed, num_envs: env)
    row = floor.run_cell("allegro_hand", "box", floor.Policy.UNIFORM, 0, 2, {})
    assert row.error == ""
    assert row.auc_overall == pytest.approx(1.0)
    assert row.mean_add == 0.0
    assert (row.ep_length, row.early_resets, row.nan_envs) == (2, 0, 1)

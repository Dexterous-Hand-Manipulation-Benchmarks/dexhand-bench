"""The eval scorer (synthetic ADD values) and recorder (hand-made step records); no simulator, no env."""

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("dexmachina.envs.evaluation", reason="needs genesis and dexmachina")
from dexmachina.envs.add_metric import ADD_PARTS, ADDMetric  # noqa: E402  (after the skip: needs dexmachina)
from dexmachina.envs.evaluation import (  # noqa: E402
    NEVER_NAN,
    PART_AGGREGATIONS,
    EvalRecorder,
    StepRecord,
    Trajectory,
    load_trajectory,
    paired_demo_frame,
    save_trajectory,
    score,
)

NUM_FRAMES = 10
TOP, BOTTOM = ADD_PARTS.index("top"), ADD_PARTS.index("bottom")
NUM_THRESHOLDS = 9


def _trajectory(add: np.ndarray, nan_from: list[int] | None = None) -> Trajectory:
    """`add` is (steps, envs, parts); other arrays are zeros."""
    steps, envs, _ = add.shape
    return Trajectory(
        hand="h", clip="c", spec_hash="s", dexmachina_sha="d",
        object_root_pos=np.zeros((steps, envs, 3)), object_root_quat=np.zeros((steps, envs, 4)),
        object_joint_pos=np.zeros((steps, envs, 1)), add=add, dropped=np.zeros((steps, envs), dtype=bool),
        demo_frame=np.zeros((steps, envs), dtype=np.int64),
        nan_from=np.array(nan_from if nan_from is not None else [NEVER_NAN] * envs),
    )  # fmt: skip


def _constant_add(top: float, bottom: float, steps: int = NUM_FRAMES) -> np.ndarray:
    return np.broadcast_to(np.array([top, bottom]), (steps, 1, len(ADD_PARTS))).copy()


def test_all_adds_at_1_5cm_give_the_hand_computed_auc() -> None:
    # Hit at 8 of 9 thresholds (2..9 cm): accuracy 0 then 1 at x = 0, 1/8, ...: area 1/16 + 7/8.
    result = score(_trajectory(_constant_add(0.015, 0.015)))
    assert result.auc_add == pytest.approx(0.9375)
    assert result.success_2cm == 1.0


def test_an_add_exactly_on_a_threshold_is_a_miss_at_that_threshold() -> None:
    # Hit at 7 of 9 thresholds (3..9 cm): area (0 + 1) / 2 / 8 + 6 / 8.
    result = score(_trajectory(_constant_add(0.02, 0.02)))
    assert result.auc_add == pytest.approx(0.8125)
    assert result.success_2cm == 0.0


def test_the_part_aggregations_summarise_the_per_part_aucs() -> None:
    # Top at 1.5 cm gives 0.9375, bottom at 2 cm gives 0.8125 (the two tests above).
    result = score(_trajectory(_constant_add(0.015, 0.02)), part_aggregations=PART_AGGREGATIONS)
    assert result.auc_add_aggregated == pytest.approx({"median": 0.875, "min": 0.8125, "max": 0.9375, "std": 0.0625})
    assert result.auc_add == pytest.approx(0.875)


def test_the_preset_part_aggregations_on_three_parts() -> None:
    # Three values, so the median differs from the mean (with the two ADD parts they coincide).
    auc_by_part = {"a": 0.1, "b": 0.2, "c": 0.6}
    aggregated = {name: aggregate(auc_by_part) for name, aggregate in PART_AGGREGATIONS.items()}
    assert aggregated == pytest.approx({"median": 0.2, "min": 0.1, "max": 0.6, "std": (14 / 300) ** 0.5})


def test_a_custom_part_aggregation_gets_the_auc_of_each_part_by_name() -> None:
    result = score(
        _trajectory(_constant_add(0.015, 0.02)),
        part_aggregations={"top_minus_bottom": lambda auc: auc["top"] - auc["bottom"]},
    )
    assert result.auc_add_aggregated == pytest.approx({"top_minus_bottom": 0.125})


def test_without_part_aggregations_there_are_no_aggregated_values() -> None:
    assert score(_trajectory(_constant_add(0.015, 0.02))).auc_add_aggregated == {}


def test_frames_from_the_first_nan_step_are_misses() -> None:
    # 4 of 10 frames are scored (steps 0..3): the 1.5 cm AUC is scaled by 0.4.
    add = _constant_add(0.015, 0.015)
    add[4:] = 0.001  # would raise AUC and lower mean_add if the NaN frames were scored
    result = score(_trajectory(add, nan_from=[4]))
    assert result.auc_add == pytest.approx(0.4 * 0.9375)
    assert result.success_2cm == pytest.approx(0.4)
    assert result.nan_rate == 1.0
    assert result.mean_add == pytest.approx(0.015)


def test_nan_from_step_zero_scores_nothing_and_has_no_mean() -> None:
    result = score(_trajectory(np.full((NUM_FRAMES, 1, len(ADD_PARTS)), np.nan), nan_from=[0]))
    assert result.auc_add == 0.0
    assert np.isnan(result.mean_add)


def test_auc_per_part_then_mean_differs_from_auc_of_the_mean() -> None:
    # Top 1.5 cm (0.9375), bottom 9.5 cm (never below 9 cm: 0): mean 0.46875. The part mean 5.5 cm is below 6..9 cm:
    # accuracy 0 for 5 thresholds, then 1 for 4: area 0.5 / 8 + 3 / 8.
    result = score(_trajectory(_constant_add(0.015, 0.095)))
    assert result.auc_add == pytest.approx(0.46875)
    assert result.auc_add_of_mean == pytest.approx(0.4375)
    assert result.auc_add_part["top"] == pytest.approx(0.9375)
    assert result.auc_add_part["bottom"] == 0.0
    assert result.success_2cm == pytest.approx(0.5)


def test_a_drop_after_the_first_nan_step_is_not_counted() -> None:
    add = np.broadcast_to(np.array([0.015, 0.015]), (NUM_FRAMES, 2, 2)).copy()
    trajectory = _trajectory(add, nan_from=[3, NEVER_NAN])
    trajectory.dropped[5, 0] = True
    trajectory.dropped[2, 1] = True
    assert score(trajectory).drop_rate == 0.5


def test_nan_and_drop_rates_count_envs() -> None:
    add = np.broadcast_to(np.array([0.015, 0.015]), (NUM_FRAMES, 4, 2)).copy()
    trajectory = _trajectory(add, nan_from=[NEVER_NAN, 3, NEVER_NAN, NEVER_NAN])
    trajectory.dropped[5, 2] = True
    result = score(trajectory)
    assert result.nan_rate == 0.25
    assert result.drop_rate == 0.25


def _step(
    step: int,
    num_envs: int = 1,
    *,
    reset: bool | list[bool] | None = None,
    nan: list[bool] | None = None,
    **overrides: Any,
):
    """A record for step `step` of a NUM_FRAMES episode; ADD of every part equals step / 100."""
    last = step == NUM_FRAMES - 1
    values: dict[str, Any] = dict(
        object_root_pos=np.full((num_envs, 3), float(step)), object_root_quat=np.ones((num_envs, 4)),
        object_joint_pos=np.full((num_envs, 1), float(step)), add=np.full((num_envs, len(ADD_PARTS)), step / 100),
        nan_mask=np.array(nan if nan is not None else [False] * num_envs),
        reset_mask=np.broadcast_to(np.array(last if reset is None else reset), (num_envs,)).copy(),
        dropped=np.zeros(num_envs, dtype=bool),
        demo_frame=np.full(num_envs, paired_demo_frame(step, NUM_FRAMES), dtype=np.int64),
    )  # fmt: skip
    return StepRecord(**{**values, **overrides})


def _recorder(num_envs: int = 1) -> EvalRecorder:
    return EvalRecorder(num_steps=NUM_FRAMES, num_envs=num_envs, num_demo_frames=NUM_FRAMES)


def test_the_reset_at_the_last_step_passes_and_keeps_the_pre_reset_add() -> None:
    recorder = _recorder()
    for step in range(NUM_FRAMES):
        recorder.record(_step(step))
    trajectory = recorder.trajectory("h", "c", "s", "d")
    np.testing.assert_allclose(trajectory.add[-1, 0], [0.09, 0.09])
    assert trajectory.nan_from.tolist() == [NEVER_NAN]


def test_a_reset_before_the_last_step_raises_with_step_and_env() -> None:
    recorder = _recorder(num_envs=2)
    recorder.record(_step(0, num_envs=2))
    with pytest.raises(ValueError, match=r"env 1 had reset=True at step 1.*early_reset_threshold"):
        recorder.record(_step(1, num_envs=2, reset=[False, True]))


def test_a_missing_reset_at_the_last_step_raises() -> None:
    recorder = _recorder()
    for step in range(NUM_FRAMES - 1):
        recorder.record(_step(step))
    with pytest.raises(ValueError, match=f"reset=False at step {NUM_FRAMES - 1}"):
        recorder.record(_step(NUM_FRAMES - 1, reset=False))


def test_a_nan_step_sets_nan_from_without_raising_even_when_the_env_resets() -> None:
    recorder = _recorder()
    for step in range(NUM_FRAMES):
        recorder.record(_step(step, nan=[step >= 3], reset=step >= 3))
    trajectory = recorder.trajectory("h", "c", "s", "d")
    assert trajectory.nan_from.tolist() == [3]
    assert not np.isnan(trajectory.add[:3]).any()
    assert np.isnan(trajectory.add[3:]).all()


def test_a_cleared_nan_flag_gives_the_same_trajectory_as_a_flag_that_stays_set() -> None:
    def run(flag_stays: bool) -> Trajectory:
        recorder = _recorder()
        for step in range(NUM_FRAMES):
            first = step == 3
            flag = step >= 3 if flag_stays else first
            reset = step >= 3 if flag_stays else first or step == NUM_FRAMES - 1
            garbage: dict[str, Any] = {} if flag_stays or step <= 3 else dict(add=np.full((1, len(ADD_PARTS)), 7.0))
            recorder.record(_step(step, nan=[flag], reset=reset, **garbage))
        return recorder.trajectory("h", "c", "s", "d")

    stays, cleared = run(True), run(False)
    for field in fields(Trajectory):
        np.testing.assert_array_equal(getattr(stays, field.name), getattr(cleared, field.name), err_msg=field.name)


def test_a_demo_frame_from_the_old_pairing_raises() -> None:
    recorder = _recorder()
    with pytest.raises(ValueError, match="paired demo frame 0 with step 0, expected 1"):
        recorder.record(_step(0, demo_frame=np.array([0])))


def test_a_trajectory_needs_every_step_and_no_step_beyond_the_episode() -> None:
    recorder = _recorder()
    recorder.record(_step(0))
    with pytest.raises(ValueError, match="1 of 10 steps"):
        recorder.trajectory("h", "c", "s", "d")
    full = _recorder()
    for step in range(NUM_FRAMES):
        full.record(_step(step))
    with pytest.raises(ValueError, match="step 10 was recorded"):
        full.record(_step(0))


class _FakePart:
    """Part pose at the origin; the demo pose of frame i is at x = i, so ADD equals the demo frame index."""

    device = torch.device("cpu")

    def sample_mesh_vertices(self, num_samples: int, part: str, seed: int) -> torch.Tensor:
        return torch.zeros(1, 3)

    def get_part_pose(self, part: str) -> torch.Tensor:
        return torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])


def test_the_paired_demo_frame_is_the_index_add_metric_uses() -> None:
    metric = ADDMetric(_FakePart())
    demo = torch.zeros(NUM_FRAMES, 7)
    demo[:, 3] = 1.0
    demo[:, 0] = torch.arange(NUM_FRAMES, dtype=torch.float32)
    metric.demo_part_pose = {part: list(demo) for part in ADD_PARTS}
    metric.finalize()
    for step in range(NUM_FRAMES):
        used_frame = metric.compute(torch.tensor([step + 1]))["add"].item()
        assert used_frame == paired_demo_frame(step, NUM_FRAMES)
    assert (
        paired_demo_frame(NUM_FRAMES - 1, NUM_FRAMES) == paired_demo_frame(NUM_FRAMES - 2, NUM_FRAMES) == NUM_FRAMES - 1
    )


def test_a_nan_env_may_restart_its_demo_frames_after_the_nan_step() -> None:
    recorder = _recorder(num_envs=2)
    for step in range(NUM_FRAMES):
        paired = paired_demo_frame(step, NUM_FRAMES)
        frames = np.array([paired if step <= 3 else 0, paired])
        record = _step(step, num_envs=2, nan=[step >= 3, False], reset=[step >= 3, step == NUM_FRAMES - 1])
        recorder.record(replace(record, demo_frame=frames))
    assert recorder.trajectory("h", "c", "s", "d").nan_from.tolist() == [3, NEVER_NAN]


def test_the_nan_step_itself_must_still_pair_the_right_frame() -> None:
    recorder = _recorder()
    with pytest.raises(ValueError, match="paired demo frame 0 with step 0"):
        recorder.record(_step(0, nan=[True], demo_frame=np.array([0])))


def test_the_recorder_stores_copies_of_the_env_buffers() -> None:
    recorder = _recorder()
    record = _step(0)
    recorder.record(record)
    record.add[:] = 5.0
    for step in range(1, NUM_FRAMES):
        recorder.record(_step(step))
    assert recorder.trajectory("h", "c", "s", "d").add[0, 0, 0] == 0.0


def test_a_record_with_the_wrong_number_of_envs_raises() -> None:
    with pytest.raises(ValueError, match=r"object_root_pos has 2 rows, expected one per env \(1\)"):
        _recorder().record(_step(0, num_envs=1, object_root_pos=np.zeros((2, 3))))


def test_a_saved_trajectory_loads_back_equal_and_scores_the_same(tmp_path: Path) -> None:
    trajectory = _trajectory(_constant_add(0.015, 0.03, steps=4), nan_from=[2])
    path = tmp_path / "trajectory.npz"
    save_trajectory(path, trajectory)
    loaded = load_trajectory(path)
    for field in fields(Trajectory):
        np.testing.assert_array_equal(getattr(loaded, field.name), getattr(trajectory, field.name), err_msg=field.name)
    assert (loaded.hand, loaded.clip, loaded.spec_hash, loaded.dexmachina_sha) == ("h", "c", "s", "d")
    assert type(loaded.hand) is str
    assert score(loaded) == score(trajectory)


def test_a_file_that_is_not_a_trajectory_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.npz"
    np.savez(path, add=np.zeros(3))
    with pytest.raises(ValueError, match="not a saved Trajectory"):
        load_trajectory(path)


def test_a_saved_trajectory_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.npz"
    save_trajectory(path, _trajectory(_constant_add(0.01, 0.01, steps=2)))
    with pytest.raises(FileExistsError, match="exists already"):
        save_trajectory(path, _trajectory(_constant_add(0.05, 0.05, steps=2)))
    assert load_trajectory(path).add.max() == 0.01

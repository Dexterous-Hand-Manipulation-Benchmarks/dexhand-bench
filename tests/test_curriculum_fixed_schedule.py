"""The fixed curriculum schedule: gains are a function of the fraction of the env step budget used.

dexmachina counts the env steps of training envs itself, so the gains do not depend on how often a method calls
``set_curriculum``, on its number of envs or on its update ratio. Evaluation steps never move the count.
"""

import json
import math
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from conftest import isolate_hand_cfgs

pytest.importorskip("dexmachina.envs.curriculum", reason="needs dexmachina")
from dexmachina.envs import curriculum  # noqa: E402  (after the skip: needs dexmachina)
from dexmachina.envs.base_env import BaseEnv  # noqa: E402
from dexmachina.envs.benchmark_task import BENCHMARK_CLIPS, BENCHMARK_TASK, RunChoices, benchmark_spec  # noqa: E402
from dexmachina.envs.factory import make_env, task_hash  # noqa: E402
from dexmachina.envs.run_spec import SPEC_KEY, load_run_spec, save_run_spec  # noqa: E402

BUDGET = 1_000_000
NUM_ENVS = 8
KP_INIT = 80.0
START_STEP = 10_000  # 1 % of BUDGET
ZERO_STEP = 600_000  # 60 % of BUDGET
UPPER_RATIO = 0.9
LOWER_RATIO = 0.8
CALL_SIZES = (NUM_ENVS, 2 * NUM_ENVS, 13 * NUM_ENVS)
CHECK_EVERY = 26 * NUM_ENVS  # a common multiple of CALL_SIZES
EPISODE_LENGTH = 200
BENCHMARK_DECAY_COUNT = 71  # the kp decays of the benchmark's 80 * 0.9**n: 80 * 0.9**71 < 0.05 <= 80 * 0.9**70
GATED_DECAY_EPOCH = 100
SMALL_BUDGET = 400  # the first decay (1 %) is after 4 env steps


class _StubObject:
    device = "cpu"

    def set_joint_gains(self, **kwargs: object) -> None:
        pass


def _make_curriculum(**changes: object) -> curriculum.Curriculum:
    cfg = curriculum.get_curriculum_cfg(
        dict(
            schedule="fixed", fixed_mode="uniform", uniform_mode="slow", kp_init=KP_INIT, kv_init=5.0,
            budget_env_steps=BUDGET, deque_len=3, skip_grad=True, zero_epoch=10**6,
            upper_ratios=dict(kp=UPPER_RATIO, kv=UPPER_RATIO, fr=1.0),
            lower_ratios=dict(kp=LOWER_RATIO, kv=LOWER_RATIO, fr=1.0),
        )
        | changes
    )
    return curriculum.Curriculum(
        cfg, _StubObject(), ["task"], num_envs=NUM_ENVS, achieved_length=EPISODE_LENGTH,
        max_episode_length=EPISODE_LENGTH,
    )


def _gains_by_step(call_size: int, total: int) -> dict[int, dict[str, float]]:
    """The gains after every CHECK_EVERY env steps, when the count is advanced in calls of `call_size` steps."""
    curr = _make_curriculum()
    gains: dict[int, dict[str, float]] = {}
    for steps in range(call_size, total + 1, call_size):
        curr.advance_env_steps(call_size)
        if steps % CHECK_EVERY == 0:
            gains[steps] = curr.get_current_gains()
    return gains


def test_gains_at_an_env_step_do_not_depend_on_the_call_size() -> None:
    by_call_size = [_gains_by_step(size, ZERO_STEP + CHECK_EVERY) for size in CALL_SIZES]
    assert len(by_call_size[0]) == (ZERO_STEP + CHECK_EVERY) // CHECK_EVERY
    assert by_call_size[0] == by_call_size[1] == by_call_size[2]


def test_epoch_calls_do_not_move_the_fixed_gains() -> None:
    curr = _make_curriculum(zero_epoch=0, wait_epochs=0)
    curr.advance_env_steps(START_STEP)
    before = curr.get_current_gains()
    for epoch in (0, 1, 10_000):
        for _ in range(3):
            curr.update_progress({"task": 0.0}, 0)
        assert curr.set_curriculum(epoch)[1] is False
    assert curr.get_current_gains() == before


def test_the_gain_is_the_start_value_before_one_percent() -> None:
    curr = _make_curriculum()
    curr.advance_env_steps(START_STEP - NUM_ENVS)
    assert curr.get_current_gains()["kp"] == KP_INIT
    curr.advance_env_steps(NUM_ENVS)
    assert curr.get_current_gains()["kp"] == pytest.approx(KP_INIT * UPPER_RATIO)


def test_the_gain_is_exactly_zero_from_sixty_percent_and_never_dials_back() -> None:
    curr = _make_curriculum()
    previous = KP_INIT
    for steps in range(NUM_ENVS, 2 * ZERO_STEP, NUM_ENVS):
        curr.advance_env_steps(NUM_ENVS)
        curr.update_progress({"task": 0.0}, 0)
        curr.set_curriculum(steps // NUM_ENVS)
        gains = curr.get_current_gains()
        assert gains["kp"] <= previous and gains["kp_lower"] <= gains["kp"]
        previous = gains["kp"]
        if steps < ZERO_STEP:
            assert gains["kp"] > 0
        else:
            assert {name: gains[name] for name in ("kp", "kv", "kp_lower", "kv_lower")} == dict.fromkeys(
                ("kp", "kv", "kp_lower", "kv_lower"), 0.0
            )


def test_the_decays_are_71_equal_steps_and_the_last_one_reaches_zero() -> None:
    curr = _make_curriculum()
    assert len(curr.fixed_decay_steps) == BENCHMARK_DECAY_COUNT
    assert curr.fixed_decay_steps[0] == START_STEP
    assert curr.fixed_decay_steps[-1] == ZERO_STEP
    exact_spacing = (ZERO_STEP - START_STEP) / (BENCHMARK_DECAY_COUNT - 1)
    for before, after in zip(curr.fixed_decay_steps, curr.fixed_decay_steps[1:], strict=False):
        assert after - before == pytest.approx(exact_spacing, abs=1)
    for decays in range(1, BENCHMARK_DECAY_COUNT):
        curr.advance_env_steps(curr.fixed_decay_steps[decays - 1] - curr.env_steps)
        assert curr.get_current_gains()["kp"] == pytest.approx(KP_INIT * UPPER_RATIO**decays)
    assert KP_INIT * UPPER_RATIO**70 > curriculum.KP_ZERO_THRESHOLD > KP_INIT * UPPER_RATIO**71


def test_the_decay_count_follows_the_start_gain_and_the_ratio() -> None:
    for kp_init, ratio in ((8.0, UPPER_RATIO), (KP_INIT, 0.5)):
        expected = math.ceil(math.log(curriculum.KP_ZERO_THRESHOLD / kp_init) / math.log(ratio))
        curr = _make_curriculum(kp_init=kp_init, upper_ratios=dict(kp=ratio, kv=UPPER_RATIO, fr=1.0))
        assert len(curr.fixed_decay_steps) == expected


def test_the_fixed_schedule_needs_a_budget() -> None:
    with pytest.raises(ValueError, match="budget_env_steps"):
        _make_curriculum(budget_env_steps=None)


def test_the_gated_schedule_still_decays_on_epochs_and_counts_env_steps_without_acting_on_them() -> None:
    curr = _make_curriculum(schedule="uniform", wait_epochs=0, budget_env_steps=None)
    curr.advance_env_steps(10**9)
    assert curr.get_current_gains()["kp"] == KP_INIT
    for epoch in range(GATED_DECAY_EPOCH):
        for _ in range(3):
            curr.update_progress({"task": 1.0}, EPISODE_LENGTH)
        curr.set_curriculum(epoch)
    assert curr.get_current_gains()["kp"] < KP_INIT


def test_the_benchmark_task_uses_the_fixed_schedule_with_the_agreed_values() -> None:
    assert BENCHMARK_TASK.curr_schedule == "fixed"
    assert BENCHMARK_TASK.budget_env_steps == 50_000_000
    assert BENCHMARK_TASK.fixed_start_fraction == 0.01
    assert BENCHMARK_TASK.fixed_zero_fraction == 0.6


def test_the_saved_env_spec_records_the_schedule_and_the_task_hash_leaves_it_to_that_record(tmp_path: Path) -> None:
    spec = benchmark_spec(RunChoices(hand="allegro_hand", clip=BENCHMARK_CLIPS[0], seed=0, num_envs=2))
    path = save_run_spec(tmp_path, spec, {})
    saved = json.loads(path.read_text())[SPEC_KEY]
    assert (saved["curr_schedule"], saved["budget_env_steps"]) == ("fixed", 50_000_000)
    assert (saved["fixed_start_fraction"], saved["fixed_zero_fraction"]) == (0.01, 0.6)
    assert load_run_spec(tmp_path).spec == spec
    assert task_hash(spec) == task_hash(replace(spec, fixed_zero_fraction=0.5, budget_env_steps=1))


@pytest.fixture(scope="module")
def training_env(genesis_cpu: None) -> Iterator[BaseEnv]:
    spec = benchmark_spec(RunChoices(hand="allegro_hand", clip=BENCHMARK_CLIPS[0], seed=0, num_envs=2))
    with pytest.MonkeyPatch.context() as monkeypatch:
        isolate_hand_cfgs(monkeypatch)
        yield make_env(spec, "cpu", {})


@pytest.mark.sim
def test_only_training_steps_move_the_env_step_clock(training_env: BaseEnv) -> None:
    env = training_env
    assert isinstance(env.curriculum, curriculum.Curriculum)
    clock = env.curriculum
    env.reset()
    actions = torch.zeros((env.num_envs, env.action_dim), device=env.device)
    start = clock.env_steps
    env.step(actions)
    assert clock.env_steps == start + env.num_envs
    env.record_curriculum = False
    env.step(actions)
    env.record_curriculum = True
    assert clock.env_steps == start + env.num_envs


@pytest.fixture
def small_budget_env(genesis_cpu: None) -> Iterator[BaseEnv]:
    spec = benchmark_spec(RunChoices(hand="allegro_hand", clip=BENCHMARK_CLIPS[0], seed=0, num_envs=2))
    with pytest.MonkeyPatch.context() as monkeypatch:
        isolate_hand_cfgs(monkeypatch)
        yield make_env(replace(spec, budget_env_steps=SMALL_BUDGET), "cpu", {})


@pytest.mark.sim
def test_the_first_fixed_decay_changes_the_gains_the_object_runs_with(small_budget_env: BaseEnv) -> None:
    env = small_budget_env
    assert isinstance(env.curriculum, curriculum.Curriculum) and env.object is not None
    clock = env.curriculum
    assert clock.fixed_decay_steps[0] == SMALL_BUDGET // 100
    env.reset()
    kp_before = env.object.entity.get_dofs_kp().clone()
    assert torch.all(kp_before == KP_INIT)
    actions = torch.zeros((env.num_envs, env.action_dim), device=env.device)
    while clock.env_steps < clock.fixed_decay_steps[0]:
        env.step(actions)
    kp_after = env.object.entity.get_dofs_kp()
    assert torch.all(kp_after <= KP_INIT * UPPER_RATIO) and torch.all(kp_after > 0)

"""A NaN env is flagged for the step that produced it only: resetting the env clears the flag."""

from collections.abc import Callable, Iterator
from dataclasses import fields, replace
from typing import Any

import numpy as np
import pytest
import torch

from conftest import isolate_hand_cfgs

pytest.importorskip("dexmachina.envs.factory", reason="needs genesis and dexmachina")
import dexmachina.envs.factory as factory  # noqa: E402  (after the skip: needs dexmachina)
from dexmachina.envs.base_env import BaseEnv  # noqa: E402
from dexmachina.envs.benchmark_task import BENCHMARK_CLIPS, RunChoices, benchmark_spec  # noqa: E402
from dexmachina.envs.evaluation import Trajectory, rollout_episode  # noqa: E402
from dexmachina.envs.spec import EnvSpec  # noqa: E402

pytestmark = pytest.mark.sim

CPU = "cpu"
HAND = "allegro_hand"
CLIP = BENCHMARK_CLIPS[0]
NUM_ENVS = 2
POISONED_ENV = 0
NAN_STEP = 3
NAN_REWARD = -1.0


def _spec(**changes: Any) -> EnvSpec:
    spec = benchmark_spec(RunChoices(hand=HAND, clip=CLIP, seed=0, num_envs=NUM_ENVS))
    return replace(spec, rand_init_ratio=0.0, **changes)


def _poison_object_position(env: BaseEnv, steps: Callable[[], bool]) -> Callable[[], None]:
    """Makes the object of POISONED_ENV read NaN from the sim whenever `steps()` is true; returns the undo."""
    obj = env.object
    assert obj is not None
    update_value_buffers = obj.update_value_buffers

    def poisoned_update() -> Any:
        update_value_buffers()
        if steps():
            obj.root_pos[POISONED_ENV] = torch.nan

    obj.update_value_buffers = poisoned_update
    return lambda: setattr(obj, "update_value_buffers", update_value_buffers)


@pytest.fixture(scope="module")
def training_env(genesis_cpu: None) -> Iterator[BaseEnv]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        isolate_hand_cfgs(monkeypatch)
        yield factory.make_env(_spec(), CPU, {})


def test_the_benchmark_task_clears_the_nan_flag_on_reset(training_env: BaseEnv) -> None:
    assert training_env.clear_nan_on_reset


def test_a_reset_clears_the_nan_flag_so_the_next_episode_is_not_cut(training_env: BaseEnv) -> None:
    env = training_env
    env.reset()
    zero_actions = torch.zeros((NUM_ENVS, env.action_dim), device=env.device)
    nan_now = [False]
    undo = _poison_object_position(env, lambda: nan_now[0])
    try:
        nan_now[0] = True
        env.step(zero_actions)
        nan_now[0] = False
        assert env.rew_buf[POISONED_ENV].item() == NAN_REWARD, "the step that produced NaN is the -1 reward step"
        assert env.reset_buf[POISONED_ENV], "a NaN env ends its episode"
        assert not env.nan_envs.any(), "the reset must clear the flag"
        env.step(zero_actions)
        assert not env.reset_buf[POISONED_ENV], "the new episode must not be cut after one step"
        assert env.rew_buf[POISONED_ENV].item() != NAN_REWARD
        assert env.episode_length_buf[POISONED_ENV] == 1, "the new episode has run exactly one step"
    finally:
        undo()


def test_with_the_option_off_the_nan_flag_stays_set_after_the_reset(genesis_cpu: None) -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        isolate_hand_cfgs(monkeypatch)
        env = factory.make_env(_spec(clear_nan_on_reset=False), CPU, {})
    env.reset()
    zero_actions = torch.zeros((NUM_ENVS, env.action_dim), device=env.device)
    obj = env.object
    assert obj is not None
    update_value_buffers = obj.update_value_buffers

    def poison_once() -> None:
        update_value_buffers()
        obj.root_pos[POISONED_ENV] = torch.nan
        obj.update_value_buffers = update_value_buffers

    obj.update_value_buffers = poison_once
    env.step(zero_actions)
    assert env.nan_envs[POISONED_ENV], "the flag stays set after the reset"
    env.step(zero_actions)
    assert env.reset_buf[POISONED_ENV], "and cuts the next episode after one step"


@pytest.fixture(scope="module")
def eval_envs(genesis_cpu: None) -> Iterator[tuple[BaseEnv, BaseEnv]]:
    """An eval env that gets a NaN episode, and a fresh one that never does."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        isolate_hand_cfgs(monkeypatch)
        yield factory.make_eval_env(_spec(), CPU), factory.make_eval_env(_spec(), CPU)


def _zero_action_rollout(env: BaseEnv) -> Trajectory:
    def zero_actions(_observation: torch.Tensor) -> torch.Tensor:
        return torch.zeros((env.num_envs, env.action_dim), device=env.device)

    return rollout_episode(env, zero_actions, HAND, CLIP, spec_hash="", dexmachina_sha="")


def test_an_eval_env_scores_an_episode_after_a_nan_episode_like_a_fresh_one(
    eval_envs: tuple[BaseEnv, BaseEnv],
) -> None:
    reused, fresh = eval_envs
    calls = [0]

    def nan_at_step() -> bool:
        calls[0] += 1
        return calls[0] == NAN_STEP

    undo = _poison_object_position(reused, nan_at_step)
    try:
        nan_episode = _zero_action_rollout(reused)
    finally:
        undo()
    assert nan_episode.nan_from[POISONED_ENV] < reused.max_episode_length, "the first episode must hit NaN"

    after_nan = _zero_action_rollout(reused)
    reference = _zero_action_rollout(fresh)
    for field in fields(Trajectory):
        np.testing.assert_array_equal(
            getattr(after_nan, field.name), getattr(reference, field.name), err_msg=field.name
        )

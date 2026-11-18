"""make_eval_env: the env settings the evaluation protocol needs, and a rollout through it."""

from collections.abc import Iterator
from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch

from conftest import isolate_hand_cfgs

pytest.importorskip("dexmachina.envs.factory", reason="needs genesis and dexmachina")
import dexmachina.envs.factory as factory  # noqa: E402  (after the skip: needs dexmachina)
from dexmachina.envs.add_metric import ADD_PARTS  # noqa: E402
from dexmachina.envs.base_env import BaseEnv  # noqa: E402
from dexmachina.envs.benchmark_task import BENCHMARK_CLIPS, RunChoices, benchmark_spec  # noqa: E402
from dexmachina.envs.evaluation import EnvReset, EnvStep, Trajectory, rollout_episode  # noqa: E402
from dexmachina.envs.spec import EnvSpec  # noqa: E402
from dexmachina.eval.compute_add import create_scene, get_all_add, load_part_verts  # noqa: E402

CPU = "cpu"
HAND = "allegro_hand"
CLIP = BENCHMARK_CLIPS[0]
OBJECT_NAME = "box"
TRAINING_NUM_ENVS = 2
TRAINING_RAND_INIT_RATIO = 0.3
ADD_TOLERANCE = 1e-5
POLICY_SEED = 7
REWARD_CFG = {"contact_rew_weight": 0.0, "imi_rew_weight": 0.0, "bc_rew_weight": 0.0}

FloatArray = npt.NDArray[np.float64]


def _training_spec(**changes: Any) -> EnvSpec:
    """The benchmark task as a training run builds it, with settings that an eval env must switch off."""
    spec = benchmark_spec(RunChoices(hand=HAND, clip=CLIP, seed=0, num_envs=TRAINING_NUM_ENVS))
    return replace(spec, use_rand=True, rand_init_ratio=TRAINING_RAND_INIT_RATIO, **changes)


def _eval_env_state(**changes: Any) -> factory.AppliedEvalConfig:
    return replace(factory.EVAL_CONFIG, **changes)


def _fake_env(applied: factory.AppliedEvalConfig) -> SimpleNamespace:
    """Stands in for a BaseEnv that has the given eval settings."""
    return SimpleNamespace(
        is_eval=applied.is_eval,
        early_reset_threshold=applied.early_reset_threshold,
        rand_init_ratio=applied.rand_init_ratio,
        rand_cfg={"randomize": applied.randomized},
        num_envs=applied.num_envs,
        object=SimpleNamespace(actuated=applied.object_actuated),
        add_metric=SimpleNamespace(ready=applied.add_metric_ready),
        record_curriculum=applied.record_curriculum,
    )


class _FakeBuild:
    """Replaces what make_eval_env builds with: records the spec and the env kwargs, returns `env`."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, env: SimpleNamespace) -> None:
        self.specs: list[EnvSpec] = []
        self.env_kwargs: dict[str, Any] = {"env_cfg": {}, "rand_cfg": {"randomize": True}, "reward_cfg": REWARD_CFG}

        def build_env_kwargs(spec: EnvSpec, device: str, reward_overrides: Any) -> dict[str, Any]:
            self.specs.append(spec)
            return self.env_kwargs

        monkeypatch.setattr(factory, "build_env_kwargs", build_env_kwargs)
        monkeypatch.setattr(factory, "BaseEnv", lambda **kwargs: env)
        monkeypatch.setattr(factory, "verify_applied_config", lambda env, expected: None)


def test_a_maniptrans_spec_is_refused() -> None:
    with pytest.raises(ValueError, match="does not support ManipTrans"):
        factory.make_eval_env(replace(_training_spec(), maniptrans=True, actuate_object=False), CPU)


def test_a_spec_without_rl_games_is_refused() -> None:
    with pytest.raises(ValueError, match="use_rl_games=True"):
        factory.make_eval_env(_training_spec(use_rl_games=False), CPU)


def test_seeds_env_counts_and_training_lengths_share_the_task_hash() -> None:
    assert factory.task_hash(_training_spec()) == factory.task_hash(
        _training_spec(seed=7, num_envs=TRAINING_NUM_ENVS * 2, max_epochs=123)
    )


def test_numbers_written_as_ints_or_floats_share_the_task_hash() -> None:
    assert factory.task_hash(_training_spec(task_rew_betas=(10, 1, 5))) == factory.task_hash(
        _training_spec(task_rew_betas=(10.0, 1.0, 5.0))
    )


def test_another_clip_is_another_task() -> None:
    assert factory.task_hash(_training_spec()) != factory.task_hash(_training_spec(clip="mixer-0-150-s01-01"))


def test_the_training_spec_is_replaced_by_the_eval_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    build = _FakeBuild(monkeypatch, _fake_env(factory.EVAL_CONFIG))
    factory.make_eval_env(_training_spec(), CPU)
    (spec,) = build.specs
    assert spec.num_envs == 1
    assert not spec.actuate_object
    assert spec.rand_init_ratio == 0.0
    assert spec.early_reset_threshold == 0.0


def test_the_env_kwargs_enable_eval_mode_and_the_add_metric_and_disable_randomisation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = _FakeBuild(monkeypatch, _fake_env(factory.EVAL_CONFIG))
    factory.make_eval_env(_training_spec(), CPU)
    assert build.env_kwargs == {
        "env_cfg": {"is_eval": True, "log_add_metric": True},
        "rand_cfg": {"randomize": False},
        "reward_cfg": REWARD_CFG,
    }


def test_the_curriculum_is_switched_off_after_the_build(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _fake_env(_eval_env_state(record_curriculum=True))
    _FakeBuild(monkeypatch, env)
    factory.make_eval_env(_training_spec(), CPU)
    assert env.record_curriculum is False


# A value that the eval protocol does not accept, per setting a build can get wrong (make_eval_env sets
# record_curriculum itself, after the build).
WRONG_AFTER_BUILD = {
    "is_eval": False,
    "early_reset_threshold": 0.6,
    "rand_init_ratio": 0.3,
    "randomized": True,
    "num_envs": 2,
    "object_actuated": True,
    "add_metric_ready": False,
}


@pytest.mark.parametrize("name, wrong_value", WRONG_AFTER_BUILD.items())
def test_a_built_env_that_misses_an_eval_setting_is_refused_with_that_setting_named(
    monkeypatch: pytest.MonkeyPatch, name: str, wrong_value: object
) -> None:
    _FakeBuild(monkeypatch, _fake_env(_eval_env_state(**{name: wrong_value})))
    with pytest.raises(ValueError, match=rf"{name}: expected .*, applied {wrong_value}"):
        factory.make_eval_env(_training_spec(), CPU)


@pytest.fixture(scope="module")
def eval_env(genesis_cpu: None) -> Iterator[BaseEnv]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        isolate_hand_cfgs(monkeypatch)
        yield factory.make_eval_env(_training_spec(), CPU)


@pytest.fixture(scope="module")
def kinematic_eval_env(genesis_cpu: None) -> Iterator[BaseEnv]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        isolate_hand_cfgs(monkeypatch)
        yield factory.make_eval_env(_training_spec(action_mode="kinematic"), CPU)


@pytest.mark.sim
def test_the_eval_env_is_in_eval_mode(eval_env: BaseEnv) -> None:
    assert eval_env.is_eval


@pytest.mark.sim
def test_the_eval_env_has_no_early_reset_and_no_random_initial_state(eval_env: BaseEnv) -> None:
    assert eval_env.early_reset_threshold == 0.0
    assert eval_env.rand_init_ratio == 0.0


@pytest.mark.sim
def test_the_eval_env_has_one_env(eval_env: BaseEnv) -> None:
    assert eval_env.num_envs == 1


@pytest.mark.sim
def test_the_eval_env_object_is_not_actuated(eval_env: BaseEnv) -> None:
    assert eval_env.object is not None
    assert not eval_env.object.actuated


@pytest.mark.sim
def test_the_eval_env_has_the_online_add_metric_ready(eval_env: BaseEnv) -> None:
    assert eval_env.add_metric is not None
    assert eval_env.add_metric.ready


@pytest.mark.sim
def test_the_eval_env_is_not_randomised(eval_env: BaseEnv) -> None:
    assert not eval_env.rand_cfg["randomize"]


@pytest.mark.sim
def test_the_eval_env_does_not_feed_the_curriculum(eval_env: BaseEnv) -> None:
    assert not eval_env.record_curriculum


def _rollout(env: BaseEnv, policy_seed: int) -> tuple[Trajectory, FloatArray]:
    """One eval episode with seeded uniform actions; returns the recorded trajectory and the demo states."""
    generator = torch.Generator(device=env.device).manual_seed(policy_seed)

    def uniform_actions(_observation: torch.Tensor) -> torch.Tensor:
        unit = torch.rand((env.num_envs, env.action_dim), generator=generator, device=env.device) * 2.0 - 1.0
        return unit * env.action_clip

    trajectory = rollout_episode(env, uniform_actions, HAND, CLIP, spec_hash="", dexmachina_sha="")
    assert env.object is not None
    return trajectory, env.object.demo_states.detach().cpu().numpy().astype(np.float64)


def _offline_add(trajectory: Trajectory, demo_states: FloatArray) -> FloatArray:
    """dexmachina's offline ADD of the stored states against demo frame t + 1 (clamped), shape (steps, parts)."""
    num_steps = trajectory.add.shape[0]
    paired_demo = demo_states[np.minimum(np.arange(num_steps) + 1, demo_states.shape[0] - 1)]
    states = np.concatenate(
        [trajectory.object_root_pos, trajectory.object_root_quat, trajectory.object_joint_pos], axis=-1
    )
    device = torch.device(CPU)
    _, obj = create_scene(OBJECT_NAME, device, num_envs=num_steps)
    part_verts = load_part_verts(obj, OBJECT_NAME, device)
    offline = get_all_add(
        obj,
        part_verts,
        torch.tensor(paired_demo, dtype=torch.float32),
        torch.tensor(states, dtype=torch.float32),
    )
    return np.stack([offline[part][0] for part in ADD_PARTS], axis=-1)


@pytest.mark.sim
def test_the_recorded_add_is_the_offline_add_of_the_stored_states_paired_with_the_next_demo_frame(
    kinematic_eval_env: BaseEnv,
) -> None:
    trajectory, demo_states = _rollout(kinematic_eval_env, POLICY_SEED)
    np.testing.assert_allclose(trajectory.add[:, 0, :], _offline_add(trajectory, demo_states), atol=ADD_TOLERANCE)


@pytest.mark.sim
def test_two_evals_of_one_policy_give_identical_trajectories(eval_env: BaseEnv) -> None:
    first, _ = _rollout(eval_env, POLICY_SEED)
    second, _ = _rollout(eval_env, POLICY_SEED)
    for field in fields(Trajectory):
        np.testing.assert_array_equal(getattr(first, field.name), getattr(second, field.name), err_msg=field.name)


@pytest.mark.sim
def test_the_policy_gets_the_observation_the_last_reset_or_step_returned(
    eval_env: BaseEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    returned: list[torch.Tensor] = []
    received: list[torch.Tensor] = []
    env_reset = cast(EnvReset, eval_env.reset)
    env_step = cast(EnvStep, eval_env.step)

    def recording_reset() -> Any:
        observations, info = env_reset()
        returned.append(observations["policy"].clone())
        return observations, info

    def recording_step(actions: torch.Tensor) -> Any:
        observations, *rest = env_step(actions)
        returned.append(observations["policy"].clone())
        return observations, *rest

    def zero_actions(observation: torch.Tensor) -> torch.Tensor:
        received.append(observation.clone())
        return torch.zeros((eval_env.num_envs, eval_env.action_dim), device=eval_env.device)

    monkeypatch.setattr(eval_env, "reset", recording_reset)
    monkeypatch.setattr(eval_env, "step", recording_step)
    rollout_episode(eval_env, zero_actions, HAND, CLIP, spec_hash="", dexmachina_sha="")
    assert len(received) == len(returned) - 1
    for step, (observation, expected) in enumerate(zip(received, returned)):
        torch.testing.assert_close(observation, expected, msg=f"step {step}")

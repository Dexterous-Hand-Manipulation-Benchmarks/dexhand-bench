"""dexmachina's make_env builds from an EnvSpec and raises when the built env differs from it."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("dexmachina.envs.constructors", reason="needs genesis and dexmachina")
import dexmachina.envs.factory as factory  # noqa: E402  (after the skip: needs dexmachina)
from dexmachina.envs.spec import EnvSpec  # noqa: E402

CPU = "cpu"
GOLDEN_CLIP = "box-30-230-s01-01"
GOLDEN_REWARD_OVERRIDES = {"force_penalty": 0.1, "task_rew_weight": 1.0, "bc_wrist_weight": 0.25}
ZERO_EPOCH = 4000  # default max_epochs 5000 - default num_zero_epoch 1000
REWARD_CFG = {"contact_rew_weight": 2.0, "imi_rew_weight": 0.3, "bc_rew_weight": 0.0, "bc_beta": 500.0}
ACTUATED_SPEC = replace(EnvSpec(), hand="allegro_hand", clip=GOLDEN_CLIP, retarget_name="para", actuate_object=True)


def _applied(**changes: Any) -> factory.AppliedEnvConfig:
    values: dict[str, Any] = dict(
        use_contact_reward=True, object_actuated=True, use_curriculum=True, zero_epoch=100,
        reward_weights={"bc_beta": 1.0, "contact_rew_weight": 2.0}, use_imi_rew=True, use_bc_rew=True,
        hand="allegro_left",
    )  # fmt: skip
    return factory.AppliedEnvConfig(**{**values, **changes})


def test_mismatches_names_each_differing_value() -> None:
    configured = _applied()
    applied = _applied(zero_epoch=50, reward_weights={"bc_beta": 1.0, "contact_rew_weight": 3.0})
    assert configured.mismatches(applied) == [
        "zero_epoch: configured 100, applied 50",
        "reward_weights.contact_rew_weight: configured 2.0, applied 3.0",
    ]


def test_no_differences_gives_no_mismatches() -> None:
    assert _applied().mismatches(_applied()) == []


def test_expected_values_follow_the_spec_and_the_reward_config() -> None:
    expected = factory.AppliedEnvConfig.expected(ACTUATED_SPEC, REWARD_CFG, ["bc_beta"])
    assert expected == factory.AppliedEnvConfig(
        use_contact_reward=True, object_actuated=True, use_curriculum=True, zero_epoch=ZERO_EPOCH,
        reward_weights={"bc_beta": 500.0}, use_imi_rew=True, use_bc_rew=False, hand="allegro_hand_left",
    )  # fmt: skip


def test_expected_without_an_actuated_object_has_no_curriculum() -> None:
    expected = factory.AppliedEnvConfig.expected(replace(ACTUATED_SPEC, actuate_object=False), REWARD_CFG, [])
    assert not expected.object_actuated
    assert not expected.use_curriculum
    assert expected.zero_epoch is None


def test_expected_without_an_object_has_no_curriculum() -> None:
    expected = factory.AppliedEnvConfig.expected(replace(ACTUATED_SPEC, no_object=True), REWARD_CFG, [])
    assert not expected.object_actuated
    assert not expected.use_curriculum
    assert expected.zero_epoch is None


def test_expected_maniptrans_curriculum_runs_on_a_fixed_object() -> None:
    spec = replace(ACTUATED_SPEC, actuate_object=False, maniptrans=True)
    expected = factory.AppliedEnvConfig.expected(spec, REWARD_CFG, [])
    assert not expected.object_actuated
    assert expected.use_curriculum
    assert expected.zero_epoch == ZERO_EPOCH


class _BuiltEnv:
    """Stands in for BaseEnv: reports the values the spec asks for, apart from the given changes."""

    def __init__(self, **changes: Any) -> None:
        values: dict[str, Any] = dict(hand="allegro_hand_left", actuated=True, contact=True, zero_epoch=ZERO_EPOCH)
        values.update(changes)
        self.object = SimpleNamespace(actuated=values["actuated"])
        self.use_contact_reward = values["contact"]
        self.use_curriculum = True
        self.curriculum = SimpleNamespace(zero_epoch=values["zero_epoch"])
        self.robots = {"left": SimpleNamespace(name=values["hand"])}
        self.reward_module = SimpleNamespace(use_imi_rew=True, use_bc_rew=True, bc_wrist_weight=0.25)


def _make_env_with_built(built: _BuiltEnv, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(factory, "BaseEnv", lambda **env_kwargs: built)
    spec = replace(ACTUATED_SPEC, imi_rew_weight=0.3, contact_rew_weight=3.0, bc_rew_weight=0.3)
    return factory.make_env(spec, CPU, {"bc_wrist_weight": 0.25})


@pytest.mark.sim
def test_make_env_returns_the_built_env_when_it_matches_the_spec(genesis_cpu, isolated_hand_cfgs, monkeypatch) -> None:
    built = _BuiltEnv()
    assert _make_env_with_built(built, monkeypatch) is built


@pytest.mark.sim
def test_make_env_raises_when_the_built_env_differs_from_the_spec(genesis_cpu, isolated_hand_cfgs, monkeypatch) -> None:
    with pytest.raises(ValueError, match="configured allegro_hand_left, applied inspire_hand_left"):
        _make_env_with_built(_BuiltEnv(hand="inspire_hand_left", zero_epoch=0), monkeypatch)


@pytest.mark.sim
def test_make_env_builds_the_golden_flashsac_env(genesis_cpu, isolated_hand_cfgs) -> None:
    spec = replace(
        EnvSpec(), num_envs=2, seed=0, hand="allegro_hand", clip=GOLDEN_CLIP, retarget_name="para",
        actuate_object=True, use_rl_games=True, task_rew_betas=(10.0, 1.0, 5.0), action_penalty=0.01,
        imi_rew_weight=0.3, imi_wrist_weight=0.5, contact_rew_weight=3.0, contact_beta=10.0, bc_rew_weight=0.3,
        bc_beta=500.0, use_retarget_contact=True,
    )  # fmt: skip
    env = factory.make_env(spec, CPU, GOLDEN_REWARD_OVERRIDES)
    assert env.reward_module.bc_wrist_weight == GOLDEN_REWARD_OVERRIDES["bc_wrist_weight"]
    assert env.num_envs == spec.num_envs

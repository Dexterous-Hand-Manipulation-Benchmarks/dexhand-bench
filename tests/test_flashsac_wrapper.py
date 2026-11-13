"""Unit tests for FlashSAC's DexMachina wrapper against a fake env (no simulator)."""
import dataclasses
import math

import numpy as np
import pytest
import torch

from conftest import flashsac_env_kwargs
from fakes import FakeDexEnv

flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")


def make(**kwargs):
    raw = FakeDexEnv(**kwargs)
    return raw, flash_dex.GenesisFlashSACWrapper(raw, rl_device="cpu")


def test_final_obs_is_the_pre_reset_observation():
    raw, env = make(episode_length=3)
    env.reset()
    for _ in range(2):
        obs, _, term, trunc, infos = env.step(torch.zeros(2, 2))
        assert not (term.any() or trunc.any())
        torch.testing.assert_close(infos["final_obs"], obs)

    obs, _, term, trunc, infos = env.step(torch.zeros(2, 2))
    assert term.all() and trunc.all()
    torch.testing.assert_close(obs[:, 0], torch.tensor([0.0, 100.0]))  # already reset
    torch.testing.assert_close(infos["final_obs"][:, 0], torch.tensor([3.0, 103.0]))  # last obs of the episode


def test_actions_are_clamped_and_broadcast():
    raw, env = make()
    env.reset()
    env.step(torch.full((1, 2), 5.0))
    torch.testing.assert_close(raw.received_actions, torch.ones(2, 2))


def test_nan_observations_are_zeroed():
    raw, env = make()
    env.reset()
    raw.get_observations = lambda: {"policy": torch.full((2, 3), float("nan"))}
    obs, *_ = env.step(torch.zeros(2, 2))
    assert torch.equal(obs, torch.zeros(2, 3))


def test_success_metrics_are_the_env_success_rates():
    raw, env = make()
    raw.success = {"success_2cm": torch.tensor([True, False]), "success_5cm": torch.tensor([True, True])}
    env.reset()
    *_, infos = env.step(torch.zeros(2, 2))
    assert infos["eval/success_2cm"] == 0.5
    assert infos["eval/success_5cm"] == 1.0
    assert math.isclose(infos["env/pos_dist"], raw.dist["pos_dist"], rel_tol=1e-6)


def test_tracking_errors_report_every_metric_dexmachina_provides():
    distances = ("pos_dist", "rot_dist", "arti_dist", "keypoint_dist")
    rew_dict = {key: torch.tensor([0.0, 1.0]) for key in distances}
    rew_dict.update({key: torch.tensor([True, False]) for key in ("success_2cm", "success_5cm")})
    means = flash_dex.TrackingErrors.from_rew_dict(rew_dict).log_means()
    expected_names = {"eval/success_2cm", "eval/success_5cm"} | {f"env/{key}" for key in distances}
    assert means == {name: 0.5 for name in expected_names}


def test_tracking_errors_leave_out_metrics_dexmachina_does_not_provide():
    means = flash_dex.TrackingErrors.from_rew_dict({"pos_dist": torch.tensor([0.25, 0.75])}).log_means()
    assert means == {"env/pos_dist": 0.5}


def test_curriculum_clock_visits_every_epoch_once_in_order():
    max_epochs = flashsac_env_kwargs({})["max_epochs"]
    num_interaction_steps = flash_dex.REPORTED_INTERACTION_STEPS
    epochs = [
        flash_dex.curriculum_epoch(step, num_interaction_steps, max_epochs)
        for step in range(1, num_interaction_steps + 1)
    ]
    # The reported run size divides evenly, so the last step lands on max_epochs itself.
    assert list(dict.fromkeys(epochs)) == list(range(max_epochs + 1))


def test_env_config_keeps_the_factory_defaults_for_absent_keys():
    config = flash_dex.DexmachinaEnvConfig.from_hydra({"max_epochs": 10, "curriculum": False}, "dexmachina/box")
    assert config.clip == "dexmachina/box"
    assert config.hand is None
    assert config.reward_overrides is None
    assert config.actuate_object is False
    assert config.retarget_name == flash_dex.DEFAULT_RETARGET_NAME


def test_env_config_field_cannot_be_reassigned():
    config = flash_dex.DexmachinaEnvConfig.from_hydra({"max_epochs": 10}, "")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_epochs = 20


def test_env_config_reward_overrides_cannot_be_written():
    config = flash_dex.DexmachinaEnvConfig.from_hydra({"max_epochs": 10, "reward_cfg": {"bc_beta": 1.0}}, "")
    with pytest.raises(TypeError):
        config.reward_overrides["bc_beta"] = 2.0


def test_env_config_rejects_curriculum_without_an_actuated_object():
    with pytest.raises(ValueError) as error:
        flash_dex.DexmachinaEnvConfig.from_hydra(
            {"max_epochs": 10, "curriculum": True, "actuate_object": False}, "dexmachina/box"
        )
    message = str(error.value)
    assert "curriculum" in message
    assert "actuate_object" in message


def _applied_config(**changes):
    values = dict(
        use_contact_reward=True, object_actuated=True, use_curriculum=True, zero_epoch=100,
        reward_weights={"bc_beta": 1.0, "contact_rew_weight": 2.0}, use_imi_rew=True, use_bc_rew=True,
        hand="allegro_left",
    )
    return flash_dex.AppliedEnvConfig(**{**values, **changes})


def test_applied_config_mismatches_names_each_differing_value():
    configured = _applied_config()
    applied = _applied_config(zero_epoch=50, reward_weights={"bc_beta": 1.0, "contact_rew_weight": 3.0})
    assert configured.mismatches(applied) == [
        "zero_epoch: configured 100, applied 50",
        "reward_weights.contact_rew_weight: configured 2.0, applied 3.0",
    ]


def test_applied_config_without_differences_has_no_mismatches():
    assert _applied_config().mismatches(_applied_config()) == []


class FakeCamera:
    def __init__(self, frame: torch.Tensor | None = None, error: Exception | None = None) -> None:
        self.frame = frame
        self.error = error

    def render(self) -> tuple[torch.Tensor | None, None, None, None]:
        if self.error is not None:
            raise self.error
        return self.frame, None, None, None


def test_render_returns_the_camera_frame_as_a_batch() -> None:
    raw, env = make()
    raw._floating_camera = FakeCamera(frame=torch.full((4, 5, 3), 1.0))
    image = env.render()
    assert image.shape == (1, 4, 5, 3)
    assert image.dtype == np.uint8
    assert (image == 255).all()


def test_render_failure_raises_instead_of_returning_a_black_image() -> None:
    raw, env = make()
    raw._floating_camera = FakeCamera(error=RuntimeError("no display"))
    with pytest.raises(RuntimeError, match="num_record_episodes=0") as excinfo:
        env.render()
    assert "no display" in str(excinfo.value.__cause__)


def test_render_without_camera_raises() -> None:
    _, env = make()
    with pytest.raises(RuntimeError, match="record_video"):
        env.render()

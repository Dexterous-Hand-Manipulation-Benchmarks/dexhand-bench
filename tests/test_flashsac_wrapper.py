"""Unit tests for FlashSAC's DexMachina wrapper against a fake env (no simulator)."""
import math

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


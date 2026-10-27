"""Unit tests for FlashSAC's DexMachina wrapper against a fake env (no simulator)."""
import math

import pytest
import torch

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


@pytest.mark.parametrize(
    "pos, rot, arti, expected_2cm, expected_5cm",
    [
        (0.01, 0.10, 0.05, 1.0, 1.0),
        (0.03, 0.10, 0.05, 0.0, 1.0),
        (0.01, 0.30, 0.05, 0.0, 1.0),
        (0.01, 0.10, 0.12, 0.0, 1.0),
        (0.06, 0.10, 0.05, 0.0, 0.0),
    ],
)
def test_success_metrics(pos, rot, arti, expected_2cm, expected_5cm):
    raw, env = make()
    raw.dist = {"pos_dist": pos, "rot_dist": rot, "arti_dist": arti}
    env.reset()
    *_, infos = env.step(torch.zeros(2, 2))
    assert infos["eval/success_2cm"] == expected_2cm
    assert infos["eval/success_5cm"] == expected_5cm
    assert math.isclose(infos["env/pos_dist"], pos, rel_tol=1e-6)

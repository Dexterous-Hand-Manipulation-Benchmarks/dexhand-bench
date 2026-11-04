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


def test_success_metrics_are_the_env_success_rates():
    raw, env = make()
    raw.success = {"success_2cm": torch.tensor([True, False]), "success_5cm": torch.tensor([True, True])}
    env.reset()
    *_, infos = env.step(torch.zeros(2, 2))
    assert infos["eval/success_2cm"] == 0.5
    assert infos["eval/success_5cm"] == 1.0
    assert math.isclose(infos["env/pos_dist"], raw.dist["pos_dist"], rel_tol=1e-6)

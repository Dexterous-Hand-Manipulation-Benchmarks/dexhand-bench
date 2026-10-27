"""Unit tests for FastTD3's DexMachina wrapper against a fake env (no simulator)."""
import pytest
import torch

from fakes import FakeDexEnv

td3_dex = pytest.importorskip("fast_td3.environments.dexmachina_env", reason="needs the fasttd3 extra")


def make(**kwargs):
    raw = FakeDexEnv(**kwargs)
    return raw, td3_dex.GenesisFastTD3Wrapper(raw, rl_device="cpu")


def test_dones_and_time_outs():
    raw, env = make(episode_length=2)
    env.reset()
    _, _, dones, infos = env.step(torch.zeros(2, 2))
    assert not dones.any() and not infos["time_outs"].any()
    _, _, dones, infos = env.step(torch.zeros(2, 2))
    assert dones.all() and infos["time_outs"].all()


def test_success_metrics_are_per_env_tensors():
    raw, env = make()
    raw.dist = {"pos_dist": 0.03, "rot_dist": 0.1, "arti_dist": 0.05}
    env.reset()
    *_, infos = env.step(torch.zeros(2, 2))
    assert infos["success_2cm"].tolist() == [False, False]
    assert infos["success_5cm"].tolist() == [True, True]


@pytest.mark.xfail(
    strict=True,
    reason="Known bug: BaseEnv never sets extras['final_obs'], so on time-outs FastTD3 stores and "
    "bootstraps from the first observation of the next episode. FlashSAC's wrapper captures the "
    "terminal observation in reset_idx; FastTD3's should do the same.",
)
def test_raw_obs_is_the_pre_reset_observation():
    raw, env = make(episode_length=2)
    env.reset()
    env.step(torch.zeros(2, 2))
    _, _, dones, infos = env.step(torch.zeros(2, 2))
    assert dones.all()
    torch.testing.assert_close(infos["observations"]["raw"]["obs"][:, 0], torch.tensor([2.0, 102.0]))

"""Unit tests for FastTD3's DexMachina wrapper against a fake env (no simulator)."""
from dataclasses import replace

import pytest
import torch

from fakes import FakeDexEnv, FakeDexVecEnv

td3_dex = pytest.importorskip("fast_td3.environments.dexmachina_env", reason="needs the fasttd3 extra")


def make(**kwargs):
    raw = FakeDexEnv(**kwargs)
    return raw, td3_dex.GenesisFastTD3Wrapper(FakeDexVecEnv(raw), rl_device="cpu")


def test_every_episode_end_is_done_and_never_a_time_out():
    raw, env = make(episode_length=2)
    env.reset()
    _, _, dones, infos = env.step(torch.zeros(2, 2))
    assert not dones.any() and not infos["time_outs"].any()
    _, _, dones, infos = env.step(torch.zeros(2, 2))
    assert dones.all() and not infos["time_outs"].any()


def test_success_metrics_are_per_env_tensors():
    raw, env = make()
    raw.dist = {"pos_dist": 0.03, "rot_dist": 0.1, "arti_dist": 0.05}
    env.reset()
    *_, infos = env.step(torch.zeros(2, 2))
    assert infos["success_2cm"].tolist() == [False, False]
    assert infos["success_5cm"].tolist() == [True, True]


def test_raw_obs_is_the_pre_reset_observation():
    raw, env = make(episode_length=2)
    env.reset()
    env.step(torch.zeros(2, 2))
    obs, _, dones, infos = env.step(torch.zeros(2, 2))
    assert dones.all()
    torch.testing.assert_close(infos["observations"]["raw"]["obs"][:, 0], torch.tensor([2.0, 102.0]))
    torch.testing.assert_close(obs[:, 0], torch.tensor([0.0, 100.0]))


class CurriculumRecordingEnv(FakeDexEnv):
    def __init__(self) -> None:
        super().__init__()
        self.curriculum_epochs: list[int] = []

    def set_curriculum(self, epoch_num: int) -> None:
        self.curriculum_epochs.append(epoch_num)


def curriculum_env(max_epochs: int):
    from dexmachina.envs.benchmark_task import BENCHMARK_TASK

    raw = CurriculumRecordingEnv()
    vec = FakeDexVecEnv(raw)
    vec.spec = replace(BENCHMARK_TASK, max_epochs=max_epochs)
    return raw, td3_dex.GenesisFastTD3Wrapper(vec, rl_device="cpu")


def run_curriculum(env, total_timesteps: int) -> int:
    epoch = -1
    for global_step in range(total_timesteps):
        epoch = td3_dex.advance_curriculum(env, global_step, total_timesteps, epoch)
    return epoch


def test_curriculum_spreads_the_spec_max_epochs_over_the_run_on_the_inner_env():
    raw, env = curriculum_env(max_epochs=4)
    final_epoch = run_curriculum(env, total_timesteps=8)
    assert raw.curriculum_epochs == [0, 1, 2, 3]
    assert final_epoch == 3


def test_env_name_without_hand_and_clip_is_refused():
    with pytest.raises(ValueError, match="dexmachina\\|<hand>\\|<clip>"):
        td3_dex.parse_env_name("dexmachina|allegro_hand")


def test_benchmark_env_spec_takes_hand_clip_seed_and_num_envs_from_the_run_and_the_rest_from_the_task():
    from dexmachina.envs.benchmark_task import BENCHMARK_TASK, RUN_FIELDS

    spec = td3_dex.benchmark_env_spec("dexmachina|allegro_hand|box-30-230", seed=7, num_envs=16)
    assert (spec.hand, spec.clip, spec.seed, spec.num_envs) == ("allegro_hand", "box-30-230-s01-01", 7, 16)
    task_fields = [name for name in BENCHMARK_TASK.__dataclass_fields__ if name not in RUN_FIELDS]
    assert all(getattr(spec, name) == getattr(BENCHMARK_TASK, name) for name in task_fields)

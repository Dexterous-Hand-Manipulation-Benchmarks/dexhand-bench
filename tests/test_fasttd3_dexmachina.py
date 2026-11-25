"""FastTD3's DexMachina env path must reproduce the recorded rollout.

The fixture was recorded through FastTD3's own copy of the env settings, before FastTD3 built its env from the shared
benchmark task. Its task values equal the shared task's, so the observations, rewards and dones still match. Two
fields changed on purpose: `raw_obs` is now the observation the ended episode had (the fixture holds the post-reset
one on done rows), and `time_outs` is now always False (the fixture equals `dones`). The fixture used the FastTD3
default max_epochs (5000) and the task uses 10000; that only matters once `set_curriculum` is called, which this
replay never does.
"""
import pytest
import torch

from conftest import assert_close, compared_steps, load_golden

pytestmark = pytest.mark.sim
td3_dex = pytest.importorskip("fast_td3.environments.dexmachina_env", reason="needs the fasttd3 extra")


def episode_length_column(base_env) -> int:
    """Index of the normalised episode length in the policy observation (BaseEnv.get_observations order)."""
    robots, objects = base_env.robots.values(), base_env.objects.values()
    robot_dims = sum(value.shape[-1] for robot in robots for value in robot.get_observations().values())
    object_dims = sum(value.shape[-1] for obj in objects for value in obj.get_observations().values())
    return robot_dims + object_dims


def test_rollout_through_the_shared_task_matches_golden(reseed, isolated_hand_cfgs):
    from dexmachina.envs.vec_env import DexMachinaVecEnv

    data, meta = load_golden("fasttd3_allegro_box")
    spec = td3_dex.benchmark_env_spec(meta["env_name"], meta["seed"], meta["num_envs"])
    reseed(meta["seed"])
    env = td3_dex.GenesisFastTD3Wrapper(DexMachinaVecEnv.from_spec(spec, "cpu", reward_overrides={}), rl_device="cpu")

    prev_obs = env.reset()
    assert_close(prev_obs, data["obs0"], "reset obs")
    length_col = episode_length_column(env.base_env)
    length_step = 2.0 / env.base_env.max_episode_length
    compared = compared_steps(meta)
    ended_rows = 0
    for t in range(meta["steps"]):
        obs, rew, dones, infos = env.step(torch.from_numpy(data["actions"][t]))
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all(), f"non-finite obs or reward at step {t}"
        assert not infos["time_outs"].any(), f"time_outs at step {t}: the shared task truncates only chunked episodes"
        raw_obs = infos["observations"]["raw"]["obs"]
        ended = dones.bool()
        torch.testing.assert_close(raw_obs[~ended], obs[~ended], msg=f"raw obs of running envs at step {t}")
        if ended.any():
            ended_rows += int(ended.sum())
            # Checked against this rollout's own previous observation, not the vec env's capture of it.
            torch.testing.assert_close(
                raw_obs[ended, length_col], prev_obs[ended, length_col] + length_step,
                msg=f"raw obs of ended envs at step {t} is not one step after their previous observation",
            )
            torch.testing.assert_close(
                obs[ended, length_col], torch.full_like(obs[ended, length_col], -1.0),
                msg=f"obs of ended envs at step {t} is not the first observation of a new episode",
            )
        prev_obs = obs
        if t >= compared:
            continue
        assert_close(obs, data["obs"][t], f"obs at step {t}")
        assert_close(rew, data["rew"][t], f"reward at step {t}")
        assert_close(dones, data["dones"][t], f"dones at step {t}")
        running = ~ended.numpy()
        assert_close(raw_obs[~ended], data["raw_obs"][t][running], f"raw obs of running envs at step {t}")
        for key in ("success_2cm", "success_5cm", "pos_dist", "rot_dist", "arti_dist"):
            assert_close(infos[key], data[key][t], f"{key} at step {t}")
    assert ended_rows > 0, "no episode ended in the replay, so the terminal observation was never checked"


def test_train_and_eval_envs_build_and_an_eval_episode_runs_to_its_end(reseed, isolated_hand_cfgs):
    meta = load_golden("fasttd3_allegro_box")[1]
    spec = td3_dex.benchmark_env_spec(meta["env_name"], meta["seed"], meta["num_envs"])
    reseed(meta["seed"])
    envs = td3_dex.build_dexmachina_envs(spec, "cpu", num_eval_envs=2)

    assert envs.train.spec == spec
    assert envs.evaluation.num_envs == 2
    assert envs.evaluation.base_env.is_eval and envs.evaluation.base_env.curriculum is None

    evaluation = envs.evaluation
    evaluation.reset()
    zero_action = torch.zeros(evaluation.num_envs, evaluation.num_actions)
    ended = torch.zeros(evaluation.num_envs, dtype=torch.bool)
    for _ in range(evaluation.max_episode_steps):
        _, _, dones, infos = evaluation.step(zero_action)
        assert not infos["time_outs"].any()
        ended |= dones.bool()
    assert ended.all(), "an eval env did not end its episode within max_episode_steps"

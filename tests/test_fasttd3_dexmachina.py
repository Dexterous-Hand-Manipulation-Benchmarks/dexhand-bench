"""FastTD3's DexMachina env path must reproduce the recorded rollout.

FastTD3 builds the env from its own DexMachinaArgs defaults, which differ from
the FlashSAC path (action mode, curriculum, observations, ...), so it has its
own golden rollout.
"""
import pytest
import torch

from conftest import assert_close, load_golden

pytestmark = pytest.mark.sim
td3_dex = pytest.importorskip("fast_td3.environments.dexmachina_env", reason="needs the fasttd3 extra")


def test_rollout_matches_golden(reseed):
    from dexmachina.envs import BaseEnv
    from dexmachina.envs.constructors import get_all_env_cfg
    from fast_td3.hyperparams import DexMachinaArgs

    data, meta = load_golden("fasttd3_allegro_box")
    # Rendering off (no viewer / offscreen GL context); it has no effect on the simulation.
    args = DexMachinaArgs(env_name=meta["env_name"], num_envs=meta["num_envs"], seed=meta["seed"],
                          vis=False, record_video=False, record_interval=0)
    td3_dex.apply_dexmachina_env_name(args)
    reseed(meta["seed"])
    env = td3_dex.GenesisFastTD3Wrapper(BaseEnv(**get_all_env_cfg(args, device="cpu")), rl_device="cpu")

    assert_close(env.reset(), data["obs0"], "reset obs")
    for t in range(meta["steps"]):
        obs, rew, dones, infos = env.step(torch.from_numpy(data["actions"][t]))
        assert_close(obs, data["obs"][t], f"obs at step {t}")
        assert_close(rew, data["rew"][t], f"reward at step {t}")
        assert_close(dones, data["dones"][t], f"dones at step {t}")
        assert_close(infos["time_outs"], data["time_outs"][t], f"time_outs at step {t}")
        assert_close(infos["observations"]["raw"]["obs"], data["raw_obs"][t], f"raw obs at step {t}")
        for key in ("success_2cm", "success_5cm", "pos_dist", "rot_dist", "arti_dist"):
            assert_close(infos[key], data[key][t], f"{key} at step {t}")

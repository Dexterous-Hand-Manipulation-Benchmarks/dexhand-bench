"""FlashSAC's DexMachina env path must reproduce the recorded rollout.

Covers dexmachina's dynamics and rewards, FlashSAC's env factory (clip
normalisation, reward overrides, curriculum flags) and the wrapper (terminal
observation capture across the step-199 time-out, online metrics).
"""
import pytest
import torch

from conftest import assert_close, load_golden

pytestmark = pytest.mark.sim
flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")


def test_rollout_matches_golden(reseed):
    data, meta = load_golden("flashsac_allegro_box")
    kwargs = meta["env_kwargs"]
    reseed(meta["seed"])
    raw_env = flash_dex.make_dexmachina_env(meta["num_envs"], "cpu", kwargs, meta["seed"], kwargs["env_name"])
    env = flash_dex.GenesisFlashSACWrapper(raw_env, rl_device="cpu")

    assert env.num_obs == meta["obs_dim"]
    assert env.num_actions == meta["action_dim"]

    obs0, _ = env.reset()
    assert_close(obs0, data["obs0"], "reset obs")

    for t in range(meta["steps"]):
        obs, rew, term, trunc, infos = env.step(torch.from_numpy(data["actions"][t]))
        assert_close(obs, data["obs"][t], f"obs at step {t}")
        assert_close(rew, data["rew"][t], f"reward at step {t}")
        assert_close(term, data["terminated"][t], f"terminated at step {t}")
        assert_close(trunc, data["truncated"][t], f"truncated at step {t}")
        assert_close(infos["final_obs"], data["final_obs"][t], f"final_obs at step {t}")
        for key in ("pos_dist", "rot_dist", "arti_dist"):
            assert_close(raw_env.rew_dict[key], data[key][t], f"{key} at step {t}")
        scalars = sorted(k for k, v in infos.items() if isinstance(v, (int, float)))
        assert scalars == meta["info_keys"]
        assert_close([infos[k] for k in scalars], data["info_scalars"][t], f"info scalars at step {t}")

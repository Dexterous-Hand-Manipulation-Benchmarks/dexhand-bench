"""FlashSAC's DexMachina env path must reproduce the recorded rollout.

Covers dexmachina's dynamics and rewards, FlashSAC's env factory (clip
normalisation, reward overrides, curriculum flags) and the wrapper (terminal
observation capture across the step-199 time-out, online metrics).
"""
import importlib.util
from pathlib import Path

import pytest
import torch

from conftest import assert_close, compared_steps, flashsac_env_kwargs, load_golden

pytestmark = pytest.mark.sim
flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")
# Only the flashsac extra installs Hydra; the other extras collect this file too.
GlobalHydra = pytest.importorskip("hydra.core.global_hydra").GlobalHydra
OmegaConf = pytest.importorskip("omegaconf").OmegaConf

FLASHSAC_DIR = Path(__file__).parents[1] / "algorithms/FlashSAC"


def test_rollout_matches_golden(reseed, isolated_hand_cfgs):
    data, meta = load_golden("flashsac_allegro_box")
    kwargs = flashsac_env_kwargs(meta["env_kwargs"])
    reseed(meta["seed"])
    raw_env = flash_dex.make_dexmachina_env(meta["num_envs"], "cpu", kwargs, meta["seed"], kwargs["env_name"])
    env = flash_dex.GenesisFlashSACWrapper(raw_env, rl_device="cpu")

    assert env.num_obs == meta["obs_dim"]
    assert env.num_actions == meta["action_dim"]

    obs0, _ = env.reset()
    assert_close(obs0, data["obs0"], "reset obs")

    compared = compared_steps(meta)
    for t in range(meta["steps"]):
        obs, rew, term, trunc, infos = env.step(torch.from_numpy(data["actions"][t]))
        # These hold whatever the dynamics do, so they are checked on every platform.
        replaced = (infos["final_obs"] != obs).any(dim=-1)
        assert torch.equal(replaced, term | trunc), f"final_obs replaced outside the done envs at step {t}"
        scalars = sorted(k for k, v in infos.items() if isinstance(v, (int, float)))
        assert scalars == meta["info_keys"]
        if t >= compared:
            continue
        assert_close(term, data["terminated"][t], f"terminated at step {t}")
        assert_close(trunc, data["truncated"][t], f"truncated at step {t}")
        assert_close(obs, data["obs"][t], f"obs at step {t}")
        assert_close(rew, data["rew"][t], f"reward at step {t}")
        assert_close(infos["final_obs"], data["final_obs"][t], f"final_obs at step {t}")
        for key in ("pos_dist", "rot_dist", "arti_dist"):
            assert_close(raw_env.rew_dict[key], data[key][t], f"{key} at step {t}")
        assert_close([infos[k] for k in scalars], data["info_scalars"][t], f"info scalars at step {t}")


def test_training_config_stays_read_only_through_env_construction(
    genesis_cpu, reseed, isolated_hand_cfgs, monkeypatch, request
):
    spec = importlib.util.spec_from_file_location("flashsac_train", FLASHSAC_DIR / "train.py")
    assert spec is not None and spec.loader is not None
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    # train.py leaves Hydra initialised; later composing tests need it cleared.
    request.addfinalizer(GlobalHydra.instance().clear)
    cfg = train.compose_config("./configs", "flashSAC_base", ["env=dexmachina", "env.rl_device=cpu"])
    assert OmegaConf.is_readonly(cfg)
    reseed(cfg.seed)
    # create_envs initialises Genesis itself, which the session fixture already did.
    monkeypatch.setattr(genesis_cpu, "init", lambda **_: None)

    train_env, _, _ = train.create_envs(**cfg.env)
    _, _, _, _, infos = train_env.step(torch.zeros(train_env.num_envs, train_env.num_actions))
    assert infos["final_obs"].shape[0] == cfg.env.num_train_envs

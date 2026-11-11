"""The evaluation's read-only config: one env per kind, and the env count of the run it evaluates."""
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ReadonlyConfigError

flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")

CONFIG_DIR = str(Path(__file__).parents[1] / "algorithms/FlashSAC/configs")
CONFIG_NAME = "flashSAC_base"
# Overrides scripts/flashsac/train.sh passes to the evaluation; env.num_record_envs is set twice there.
TRAIN_SH_OVERRIDES = [
    "env=dexmachina",
    "env.hand=allegro_hand",
    "env.clip=box-30-230",
    "env.retarget_name=para",
    "agent=flashSAC",
    "env.num_record_envs=1",
    "env.num_record_envs=1",
]


def _compose(overrides: list[str]) -> tuple[DictConfig, int]:
    return flash_dex.compose_eval_config(CONFIG_DIR, CONFIG_NAME, overrides)


def test_train_script_overrides_compose_with_the_reported_env_count():
    cfg, train_env_scale = _compose(TRAIN_SH_OVERRIDES)
    assert train_env_scale == flash_dex.REPORTED_NUM_TRAIN_ENVS
    assert (cfg.env.num_train_envs, cfg.env.num_eval_envs, cfg.env.num_record_envs) == (1, 1, 1)


def test_train_env_scale_follows_num_train_envs():
    num_train_envs = 512
    cfg, train_env_scale = _compose([*TRAIN_SH_OVERRIDES, f"num_train_envs={num_train_envs}"])
    assert train_env_scale == num_train_envs
    assert cfg.env.num_train_envs == 1


def test_composed_config_is_read_only():
    cfg, _ = _compose(TRAIN_SH_OVERRIDES)
    with pytest.raises(ReadonlyConfigError):
        cfg.env.num_train_envs = 2


def test_env_config_reads_a_read_only_config():
    cfg, _ = _compose(TRAIN_SH_OVERRIDES)
    assert OmegaConf.is_readonly(cfg.env)
    config = flash_dex.DexmachinaEnvConfig.from_hydra(cfg.env, cfg.env.env_name)
    assert config.clip == "box-30-230"

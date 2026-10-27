"""Characterisation tests for the unified W&B logger (no network, no simulator)."""
import math

import pytest
import torch
import wandb

from dexhand_bench import wandb_logger
from dexhand_bench.wandb_logger import UniversalLogger


@pytest.fixture
def logger():
    """Install the logger and undo its global wandb patches afterwards."""
    orig_init, orig_log = wandb.init, wandb.sdk.wandb_run.Run.log
    yield UniversalLogger
    wandb.init, wandb.sdk.wandb_run.Run.log = orig_init, orig_log
    UniversalLogger._installed = False


def test_shim_exports_the_same_logger():
    import wandb_logger as shim

    assert shim.UniversalLogger is UniversalLogger


@pytest.mark.parametrize("name, canonical", [("FastTD3", "fast_td3"), ("flash-sac", "flash_sac"), ("reppo", "reppo")])
def test_install_canonicalises_algo_and_patches_wandb(logger, name, canonical):
    logger.install(algo=name, hand="allegro_hand", clip="box-30-230", seed=1, num_train_envs=8)
    assert logger.algo == canonical
    assert wandb.init is wandb_logger._interceptor_wandb_init
    assert wandb.sdk.wandb_run.Run.log is wandb_logger._interceptor_run_log


def test_payload_maps_cleans_and_filters(logger):
    logger.install(algo="fast_td3", hand="allegro_hand", clip="box-30-230", seed=1, num_train_envs=8)
    payload = logger.build_sanitized_payload(
        {
            "avg_return": torch.tensor(1.5),   # global mapping
            "qf_loss": 0.25,                    # fast_td3-specific mapping
            "eval/custom": 3,                   # allowed namespace passes through
            "not_namespaced": 4.0,              # dropped
            "nan_metric": float("nan"),         # dropped
            "episode_rew": 9.0,                 # always dropped
        },
        explicit_step=1000,
    )
    assert payload["global/env_step"] == 1000
    assert payload["eval/return"] == 1.5
    assert payload["fast_td3/qf_loss"] == 0.25
    assert payload["eval/custom"] == 3
    for dropped in ("not_namespaced", "nan_metric", "episode_rew"):
        assert dropped not in payload


def test_env_step_precedence(logger):
    logger.install(algo="flash_sac", hand="h", clip="c", seed=0, num_train_envs=4, horizon_length=16)
    assert logger._calculate_env_step({"global/env_step": 7, "env_step": 8}, explicit_step=9) == 7
    assert logger._calculate_env_step({"env_step": 8}, explicit_step=9) == 8
    assert logger._calculate_env_step({}, explicit_step=9) == 9
    assert logger._calculate_env_step({"epoch": 2}, explicit_step=None) == 2 * 4 * 16


def test_clean_val():
    assert UniversalLogger._clean_val(torch.tensor([1.0, float("nan")])) == 1.0
    assert UniversalLogger._clean_val(torch.tensor([1.0, 2.0])) is None
    assert UniversalLogger._clean_val(float("inf")) is None
    assert math.isclose(UniversalLogger._clean_val(torch.tensor(0.5)), 0.5)


def test_standardize_config_adds_benchmark_keys(logger):
    logger.install(algo="flash_sac", hand="xhand", clip="mixer-0-150", seed=3, num_train_envs=1024)
    cfg = logger.standardize_config({"agent": {"gamma": 0.97, "sample_batch_size": 4096}})
    assert cfg["algo"] == "FlashSAC" and cfg["benchmark/algo"] == "FlashSAC"
    assert (cfg["hand"], cfg["clip"], cfg["seed"]) == ("xhand", "mixer-0-150", 3)
    assert cfg["gamma"] == 0.97 and cfg["batch_size"] == 4096

"""REPPO's DexMachina wrapper: episode-end flags and seeding, without a simulator or GPU."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

# Skip only where the reppo extra is not installed; any other import error (e.g. a broken jaxlib) must fail loudly.
wrappers = pytest.importorskip(
    "src.env_utils.jax_wrappers", reason="needs the reppo extra", exc_type=ModuleNotFoundError
)


def test_every_episode_end_is_terminal():
    # DexMachina sets terminated == truncated == True on every reset (time-out, drop, early reset, NaN env).
    terminated = torch.tensor([True, False, True])
    truncated = torch.tensor([True, False, True])
    done, truncation = wrappers.dexmachina_episode_end_flags(terminated, truncated)
    assert done.tolist() == [True, False, True]
    assert truncation.dtype == torch.float32
    assert truncation.tolist() == [0.0, 0.0, 0.0]


def _minimal_env_kwargs():
    return {
        "curriculum_cfg": {
            "schedule": "uniform",
            "zero_epoch": 9000,
            "wait_epochs": 100,
            "interval": 40,
            "upper_ratios": {},
            "rew_thresholds": {},
        },
        "object_cfgs": {"box": {"actuated": True, "kp": 80.0}},
        "env_cfg": {"scene_kwargs": {}},
    }


def _build(monkeypatch, seed):
    """Runs make_dexmachina_env with Genesis, dexmachina's config builder and BaseEnv replaced by fakes."""
    calls = {}

    def fake_gs_init(**kwargs):
        calls["gs_init"] = kwargs

    def fake_get_all_env_cfg(args, device):
        calls["args_seed"] = args.seed
        return _minimal_env_kwargs()

    class ReseedingBaseEnv:
        """Like dexmachina's BaseEnv: building it reseeds the global RNGs with 42 (curriculum, ADD vertices)."""

        def __init__(self, **kwargs):
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

    monkeypatch.setattr(wrappers, "_GENESIS_INITIALIZED", False)
    monkeypatch.setattr(wrappers.gs, "init", fake_gs_init)
    monkeypatch.setattr(wrappers, "get_all_env_cfg", fake_get_all_env_cfg)
    monkeypatch.setattr(wrappers, "BaseEnv", ReseedingBaseEnv)
    hyper = SimpleNamespace(total_time_steps=1000, num_steps=10, num_envs=4, lr=3e-4)
    wrappers.make_dexmachina_env(OmegaConf.create({}), hyper, seed=seed)
    calls["draws"] = (torch.rand(3), np.random.rand(3), random.random())
    return calls


def test_run_seed_reaches_genesis_and_dexmachina(monkeypatch):
    calls = _build(monkeypatch, seed=7)
    assert calls["gs_init"]["seed"] == 7
    assert calls["args_seed"] == 7


def test_run_seed_decides_global_draws_after_env_construction(monkeypatch):
    first, second, other = (_build(monkeypatch, s)["draws"] for s in (0, 0, 1))
    assert torch.equal(first[0], second[0]) and np.array_equal(first[1], second[1]) and first[2] == second[2]
    assert not torch.equal(first[0], other[0])
    assert not np.array_equal(first[1], other[1])
    assert first[2] != other[2]

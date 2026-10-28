"""The env configuration each method's DexMachina path builds, compared without building a scene.

The golden rollouts are too short to exercise some constructor parameters (the curriculum schedule
acts only after thousands of epochs), so the full constructor arguments are snapshotted here.
Record a new snapshot deliberately: DHB_RECORD_SNAPSHOT=1 pytest tests/test_env_construction.py
"""
import hashlib
import json
import os
from pathlib import Path, PurePath
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import GOLDEN_DIR

pytestmark = pytest.mark.sim

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_FILE = GOLDEN_DIR / "env_construction.json"
RECORD_ENV_VAR = "DHB_RECORD_SNAPSHOT"
HASH_DECIMALS = 6  # absorbs float noise between platforms, keeps every config value change visible
FLASHSAC_GOLDEN = "flashsac_allegro_box"
FASTTD3_GOLDEN = "fasttd3_allegro_box"
SNAPSHOT_NUM_ENVS = 2
SNAPSHOT_SEED = 0
DEVICE = "cpu"
NEUTRAL_ENV_NAME = "dexmachina/allegro_hand"
DEFAULT_CLIP_SUFFIX = "s01-01"
FLASHSAC_FALLBACK_CLIP = "box-30-230-s01-01"
FLASHSAC_YAML_DEFAULT_CLIP = "s01_ketchup_01"  # `clip` in FlashSAC configs/env/dexmachina.yaml


def _golden_meta(name):
    return json.loads((GOLDEN_DIR / f"{name}.json").read_text())


def _normalise(value):
    """JSON-comparable form: arrays by shape, dtype and rounded-value hash; paths relative to the repo."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        rounded = np.round(value.astype(np.float64), HASH_DECIMALS) if value.dtype.kind in "fc" else value
        digest = hashlib.sha256(np.ascontiguousarray(rounded).tobytes()).hexdigest()
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": digest}
    if isinstance(value, str) and os.path.isabs(value):
        value = PurePath(value)
    if isinstance(value, PurePath):
        if not value.is_relative_to(REPO_ROOT):
            raise ValueError(f"absolute path outside the repo would leak into a public fixture: {value}")
        return value.relative_to(REPO_ROOT).as_posix()
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    raise TypeError(f"no snapshot form for {type(value).__name__}")


def _assert_matches_snapshot(key, snapshot):
    recorded = json.loads(SNAPSHOT_FILE.read_text()) if SNAPSHOT_FILE.exists() else {}
    if os.environ.get(RECORD_ENV_VAR):
        recorded[key] = snapshot
        SNAPSHOT_FILE.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"recorded {key}")
    assert key in recorded, f"no snapshot for {key}; record it with {RECORD_ENV_VAR}=1"
    assert snapshot == recorded[key]


class _CapturedEnv:
    """Stands in for BaseEnv: keeps the constructor arguments and the reward overrides applied afterwards."""

    def __init__(self, **env_kwargs):
        self.env_kwargs = env_kwargs
        self.reward_module = SimpleNamespace(cfg={})


def _flashsac_variants():
    golden = _golden_meta(FLASHSAC_GOLDEN)["env_kwargs"]
    without_rewards = {k: v for k, v in golden.items() if k != "reward_cfg"}
    return {
        "golden": golden,
        "no_reward_overrides": without_rewards,
        "mixer": {**golden, "clip": "mixer-0-150"},
    }


FLASHSAC_VARIANTS = _flashsac_variants()


@pytest.mark.parametrize("variant, hydra_kwargs", FLASHSAC_VARIANTS.items(), ids=FLASHSAC_VARIANTS)
def test_flashsac_env_config(variant, hydra_kwargs, genesis_cpu, monkeypatch):
    flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")
    import dexmachina.envs.base_env as base_env

    monkeypatch.setattr(base_env, "BaseEnv", _CapturedEnv)
    env = flash_dex.make_dexmachina_env(SNAPSHOT_NUM_ENVS, DEVICE, hydra_kwargs, SNAPSHOT_SEED,
                                        hydra_kwargs["env_name"])
    snapshot = {"env_kwargs": _normalise(env.env_kwargs), "reward_module_cfg": _normalise(env.reward_module.cfg)}
    _assert_matches_snapshot(f"flashsac/{variant}", snapshot)


def test_fasttd3_env_config(genesis_cpu):
    td3_dex = pytest.importorskip("fast_td3.environments.dexmachina_env", reason="needs the fasttd3 extra")
    from dexmachina.envs.constructors import get_all_env_cfg
    from fast_td3.hyperparams import DexMachinaArgs

    meta = _golden_meta(FASTTD3_GOLDEN)
    args = DexMachinaArgs(env_name=meta["env_name"], num_envs=SNAPSHOT_NUM_ENVS, seed=SNAPSHOT_SEED,
                          vis=False, record_video=False, record_interval=0)
    td3_dex.apply_dexmachina_env_name(args)
    _assert_matches_snapshot("fasttd3/golden", {"env_kwargs": _normalise(get_all_env_cfg(args, device=DEVICE))})


class _ClipCaptured(Exception):
    def __init__(self, clip):
        self.clip = clip


def _flashsac_clip_for(requested_clip, monkeypatch):
    flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")
    import dexmachina.envs.constructors as constructors

    def capture_clip(args, device):
        raise _ClipCaptured(args.clip)

    monkeypatch.setattr(constructors, "get_all_env_cfg", capture_clip)
    with pytest.raises(_ClipCaptured) as captured:
        flash_dex.make_dexmachina_env(SNAPSHOT_NUM_ENVS, DEVICE, {"clip": requested_clip}, SNAPSHOT_SEED,
                                      NEUTRAL_ENV_NAME)
    return captured.value.clip


@pytest.mark.parametrize("requested, expected", [
    ("box-30-230", f"box-30-230-{DEFAULT_CLIP_SUFFIX}"),
    ("mixer-0-150", f"mixer-0-150-{DEFAULT_CLIP_SUFFIX}"),
    ("ketchup-0-200", f"ketchup-0-200-{DEFAULT_CLIP_SUFFIX}"),
    (f"ketchup-0-200-{DEFAULT_CLIP_SUFFIX}", f"ketchup-0-200-{DEFAULT_CLIP_SUFFIX}"),
])
def test_flashsac_clip_mapping(requested, expected, monkeypatch):
    assert _flashsac_clip_for(requested, monkeypatch) == expected


@pytest.mark.xfail(strict=True, reason="known issue #3: clips are matched by substring or fall back to "
                                       f"{FLASHSAC_FALLBACK_CLIP}")
@pytest.mark.parametrize("requested", ["box-0-200", "mixer-30-230"])
def test_flashsac_keeps_other_clip_ranges(requested, monkeypatch):
    assert _flashsac_clip_for(requested, monkeypatch) == f"{requested}-{DEFAULT_CLIP_SUFFIX}"


@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception,
                   reason=f"known issue #3: unparseable clips silently become {FLASHSAC_FALLBACK_CLIP}")
def test_flashsac_rejects_unparseable_clip(monkeypatch):
    with pytest.raises(ValueError):
        _flashsac_clip_for(FLASHSAC_YAML_DEFAULT_CLIP, monkeypatch)

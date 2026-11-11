import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"
FLASHSAC_ENV_CONFIG = Path(__file__).parents[1] / "algorithms/FlashSAC/configs/env/dexmachina.yaml"

# Golden rollouts are recorded on CPU and are bit-identical between runs on the same machine.
ATOL = 1e-4
RTOL = 1e-4
# On another platform float rounding differs and contact dynamics amplify it past the tolerance, including the
# done flags (early resets, object falling off). The first step over 1e-4 was 19 on Linux x86_64 for the macOS
# arm64 fixtures, so 15 leaves a margin.
CROSS_PLATFORM_STEPS = 15
# Fixture JSONs store `recorded_on` in this format; compared_steps uses it.
CURRENT_PLATFORM = f"{sys.platform}-{platform.machine()}"

# dexmachina finds a hand's config dict by this attribute suffix (get_hand_specific_cfg in envs/robot.py).
HAND_CFG_SUFFIX = "_CFGS"


def load_golden(name):
    data = np.load(GOLDEN_DIR / f"{name}.npz")
    meta = json.loads((GOLDEN_DIR / f"{name}.json").read_text())
    return data, meta


def compared_steps(meta: dict[str, Any]) -> int:
    """Steps whose values are compared with the fixture: all of them on the platform it was recorded on."""
    return int(meta["steps"]) if meta["recorded_on"] == CURRENT_PLATFORM else CROSS_PLATFORM_STEPS


def flashsac_env_kwargs(recorded):
    """Env-factory kwargs as Hydra passes them now: recorded kwargs plus keys added to the env config since."""
    import yaml  # optional dependency (flashsac extra); only the flashsac tests call this

    env_config = yaml.safe_load(FLASHSAC_ENV_CONFIG.read_text())["env"]
    return {**recorded, "max_epochs": env_config["max_epochs"]}


def assert_close(actual, expected, what):
    actual = actual.detach().cpu().numpy() if hasattr(actual, "detach") else np.asarray(actual)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, err_msg=what)


@pytest.fixture(scope="session")
def genesis_cpu():
    """Initialise Genesis once per test session (it cannot be re-initialised)."""
    gs = pytest.importorskip("genesis")
    gs.init(backend=gs.cpu, logging_level="warning", seed=0)
    return gs


@pytest.fixture
def reseed(genesis_cpu):
    """Reset every RNG Genesis seeds, so each env build starts from the same state."""
    from genesis.utils import set_random_seed

    def _reseed(seed):
        set_random_seed(seed)

    return _reseed


@pytest.fixture
def isolated_hand_cfgs(monkeypatch):
    """Let the test build envs against copies of dexmachina's module-level hand configs.

    Every env build extends a hand's ``kpt_link_names`` list in place (dexmachina ``robot.py``,
    ``link_names += ...`` on a shallow copy), so a later test in the same process would see the grown list.
    """
    import copy
    import importlib
    import pkgutil

    hand_cfgs = importlib.import_module("dexmachina.envs.hand_cfgs")
    for module_info in pkgutil.iter_modules(hand_cfgs.__path__):
        module = importlib.import_module(f"{hand_cfgs.__name__}.{module_info.name}")
        for name in dir(module):
            if name.upper().endswith(HAND_CFG_SUFFIX):
                monkeypatch.setattr(module, name, copy.deepcopy(getattr(module, name)))

import json
from pathlib import Path

import numpy as np
import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"

# Golden rollouts are recorded on CPU and are bit-identical between runs on the
# same machine. The tolerance only absorbs float differences between platforms.
ATOL = 1e-4
RTOL = 1e-4


def load_golden(name):
    data = np.load(GOLDEN_DIR / f"{name}.npz")
    meta = json.loads((GOLDEN_DIR / f"{name}.json").read_text())
    return data, meta


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

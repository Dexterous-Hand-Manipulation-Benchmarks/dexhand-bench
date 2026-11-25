"""Loads FastTD3's scripts the way they run: from fast_td3/ with its modules imported flat.

The installed fast_td3 package would shadow the flat fast_td3.py, so the flat modules are evicted before and after.
"""

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

FASTTD3_DIR = Path(__file__).parents[1] / "algorithms/FastTD3/fast_td3"
FLAT_MODULES = ("fast_td3", "fast_td3_simbav2", "fast_td3_utils", "hyperparams", "environments", "agent_builders")


@contextmanager
def flat_imports(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """fast_td3/ first on sys.path and no flat module cached, inside and after the block."""
    modules_before = set(sys.modules)
    monkeypatch.syspath_prepend(str(FASTTD3_DIR))
    for name in FLAT_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        yield
    finally:
        for name in set(sys.modules) - modules_before:
            if name.split(".")[0] in FLAT_MODULES:
                del sys.modules[name]


def load_script(name: str) -> ModuleType:
    """Executes fast_td3/<name>.py as a module; call inside `flat_imports`."""
    spec = importlib.util.spec_from_file_location(f"fasttd3_{name}", FASTTD3_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

"""Compatibility shim: the logger now lives in ``dexhand_bench.wandb_logger``."""
from dexhand_bench.wandb_logger import *  # noqa: F401,F403
from dexhand_bench.wandb_logger import UniversalLogger  # noqa: F401

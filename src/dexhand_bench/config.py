"""Benchmark configuration: configs/default.yaml, overridden by configs/local.yaml.

Set DHB_CONFIG to use a different local override file.
"""
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def load_config() -> DictConfig:
    cfg = OmegaConf.load(CONFIG_DIR / "default.yaml")
    local = Path(os.environ.get("DHB_CONFIG", CONFIG_DIR / "local.yaml"))
    if local.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(local))
    return cfg


def wandb_settings(cfg: DictConfig) -> dict:
    """W&B entity/project; WANDB_ENTITY / WANDB_PROJECT in the environment take precedence."""
    return {
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY") or cfg.wandb.entity or "",
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT") or cfg.wandb.project or "",
    }


def sbatch_args(cfg: DictConfig, job: str) -> list:
    """sbatch flags for `job`, from slurm.* merged with slurm.jobs.<job>.*."""
    slurm = OmegaConf.to_container(cfg.slurm, resolve=True)
    settings = {k: v for k, v in slurm.items() if k != "jobs"}
    settings.update((slurm.get("jobs") or {}).get(job) or {})
    args = []
    for key in ("partition", "account", "time", "gres", "cpus_per_task"):
        if settings.get(key) is not None:
            args.append(f"--{key.replace('_', '-')}={settings[key]}")
    args.extend(settings.get("extra_args") or [])
    return args

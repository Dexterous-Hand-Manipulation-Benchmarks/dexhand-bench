import os
import time
import wandb
import pandas as pd
from typing import Dict, Any, Optional

# --- Universal Metric Mapping Dictionary ---
METRIC_MAPPING: Dict[str, str] = {
    # 3D Mesh Tracking (DexMachina ADD / AUC-ADD) & Success Thresholds
    "auc_add": "eval/auc_add",
    "AUC_ADD": "eval/auc_add",
    "AUC-ADD": "eval/auc_add",
    "add_auc": "eval/auc_add",
    "ADD_AUC": "eval/auc_add",
    "eval/auc_add": "eval/auc_add",
    "add": "eval/add",
    "ADD": "eval/add",
    "add_mean": "eval/add",
    "eval/add": "eval/add",
    "success_2cm": "eval/success_2cm",
    "success_5cm": "eval/success_5cm",
    "eval/success_2cm": "eval/success_2cm",
    "eval/success_5cm": "eval/success_5cm",
    "avg_success_end": "eval/success_5cm",
    "avg_success_once": "eval/success_once",
    "eval/success_rate": "eval/success_5cm",
    "success_rate": "eval/success_5cm",

    # Returns & Episode Lengths
    "avg_return": "eval/return",
    "eval/mean_reward": "eval/return",
    "rewards/iter": "train/return",
    "rewards/test": "eval/return",
    "eval/avg_return": "eval/return",
    "avg_length": "eval/ep_length",
    "eval/mean_length": "eval/ep_length",
    "length/iter": "train/ep_length",
    "length/test": "eval/ep_length",

    # DexMachina Physical Errors & Contact Forces
    "contact_force": "env/contact_force",
    "contact_forces": "env/contact_force",
    "Episode/pos_dist": "env/pos_dist",
    "pos_dist": "env/pos_dist",
    "Episode/rot_dist": "env/rot_dist",
    "rot_dist": "env/rot_dist",
    "Episode/arti_dist": "env/arti_dist",
    "arti_dist": "env/arti_dist",

    # Policy Dynamics & Losses
    "actor/entropy": "train/entropy",
    "entropy": "train/entropy",
    "actor/loss": "train/actor_loss",
    "actor_loss": "train/actor_loss",
    "a_loss": "train/actor_loss",
    "critic/loss": "train/critic_loss",
    "critic_loss": "train/critic_loss",
    "c_loss": "train/critic_loss",

    # System & Throughput
    "fps/total": "perf/fps",
    "fps": "perf/fps"
}

ALLOWED_NAMESPACES = ("global/", "train/", "eval/", "env/", "perf/")

MODEL_SPECIFIC_ALLOWLIST = {
    "flashsac": {
        "critic/max_entropy_bonus": "flash_sac/max_entropy_bonus",
        "actor/mean_action": "flash_sac/mean_action"
    },
    "fasttd3": {
        "q_loss": "fast_td3/q_loss",
        "q1": "fast_td3/q1_val",
        "q2": "fast_td3/q2_val"
    },
    "ppo": {
        "info/kl": "ppo/kl_divergence",
        "info/e_clip": "ppo/clip_fraction",
        "info/last_lr": "ppo/learning_rate"
    },
    "reppo": {
        "measure_burnin": "reppo/measure_burnin"
    }
}


class UniversalLogger:
    _installed: bool = False
    _orig_wandb_log = None

    @classmethod
    def install(
        cls,
        algo: str,
        hand: str,
        clip: str,
        seed: int,
        num_envs: int = 1024,
        horizon_length: int = 16,
        keep_model_specific: bool = True
    ):
        if cls._installed:
            return

        cls.algo = algo.lower()
        cls.hand = hand
        cls.clip = clip
        cls.seed = seed
        cls.num_envs = int(num_envs)
        cls.horizon_length = int(horizon_length)
        cls.keep_model_specific = keep_model_specific
        cls.start_time = time.time()
        cls.last_env_step = 0
        cls.algo_specific_map = MODEL_SPECIFIC_ALLOWLIST.get(cls.algo, {})

        cls._orig_wandb_log = wandb.log
        wandb.log = cls._intercepted_log
        cls._installed = True

        if wandb.run is not None:
            cls._setup_metrics()

    @classmethod
    def _setup_metrics(cls):
        wandb.define_metric("global/env_step")
        wandb.define_metric("global/*", step_metric="global/env_step")
        wandb.define_metric("train/*", step_metric="global/env_step")
        wandb.define_metric("eval/*", step_metric="global/env_step")
        wandb.define_metric("env/*", step_metric="global/env_step")
        wandb.define_metric("perf/*", step_metric="global/env_step")
        if cls.keep_model_specific:
            wandb.define_metric(f"{cls.algo}/*", step_metric="global/env_step")

    @classmethod
    def _calculate_env_step(cls, raw_data: Dict[str, Any], explicit_step: Optional[int]) -> int:
        if "num_env_steps" in raw_data and pd.notna(raw_data["num_env_steps"]):
            return int(raw_data["num_env_steps"])
        if "total_timesteps" in raw_data and pd.notna(raw_data["total_timesteps"]):
            return int(raw_data["total_timesteps"])
        if "frame" in raw_data and pd.notna(raw_data["frame"]):
            return int(raw_data["frame"])
        if "epoch" in raw_data and pd.notna(raw_data["epoch"]):
            return int(raw_data["epoch"]) * (cls.num_envs * cls.horizon_length)

        if explicit_step is not None:
            if cls.algo == "ppo":
                return int(explicit_step) * (cls.num_envs * cls.horizon_length)
            return int(explicit_step) * cls.num_envs

        cls.last_env_step += cls.num_envs
        return cls.last_env_step

    @classmethod
    def _intercepted_log(cls, data: Dict[str, Any], *args, **kwargs):
        if not isinstance(data, dict):
            cls._orig_wandb_log(data, *args, **kwargs)
            return

        step_arg = kwargs.get("step", None)
        env_step = cls._calculate_env_step(data, step_arg)
        cls.last_env_step = max(cls.last_env_step, env_step)

        sanitized_payload = {
            "global/env_step": env_step,
            "global/wall_time_s": time.time() - cls.start_time
        }

        for key, val in data.items():
            if not pd.notna(val):
                continue
            if key in METRIC_MAPPING:
                sanitized_payload[METRIC_MAPPING[key]] = val
            elif key.startswith(ALLOWED_NAMESPACES):
                sanitized_payload[key] = val
            elif cls.keep_model_specific and key in cls.algo_specific_map:
                sanitized_payload[cls.algo_specific_map[key]] = val
            elif isinstance(val, (wandb.Video, wandb.Image)):
                sanitized_payload[key] = val

        if wandb.run is not None and not getattr(cls, "_metrics_defined", False):
            cls._setup_metrics()
            cls._metrics_defined = True

        kwargs.pop("step", None)
        cls._orig_wandb_log(sanitized_payload, *args, **kwargs)

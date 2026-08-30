import os
import time
import wandb
import wandb.sdk.wandb_run
import numpy as np
import torch
from typing import Dict, Any, Optional

METRIC_MAPPING: Dict[str, str] = {
    # 3D Mesh Tracking
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
    "fasttd3": {
        "q_loss": "fast_td3/q_loss",
        "q1": "fast_td3/q1_val",
        "q2": "fast_td3/q2_val",
        "actor_loss": "train/actor_loss",
        "critic_loss": "train/critic_loss",
    },
    "flashsac": {
        "critic/max_entropy_bonus": "flash_sac/max_entropy_bonus",
        "actor/mean_action": "flash_sac/mean_action"
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
    _orig_run_log = None

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

        # 1. Hook module-level wandb.log
        cls._orig_wandb_log = wandb.log
        wandb.log = cls._intercepted_wandb_log

        # 2. Hook class-level Run.log
        cls._orig_run_log = wandb.sdk.wandb_run.Run.log
        wandb.sdk.wandb_run.Run.log = cls._intercepted_run_log

        cls._installed = True

    @classmethod
    def _clean_val(cls, val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            if val.numel() == 0:
                return None
            if val.numel() == 1:
                item = val.item()
                return None if not np.isfinite(item) else item
            val = val.detach().cpu().numpy()
        if isinstance(val, (np.ndarray, list, tuple)):
            arr = np.asarray(val)
            if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
                return None
            finite_mask = np.isfinite(arr)
            if not finite_mask.any():
                return None
            if not finite_mask.all():
                arr = arr[finite_mask]
            if arr.size == 1:
                return float(arr.item())
            return None
        if isinstance(val, (int, float, np.floating, np.integer)):
            return None if not np.isfinite(val) else (float(val) if isinstance(val, (float, np.floating)) else int(val))
        return val

    @classmethod
    def _calculate_env_step(cls, raw_data: Dict[str, Any], explicit_step: Optional[int]) -> int:
        if "global_step" in raw_data and raw_data["global_step"] is not None:
            return int(raw_data["global_step"])
        if "num_env_steps" in raw_data and raw_data["num_env_steps"] is not None:
            return int(raw_data["num_env_steps"])
        if "total_timesteps" in raw_data and raw_data["total_timesteps"] is not None:
            return int(raw_data["total_timesteps"])
        if "epoch" in raw_data and raw_data["epoch"] is not None:
            return int(raw_data["epoch"]) * (cls.num_envs * cls.horizon_length)
        if explicit_step is not None:
            return int(explicit_step)

        cls.last_env_step += cls.num_envs
        return cls.last_env_step

    @classmethod
    def _build_sanitized_payload(cls, data: Dict[str, Any], explicit_step: Optional[int]) -> Dict[str, Any]:
        env_step = cls._calculate_env_step(data, explicit_step)
        cls.last_env_step = max(cls.last_env_step, env_step)

        sanitized_payload = {
            "global/env_step": env_step,
            "global/wall_time_s": time.time() - cls.start_time
        }

        for key, raw_val in data.items():
            if key == "episode_rew":
                continue
            val = cls._clean_val(raw_val)
            if val is None:
                continue
            if key in METRIC_MAPPING:
                sanitized_payload[METRIC_MAPPING[key]] = val
            elif key.startswith(ALLOWED_NAMESPACES):
                sanitized_payload[key] = val
            elif cls.keep_model_specific and key in cls.algo_specific_map:
                sanitized_payload[cls.algo_specific_map[key]] = val
            elif isinstance(raw_val, (wandb.Video, wandb.Image)):
                sanitized_payload[key] = raw_val

        return sanitized_payload

    @classmethod
    def _intercepted_wandb_log(cls, data: Dict[str, Any], *args, **kwargs):
        if not isinstance(data, dict):
            return cls._orig_wandb_log(data, *args, **kwargs)
        payload = cls._build_sanitized_payload(data, kwargs.get("step", None))
        kwargs.pop("step", None)
        return cls._orig_wandb_log(payload, *args, **kwargs)

    @classmethod
    def _intercepted_run_log(cls, run_self, data: Dict[str, Any], *args, **kwargs):
        if not isinstance(data, dict):
            return cls._orig_run_log(run_self, data, *args, **kwargs)
        payload = cls._build_sanitized_payload(data, kwargs.get("step", None))
        kwargs.pop("step", None)
        return cls._orig_run_log(run_self, payload, *args, **kwargs)
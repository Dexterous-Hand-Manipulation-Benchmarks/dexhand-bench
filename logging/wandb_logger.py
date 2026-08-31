import os
import time
import wandb
import wandb.sdk.wandb_run
import numpy as np
import torch
from typing import Dict, Any, Optional

# --- Universal Metric Mapping ---
METRIC_MAPPING: Dict[str, str] = {
    # 3D Mesh Tracking & Evaluation
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

    # Returns & Lengths
    "avg_return": "eval/return",
    "eval/mean_reward": "eval/return",
    "rewards/iter": "train/return",
    "rewards/test": "eval/return",
    "eval/avg_return": "eval/return",
    "avg_length": "eval/ep_length",
    "eval/mean_length": "eval/ep_length",
    "length/iter": "train/ep_length",
    "length/test": "eval/ep_length",

    # Physical Errors & Contact Forces
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
    "actor/mean_action": "flash_sac/mean_action",
    "critic/max_entropy_bonus": "flash_sac/max_entropy_bonus",

    # System
    "fps/total": "perf/fps",
    "fps": "perf/fps"
}

ALLOWED_NAMESPACES = ("global/", "train/", "eval/", "env/", "perf/", "flash_sac/", "fast_td3/", "ppo/", "reppo/")

MODEL_SPECIFIC_ALLOWLIST = {
    "fasttd3": {
        "q_loss": "fast_td3/q_loss",
        "q1": "fast_td3/q1_val",
        "q2": "fast_td3/q2_val",
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

# Preserve raw SDK methods before hooking
_ORIG_WANDB_INIT = wandb.init
_ORIG_RUN_LOG = wandb.sdk.wandb_run.Run.log


def _extract_val(cfg: dict, *keys, default=None):
    for k in keys:
        if isinstance(k, (list, tuple)):
            curr = cfg
            found = True
            for sub_k in k:
                if isinstance(curr, dict) and sub_k in curr:
                    curr = curr[sub_k]
                else:
                    found = False
                    break
            if found and curr is not None:
                return curr
        else:
            if isinstance(cfg, dict) and k in cfg and cfg[k] is not None:
                return cfg[k]
    return default


def _to_clean_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "_content"):  # OmegaConf DictConfig
        try:
            from omegaconf import OmegaConf
            return OmegaConf.to_container(obj, resolve=True)
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return vars(obj)
    try:
        return dict(obj)
    except Exception:
        return {}


class UniversalLogger:
    _installed: bool = False
    algo: str = "flashsac"
    hand: str = "allegro_hand"
    clip: str = "box-30-230"
    seed: int = 0
    num_envs: int = 1024
    horizon_length: int = 16
    keep_model_specific: bool = True
    start_time: float = time.time()
    last_env_step: int = 0
    algo_specific_map: Dict[str, str] = {}

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
        cls.algo = algo.lower()
        cls.hand = str(hand)
        cls.clip = str(clip)
        cls.seed = int(seed)
        cls.num_envs = int(num_envs)
        cls.horizon_length = int(horizon_length)
        cls.keep_model_specific = keep_model_specific
        cls.start_time = time.time()
        cls.last_env_step = 0
        cls.algo_specific_map = MODEL_SPECIFIC_ALLOWLIST.get(cls.algo, {})

        if not cls._installed:
            wandb.init = _interceptor_wandb_init
            wandb.sdk.wandb_run.Run.log = _interceptor_run_log
            cls._installed = True

    @classmethod
    def standardize_config(cls, raw_cfg: Any) -> Dict[str, Any]:
        cfg = _to_clean_dict(raw_cfg)

        # 1. Standard Algorithm Canonical Name
        raw_algo = (
            cls.algo
            or _extract_val(cfg, ("agent", "agent_type"), "agent", "name", ("params", "algo", "name"), "group_name")
            or "unknown_algo"
        )
        algo_str = str(raw_algo).lower()
        if "flash" in algo_str:
            algo_canonical = "FlashSAC"
        elif "fast" in algo_str:
            algo_canonical = "FastTD3"
        elif "reppo" in algo_str:
            algo_canonical = "REPPO"
        elif "ppo" in algo_str:
            algo_canonical = "PPO"
        else:
            algo_canonical = str(raw_algo).upper()

        # 2. Hand & Clip
        hand = cls.hand or _extract_val(cfg, ("env", "hand"), "hand", ("env", "name"), default="unknown_hand")
        clip = cls.clip or _extract_val(cfg, ("env", "clip"), "clip", default="unknown_clip")

        # 3. Seed & Scale
        seed = cls.seed if cls.seed is not None else _extract_val(cfg, "seed", ("agent", "seed"), ("params", "seed"), default=0)
        num_envs = cls.num_envs if cls.num_envs is not None else _extract_val(
            cfg, ("env", "num_train_envs"), "num_train_envs", "num_envs", ("hyperparameters", "num_envs"), ("params", "config", "num_actors"), default=1024
        )

        # 4. Learning Rate & Architecture
        lr = _extract_val(cfg, ("agent", "learning_rate_init"), "actor_learning_rate", ("hyperparameters", "lr"), ("params", "config", "learning_rate"), default=0.0003)
        actor_dim = _extract_val(cfg, ("agent", "actor_hidden_dim"), "actor_hidden_dim", ("hyperparameters", "actor_hidden_dim"), default=512)
        critic_dim = _extract_val(cfg, ("agent", "critic_hidden_dim"), "critic_hidden_dim", ("hyperparameters", "critic_hidden_dim"), default=512)
        actor_blocks = _extract_val(cfg, ("agent", "actor_num_blocks"), "actor_num_blocks", ("hyperparameters", "num_actor_layers"), default=1)
        critic_blocks = _extract_val(cfg, ("agent", "critic_num_blocks"), "critic_num_blocks", ("hyperparameters", "num_critic_pred_layers"), default=1)
        gamma = _extract_val(cfg, ("agent", "gamma"), ("hyperparameters", "gamma"), "gamma", ("params", "config", "gamma"), default=0.99)
        batch_size = _extract_val(cfg, ("agent", "sample_batch_size"), "batch_size", ("hyperparameters", "num_mini_batches"), ("params", "config", "minibatch_size"), default=2048)
        use_amp = _extract_val(cfg, ("agent", "use_amp"), "amp", ("platform", "amp_enabled"), ("params", "config", "mixed_precision"), default=True)

        canonical_dict = {
            "algo": algo_canonical,
            "hand": str(hand),
            "clip": str(clip),
            "seed": int(seed),
            "num_envs": int(num_envs),
            "learning_rate": float(lr),
            "actor_hidden_dim": int(actor_dim),
            "critic_hidden_dim": int(critic_dim),
            "actor_blocks": int(actor_blocks),
            "critic_blocks": int(critic_blocks),
            "gamma": float(gamma),
            "batch_size": int(batch_size),
            "use_amp": bool(use_amp),
        }

        # Inject into both root level and benchmark/ namespace for flexible W&B grouping
        unified_config = {}
        for k, v in canonical_dict.items():
            unified_config[k] = v
            unified_config[f"benchmark/{k}"] = v

        return unified_config

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
        if "global/env_step" in raw_data and raw_data["global/env_step"] is not None:
            return int(raw_data["global/env_step"])
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
    def build_sanitized_payload(cls, data: Dict[str, Any], explicit_step: Optional[int]) -> Dict[str, Any]:
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


def _interceptor_wandb_init(*args, **kwargs):
    """Intercepts wandb.init() to standardize all run configurations."""
    raw_cfg = kwargs.get("config", {})
    std_cfg = UniversalLogger.standardize_config(raw_cfg)
    kwargs["config"] = std_cfg
    run = _ORIG_WANDB_INIT(*args, **kwargs)

    # Define standard x-axis metric for custom charts
    wandb.define_metric("global/env_step")
    wandb.define_metric("global/*", step_metric="global/env_step")
    wandb.define_metric("train/*", step_metric="global/env_step")
    wandb.define_metric("eval/*", step_metric="global/env_step")
    wandb.define_metric("env/*", step_metric="global/env_step")
    wandb.define_metric("perf/*", step_metric="global/env_step")
    if UniversalLogger.keep_model_specific:
        wandb.define_metric(f"{UniversalLogger.algo}/*", step_metric="global/env_step")

    return run


def _interceptor_run_log(self, data: Dict[str, Any], *args, **kwargs):
    """Intercepts Run.log() across all frameworks."""
    if not isinstance(data, dict):
        return _ORIG_RUN_LOG(self, data, *args, **kwargs)
    payload = UniversalLogger.build_sanitized_payload(data, kwargs.get("step", None))
    kwargs.pop("step", None)
    return _ORIG_RUN_LOG(self, payload, *args, **kwargs)
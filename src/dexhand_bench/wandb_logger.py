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
    # ADD-based success (paper definition: mean per-vertex mesh distance below the
    # threshold) is a DIFFERENT metric from success_*, which FastTD3's
    # envs/FastTD3/dexmachina_wrapper.py computes as a conjunction
    # (pos < p) & (rot < r) & (arti < a). Keep them on separate panels so the two
    # are never silently overlaid on one chart.
    "add_success_1cm": "eval/add_success_1cm",
    "add_success_2cm": "eval/add_success_2cm",
    "add_success_5cm": "eval/add_success_5cm",
    "eval/add_success_1cm": "eval/add_success_1cm",
    "eval/add_success_2cm": "eval/add_success_2cm",
    "eval/add_success_5cm": "eval/add_success_5cm",
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
    "eval_avg_return": "eval/return",
    "env_rewards": "train/return",
    "avg_length": "eval/ep_length",
    "eval/mean_length": "eval/ep_length",
    "length/iter": "train/ep_length",
    "length/test": "eval/ep_length",
    "eval_avg_length": "eval/ep_length",

    ## REPPO mappings
    "eval/episode_return": "eval/return",
    "eval/episode_length": "eval/ep_length",
    "train/episode_return": "train/return",
    "train/episode_length": "train/ep_length",

    # Physical Errors, Forces & Gains
    "contact_force": "env/contact_force",
    "contact_forces": "env/contact_force",
    "Episode/pos_dist": "env/pos_dist",
    "pos_dist": "env/pos_dist",
    "Episode/rot_dist": "env/rot_dist",
    "rot_dist": "env/rot_dist",
    "Episode/arti_dist": "env/arti_dist",
    "arti_dist": "env/arti_dist",
    "keypoint_dist": "env/keypoint_dist",
    "wrist_pdist": "env/wrist_pdist",
    "Episode/keypoint_dist": "env/keypoint_dist",
    "Episode/wrist_pdist": "env/wrist_pdist",
    "kp": "env/kp",
    "kv": "env/kv",
    "fr": "env/fr",
    "curriculum_epoch": "env/curriculum_epoch",

    # Policy & Critic Dynamics
    "actor/entropy": "train/entropy",
    "entropy": "train/entropy",
    "actor/loss": "train/actor_loss",
    "actor_loss": "train/actor_loss",
    "a_loss": "train/actor_loss",
    "critic/loss": "train/critic_loss",
    "critic_loss": "train/critic_loss",
    "c_loss": "train/critic_loss",
    "critic_lr": "train/critic_lr",
    "actor_lr": "train/actor_lr",
    "learning_rate": "train/learning_rate",
    "actor/mean_action": "flash_sac/mean_action",
    "critic/max_entropy_bonus": "flash_sac/max_entropy_bonus",

    ## REPPO mappings
    "train/entropy": "train/entropy",
    "train/critic_loss": "train/critic_loss",
    "train/actor_loss": "train/actor_loss",

    # System
    "fps/total": "perf/fps",
    "fps": "perf/fps",
    "speed": "perf/fps",

    # Steps & Media
    "frame": "global/env_step",
    "env_step": "global/env_step",
    "eval_video": "media/eval_video",
    "render_video": "media/train_video",
    "train_video": "media/train_video",
    "video": "media/eval_video",
    "render": "media/train_video"
}

ALLOWED_NAMESPACES = (
    "global/", "train/", "eval/", "env/", "perf/", "media/", "best/", "fast_td3/", "flash_sac/", "ppo/", "reppo/", "curriculum/"
)

MODEL_SPECIFIC_ALLOWLIST = {
    "fast_td3": {
        "q_loss": "fast_td3/q_loss",
        "qf_loss": "fast_td3/qf_loss",
        "q1": "fast_td3/q1_val",
        "q2": "fast_td3/q2_val",
        "qf_max": "fast_td3/qf_max",
        "qf_min": "fast_td3/qf_min",
        "actor_grad_norm": "fast_td3/actor_grad_norm",
        "critic_grad_norm": "fast_td3/critic_grad_norm",
        "buffer_rewards": "fast_td3/buffer_rewards"
    },
    "flash_sac": {
        "critic/max_entropy_bonus": "flash_sac/max_entropy_bonus",
        "actor/mean_action": "flash_sac/mean_action"
    },
    "ppo": {
        "info/kl": "ppo/kl_divergence",
        "info/e_clip": "ppo/clip_fraction",
        "info/last_lr": "ppo/learning_rate"
    },
    "reppo": {
        "train/q": "reppo/q_value",
        "train/temp": "reppo/temperature",
        "train/kl": "reppo/kl_divergence",
        "train/lagrangian": "reppo/lagrangian",
        "train/lagrangian_loss": "reppo/lagrangian_loss",
        "train/entropy_loss": "reppo/entropy_loss",
        "train/abs_batch_action": "reppo/abs_batch_action",
        "train/abs_pred_action": "reppo/abs_pred_action",
        "train/critic_update_loss": "reppo/critic_update_loss",
        "train/aux_loss": "reppo/aux_loss",
        "train/rew_aux_loss": "reppo/rew_aux_loss",
        "train/target_values": "reppo/target_values",
        "train/reward_mean": "reppo/reward_mean",
        "train/value_loss": "reppo/mse_loss",
        "train/critic_total_loss": "reppo/critic_total_loss",
        "train/actor_total_loss": "reppo/actor_total_loss",
    }
}

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
    if hasattr(obj, "_content"):
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
    is_eval: bool = False
    algo: str = "flash_sac"
    hand: str = "allegro_hand"
    clip: str = "box-30-230"
    seed: int = 0
    num_envs: int = 1024
    num_train_envs: int = 1024
    num_eval_envs: Optional[int] = None
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
        num_train_envs: Optional[int] = None,
        num_eval_envs: Optional[int] = None,
        horizon_length: int = 16,
        keep_model_specific: bool = True,
        is_eval: bool = False,
        **kwargs
    ):
        algo_key = algo.lower().replace("-", "_")
        if algo_key in ("fasttd3", "fast_td3"):
            algo_key = "fast_td3"
        elif algo_key in ("flashsac", "flash_sac"):
            algo_key = "flash_sac"
        cls.algo = algo_key
        cls.hand = str(hand)
        cls.clip = str(clip)
        cls.seed = int(seed)
        cls.num_train_envs = int(num_train_envs if num_train_envs is not None else num_envs)
        cls.num_envs = cls.num_train_envs
        cls.num_eval_envs = int(num_eval_envs) if num_eval_envs is not None else None
        cls.horizon_length = int(horizon_length)
        cls.keep_model_specific = keep_model_specific
        cls.is_eval = is_eval
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

        hand = cls.hand or _extract_val(cfg, ("env", "hand"), "hand", ("env", "name"), default="unknown_hand")
        clip = cls.clip or _extract_val(cfg, ("env", "clip"), "clip", default="unknown_clip")

        # 3. Seed & Environment Counts (Explicit Train vs Eval)
        seed = cls.seed if cls.seed is not None else _extract_val(cfg, "seed", ("agent", "seed"), ("params", "seed"), default=0)

        num_train = cls.num_train_envs or _extract_val(
            cfg, ("env", "num_train_envs"), "num_train_envs", "num_envs", ("hyperparameters", "num_envs"), ("params", "config", "num_actors"), default=1024
        )
        num_eval = cls.num_eval_envs or _extract_val(
            cfg, ("env", "num_eval_envs"), "num_eval_envs", "num_eval", default=None
        )

        lr = _extract_val(cfg, ("agent", "learning_rate_init"), "actor_learning_rate", ("hyperparameters", "lr"), ("params", "config", "learning_rate"), default=0.0003)
        actor_dim = _extract_val(cfg, ("agent", "actor_hidden_dim"), "actor_hidden_dim", ("hyperparameters", "actor_hidden_dim"), default=512)
        critic_dim = _extract_val(cfg, ("agent", "critic_hidden_dim"), "critic_hidden_dim", ("hyperparameters", "critic_hidden_dim"), default=512)
        actor_blocks = _extract_val(cfg, ("agent", "actor_num_blocks"), "actor_num_blocks", ("hyperparameters", "num_actor_layers"), default=1)
        critic_blocks = _extract_val(cfg, ("agent", "critic_num_blocks"), "critic_num_blocks", ("hyperparameters", "num_critic_pred_layers"), default=1)
        gamma = _extract_val(cfg, ("agent", "gamma"), ("hyperparameters", "gamma"), "gamma", ("params", "config", "gamma"), default=0.99)
        batch_size = _extract_val(cfg, ("agent", "sample_batch_size"), "batch_size", ("hyperparameters", "num_mini_batches"), ("params", "config", "minibatch_size"), default=2048)
        use_amp = _extract_val(cfg, ("agent", "use_amp"), "amp", ("platform", "amp_enabled"), ("params", "config", "mixed_precision"), default=True)
        normalize_reward = _extract_val(cfg, ("agent", "normalize_reward"), "reward_normalization", default=False)

        canonical_dict = {
            "algo": algo_canonical,
            "hand": str(hand),
            "clip": str(clip),
            "seed": int(seed),
            "num_envs": int(num_train),
            "num_train_envs": int(num_train),
            "learning_rate": float(lr),
            "actor_hidden_dim": int(actor_dim),
            "critic_hidden_dim": int(critic_dim),
            "actor_blocks": int(actor_blocks),
            "critic_blocks": int(critic_blocks),
            "gamma": float(gamma),
            "batch_size": int(batch_size),
            "use_amp": bool(use_amp),
            "normalize_reward": bool(normalize_reward),
        }
        if num_eval is not None:
            canonical_dict["num_eval_envs"] = int(num_eval)

        unified_config = dict(cfg)
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
        if "env_step" in raw_data and raw_data["env_step"] is not None:
            return int(raw_data["env_step"])
        if "frame" in raw_data and raw_data["frame"] is not None:
            return int(raw_data["frame"])
        if explicit_step is not None:
            return int(explicit_step)
        if "global_step" in raw_data and raw_data["global_step"] is not None:
            return int(raw_data["global_step"])
        if "num_env_steps" in raw_data and raw_data["num_env_steps"] is not None:
            return int(raw_data["num_env_steps"])
        if "total_timesteps" in raw_data and raw_data["total_timesteps"] is not None:
            return int(raw_data["total_timesteps"])
        if "epoch" in raw_data and raw_data["epoch"] is not None:
            return int(raw_data["epoch"]) * (cls.num_envs * cls.horizon_length)

        cls.last_env_step += cls.num_envs
        return cls.last_env_step

    @classmethod
    def build_sanitized_payload(cls, data: Dict[str, Any], explicit_step: Optional[int]) -> Dict[str, Any]:
        env_step = cls._calculate_env_step(data, explicit_step)
        cls.last_env_step = max(cls.last_env_step, env_step)

        sanitized_payload = {
            "global/env_step": env_step,
            "global/step": env_step,
            "global/wall_time_s": time.time() - cls.start_time
        }

        for key, raw_val in data.items():
            if key in ("episode_rew", "checkpoint_step"):
                continue
            # Preserve videos, images, and tables
            if isinstance(raw_val, (wandb.Video, wandb.Image, wandb.Table)):
                mapped_key = METRIC_MAPPING.get(key, key if key.startswith(ALLOWED_NAMESPACES) else f"media/{key}")
                sanitized_payload[mapped_key] = raw_val
                continue

            val = cls._clean_val(raw_val)
            if val is None:
                continue
                
            # 1. Global mapping (Highest Priority)
            if key in METRIC_MAPPING:
                sanitized_payload[METRIC_MAPPING[key]] = val
                
            # 2. Algorithm-Specific overrides (Second Priority)
            elif cls.keep_model_specific and key in cls.algo_specific_map:
                sanitized_payload[cls.algo_specific_map[key]] = val

            # 3. Generic wildcard fallback (Lowest Priority)
            elif key.startswith(ALLOWED_NAMESPACES):
                sanitized_payload[key] = val

        return sanitized_payload


def _interceptor_wandb_init(*args, **kwargs):
    """Intercepts wandb.init() to standardize all run configurations."""
    tags = list(kwargs.get("tags", []))
    if "curriculum" not in tags:
        tags.append("curriculum")
    kwargs["tags"] = tags

    # Only standardize/inject config during training, never overwrite during evaluation
    is_resumed = kwargs.get("resume") is not None
    if not UniversalLogger.is_eval and not is_resumed:
        raw_cfg = kwargs.get("config", {})
        std_cfg = UniversalLogger.standardize_config(raw_cfg)
        std_cfg["curriculum"] = True
        std_cfg["benchmark/curriculum"] = True
        kwargs["config"] = std_cfg
    elif UniversalLogger.is_eval:
        # In evaluation mode, never overwrite the training run's config
        kwargs.pop("config", None)
    elif is_resumed and "config" in kwargs:
        # Resuming training with an explicit config update
        raw_cfg = kwargs.get("config", {})
        std_cfg = UniversalLogger.standardize_config(raw_cfg)
        std_cfg["curriculum"] = True
        std_cfg["benchmark/curriculum"] = True
        kwargs["config"] = std_cfg

    run = _ORIG_WANDB_INIT(*args, **kwargs)

    # Define standard x-axis metric for custom charts
    try:
        wandb.define_metric("global/env_step")
        wandb.define_metric("global/step")
        wandb.define_metric("*", step_metric="global/env_step")
        wandb.define_metric("global/*", step_metric="global/env_step")
        wandb.define_metric("train/*", step_metric="global/env_step")
        wandb.define_metric("eval/*", step_metric="global/env_step")
        wandb.define_metric("env/*", step_metric="global/env_step")
        wandb.define_metric("perf/*", step_metric="global/env_step")
        wandb.define_metric("media/*", step_metric="global/env_step")
        wandb.define_metric("fast_td3/*", step_metric="global/env_step")
        wandb.define_metric("flash_sac/*", step_metric="global/env_step")
        wandb.define_metric("ppo/*", step_metric="global/env_step")
        wandb.define_metric("reppo/*", step_metric="global/env_step")
        wandb.define_metric("best/*", step_metric="global/env_step")
    except Exception:
        pass

    return run


def _interceptor_run_log(self, data: Dict[str, Any], *args, **kwargs):
    """Intercepts Run.log() across all frameworks."""
    if not isinstance(data, dict):
        return _ORIG_RUN_LOG(self, data, *args, **kwargs)
    explicit_step = kwargs.get("step", None)
    payload = UniversalLogger.build_sanitized_payload(data, explicit_step)
    env_step = payload.get("global/env_step", explicit_step)

    # In evaluation mode, when resuming an existing run, or when explicit_step was not passed,
    # avoid overriding WandB's internal step counter. Metrics are mapped to global/env_step
    # via wandb.define_metric(step_metric="global/env_step"). Forcing the internal step
    # triggers non-monotonic step errors if checkpoints are evaluated out-of-order or post-training.
    is_resumed = getattr(self, "resumed", False)
    if UniversalLogger.is_eval or is_resumed or explicit_step is None:
        kwargs.pop("step", None)
    else:
        current_wandb_step = getattr(self, "step", 0)
        if env_step is not None and env_step >= current_wandb_step:
            kwargs["step"] = int(env_step)
        else:
            kwargs.pop("step", None)

    return _ORIG_RUN_LOG(self, payload, *args, **kwargs)
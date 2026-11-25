# pyright: strict
"""Writes env_spec.json and config.yaml for the lab FlashSAC checkpoints, which were trained before runs saved them.

The files are rebuilt from the W&B config of each run (the export JSON of `export_wandb_runs.py`), so that
`evaluate_benchmark` can evaluate the checkpoints on the shared protocol. Per run the output directory gets

    <out>/<run>/env_spec.json       the run spec, written by dexmachina's `save_run_spec`; its dexmachina commit is
                                    the one the lab runs used, not today's
    <out>/<run>/config.yaml         the W&B config as the resolved Hydra config the run had
    <out>/<run>/reconstructed.json  marks the files as reconstructed and lists every spec field that no W&B key set
    <out>/<run>/step<N>/            links to the checkpoint files, the layout `evaluate_benchmark` reads

The links are absolute paths into the checkpoint directories of this machine, so the output stays local: never commit
it. The output directory must not exist yet.

The spec is the translation of the Hydra env config that the lab runs used (`lab_spec`: `EnvSpec` defaults plus what
the config sets), not the benchmark task that FlashSAC trains on today. Fields that the W&B config does not set take
the `EnvSpec` defaults.
The lab runs used dexmachina commit `LAB_DEXMACHINA_SHA`; its argparse defaults equal today's `EnvSpec` defaults.
A W&B key that has no mapping stops the run with an error.

    uv run --exact --extra flashsac python scripts/analysis/write_lab_run_files.py \
        --export <wandb_runs.json> --checkpoint-root <dir with one directory per run> --out-dir <dir>

The saved configs keep `use_compile` true (the checkpoint keys carry the compiled module's prefix). On a CPU where
torch's compiler fails, evaluate with `TORCHDYNAMO_DISABLE=1`; do not switch `use_compile` off, or loading fails.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, cast

from dexmachina.envs.constructors import (
    parse_clip_string,  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
)
from dexmachina.envs.rewards import get_reward_cfg  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from dexmachina.envs.run_spec import (  # pyright: ignore[reportMissingTypeStubs]
    DIFF_KEY,
    DIRTY_KEY,
    RUN_SPEC_FILENAME,
    SHA_KEY,
    load_run_spec,
    save_run_spec,
)
from dexmachina.envs.spec import EnvSpec  # pyright: ignore[reportMissingTypeStubs]
from omegaconf import OmegaConf

from flash_rl.envs.dexmachina import (  # pyright: ignore[reportMissingTypeStubs]
    EXTRA_REWARD_KEYS,
    RUN_CONFIG_FILENAME,
    DexmachinaEnvConfig,
    full_clip_id,
)

# The dexmachina code the lab runs used: the commit whose envs/, eval/, rl/ and the other code directories are
# byte-identical (same git tree hashes) to the copy vendored in the lab repository at both commits that W&B records for
# the lab runs (2026-09-12, "new fix"). The vendored code is unchanged from lab commit aa40f41 to 5728dd2; the URDFs
# differ by line endings only. The retarget data behind the assets symlink is outside git and not verified. Its
# get_common_argparser defaults equal today's EnvSpec defaults, field for field.
LAB_DEXMACHINA_SHA = "6e1e5ad1ea059222c2b8dc7317d60fb61c00b015"
# Lab runs had no action_mode key; dexmachina's default then was residual.
LAB_ACTION_MODE = "residual"
# The lab env factory read `hydra_kwargs.get("max_epochs", 10000)` and the runs set none; the lab report gives the same
# curriculum horizon E_max = 10,000. task_hash excludes max_epochs, so the value does not change what the eval scores.
LAB_MAX_EPOCHS = 10_000
LAB_GROUP = "FlashSAC"
LAB_CHECKPOINT_SUBDIR = "evalseed0"
RECONSTRUCTED_FILENAME = "reconstructed.json"
ACTOR_FILENAME = "actor.pt"
# The server working tree of the lab runs is unknown (the tree equality above covers the commit only), so the spec
# file says dirty and evaluation warns; reconstructed.json says why.
DIRTY_UNKNOWN_REASON = "the lab runs' working tree is unknown; only the committed code was compared"
# The lab FlashSAC commits W&B records per run, and what each had in update.py (the NaN check and clamp were added
# between them). These are the commits of the eval runs, not of training.
UPDATE_WITHOUT_NAN_CHECK = "5ea21f6f0465f3a007705d5057d2dcc601039c36"
UPDATE_WITH_NAN_CHECK = "5728dd20c6576a27e28461f4fc01755fed6a5297"
UPDATE_VARIANTS = {
    UPDATE_WITHOUT_NAN_CHECK: "update.py without the NaN check and the extra clamp",
    UPDATE_WITH_NAN_CHECK: "update.py with the NaN check and the extra clamp",
}
VELOCITY_OBSERVATION_DIFFERENCE = (
    "The lab runs trained with velocity observations scaled twice on steps with a reset: dexmachina before commit "
    "90d1151 scaled the state buffers in place in the robot and object get_observations, and the lab wrapper's reset "
    "hook called get_observations again. Today's evaluation feeds singly scaled velocities."
)

# Top-level W&B config keys that are the run's Hydra config; config.yaml keeps them.
HYDRA_KEYS = frozenset(
    {
        "agent", "env", "agent_load_path", "buffer_load_path", "evaluation_per_interaction_step", "exp_name",
        "gamma", "group_name", "logger_type", "logging_per_interaction_step", "metrics_per_interaction_step", "n_step",
        "num_env_steps", "num_eval_envs", "num_eval_episodes", "num_interaction_steps", "num_record_envs",
        "num_record_episodes", "num_train_envs", "num_update_steps", "project_name",
        "recording_per_interaction_step", "save_buffer_per_interaction_step",
        "save_checkpoint_per_interaction_step", "save_path", "seed", "updates_per_interaction_step",
    }
)  # fmt: skip
# The logger's own copies of Hydra values (UniversalLogger); not part of the Hydra config, so not written.
LOGGER_PREFIX = "benchmark/"
LOGGER_ALIAS_KEYS = frozenset(
    {
        "actor_blocks", "actor_hidden_dim", "algo", "batch_size", "clip", "critic_blocks", "critic_hidden_dim",
        "curriculum", "hand", "learning_rate", "normalize_reward", "num_envs", "use_amp",
    }
)  # fmt: skip
# The W&B team name; the saved config gets null in its place (the Hydra default), the files must stay shareable.
PRIVATE_KEY = "entity_name"
# env keys: the ones that set spec fields, and the ones that only describe the lab run's infrastructure.
SPEC_ENV_KEYS = frozenset({"hand", "clip", "retarget_name", "reward_cfg", "curriculum", "seed", "env_name"})
INFRASTRUCTURE_ENV_KEYS = frozenset(
    {"env_type", "num_env_steps", "num_train_envs", "num_eval_envs", "num_record_envs", "rl_device"}
)
# These must be unset: a value would change the env or the action scaling, and nothing here maps it.
UNSET_ENV_KEYS = ("rescale_action", "max_episode_steps")
# Spec fields that the translation always sets, whatever the W&B config holds.
ALWAYS_SET_SPEC_FIELDS = frozenset(
    {"num_envs", "seed", "actuate_object", "max_epochs", "retarget_name", "use_rl_games", "hand", "clip", "action_mode"}
)
CLIP_DERIVED_SPEC_FIELDS = frozenset({"frame_start", "frame_end", "arctic_object"})
# Keys of dexmachina's reward config (envs/rewards.py) that the factory also maps onto dexmachina's args.
OBJ_POS_BETA_KEY = "obj_pos_beta"
OBJ_ROT_BETA_KEY = "obj_rot_beta"
OBJ_ARTI_BETA_KEY = "obj_arti_beta"
BETA_KEYS = (OBJ_POS_BETA_KEY, OBJ_ROT_BETA_KEY, OBJ_ARTI_BETA_KEY)
CHECKPOINT_DIRNAME = "step{step}"


@dataclass(frozen=True)
class LabRun:
    config: Mapping[str, Any]
    flashsac_commit: str


def lab_runs(export: Path) -> dict[str, LabRun]:
    """Every FlashSAC run in the export, by experiment name (the checkpoint directory's name)."""
    runs = cast(list[dict[str, Any]], json.loads(export.read_text()))
    configs: dict[str, LabRun] = {}
    for run in runs:
        if run["group"] != LAB_GROUP:
            continue
        name = str(run["config"]["exp_name"])
        if name in configs:
            raise ValueError(
                f"{export} has two {LAB_GROUP} runs named {name!r}. The checkpoints of {name} cannot be matched to "
                "one of them, and their configs could differ. Remove the run that is not the lab run from a copy of "
                "the export."
            )
        configs[name] = LabRun(run["config"], str(run["metadata"]["git"]["commit"]))
    return configs


def hydra_config(name: str, wandb_config: Mapping[str, Any]) -> dict[str, Any]:
    """The run's Hydra config out of its W&B config; fails on a key that is neither Hydra nor a logger copy."""
    unmapped = sorted(
        key
        for key in wandb_config
        if key not in HYDRA_KEYS
        and key not in LOGGER_ALIAS_KEYS
        and key != PRIVATE_KEY
        and not key.startswith(LOGGER_PREFIX)
    )
    if unmapped:
        raise ValueError(
            f"Run {name}: the W&B config has keys {unmapped} that the reconstruction does not know. They could change "
            "what the run trained on, and nothing here would carry them into env_spec.json or config.yaml. Add each "
            "to HYDRA_KEYS (it is part of the Hydra config) or to LOGGER_ALIAS_KEYS (the logger's copy of a Hydra "
            "value) after checking which it is."
        )
    missing = sorted(HYDRA_KEYS - wandb_config.keys())
    if missing:
        raise ValueError(
            f"Run {name}: the W&B config lacks the Hydra keys {missing}. config.yaml would be incomplete and the "
            "run's settings for them unknown. Check that the export has the run's full config (export it again), or "
            "remove the key from HYDRA_KEYS if the lab runs never had it."
        )
    config = {key: wandb_config[key] for key in HYDRA_KEYS}
    config[PRIVATE_KEY] = None
    return config


def check_logger_copies(name: str, wandb_config: Mapping[str, Any]) -> None:
    env = cast(Mapping[str, Any], wandb_config["env"])
    copies = {
        "hand": env["hand"], "clip": env["clip"], "num_envs": env["num_train_envs"], "curriculum": env["curriculum"],
    }  # fmt: skip
    differing = {key: (wandb_config[key], value) for key, value in copies.items() if wandb_config[key] != value}
    if differing:
        raise ValueError(
            f"Run {name}: the logger's copies disagree with the Hydra env config (copy, env): {differing}. The run's "
            "settings are then not what the W&B config says in one place; find out which one the run used before "
            "reconstructing its files."
        )


def check_env_keys(name: str, env: Mapping[str, Any]) -> None:
    unmapped = sorted(set(env) - SPEC_ENV_KEYS - INFRASTRUCTURE_ENV_KEYS - set(UNSET_ENV_KEYS))
    if unmapped:
        raise ValueError(
            f"Run {name}: the W&B env config has keys {unmapped} that the reconstruction does not map to a spec field. "
            "A key such as action_mode would change the env, so a default must not stand in for it. Map the key to "
            "its EnvSpec field in this script after checking what the lab code did with it."
        )
    set_keys = [key for key in UNSET_ENV_KEYS if env.get(key) is not None]
    if set_keys:
        raise ValueError(
            f"Run {name}: the W&B env config sets {set_keys}, which the reconstruction does not map (the lab runs "
            "left them null). Map them to what they changed in the env, or leave them null."
        )


def spec_fields_set_by(env: Mapping[str, Any]) -> frozenset[str]:
    """The EnvSpec fields that the W&B env config sets; the others take the EnvSpec default."""
    spec_field_names = {field.name for field in fields(EnvSpec)}
    reward_cfg = cast(Mapping[str, Any], env.get("reward_cfg") or {})
    set_by_rewards = {key for key in reward_cfg if key in spec_field_names}
    if any(key in reward_cfg for key in BETA_KEYS):
        set_by_rewards.add("task_rew_betas")
    return ALWAYS_SET_SPEC_FIELDS | frozenset(set_by_rewards)


def lab_spec(
    num_envs: int, hydra_kwargs: Mapping[str, Any], seed: int, env_name: str
) -> tuple[EnvSpec, dict[str, Any]]:
    """The EnvSpec the lab FlashSAC runs built from their Hydra env kwargs, and their reward overrides.

    A frozen copy of the translation the lab runs used, on purpose: it must not follow `dexmachina_spec`, which now
    builds the benchmark task. It has no check that max_epochs exceeds num_zero_epoch: the lab value
    (`LAB_MAX_EPOCHS`) is 10_000, so that check could never fire here.
    """
    config = DexmachinaEnvConfig.from_hydra(hydra_kwargs, env_name)
    spec_values: dict[str, Any] = {
        "num_envs": num_envs,
        "seed": seed,
        "actuate_object": config.actuate_object,
        "max_epochs": config.max_epochs,
        "retarget_name": config.retarget_name,
        "use_rl_games": True,
        "clip": full_clip_id(config.clip),
    }
    if config.hand is not None:
        spec_values["hand"] = config.hand
    rewards = config.reward_overrides
    if rewards is not None:
        known_keys = set(cast(Mapping[str, Any], get_reward_cfg()))
        unknown_keys = sorted(set(rewards) - known_keys - EXTRA_REWARD_KEYS)
        if unknown_keys:
            raise ValueError(f"unknown reward_cfg keys: {unknown_keys}")
        spec_field_names = {f.name for f in fields(EnvSpec)}
        spec_values.update({k: v for k, v in rewards.items() if k in spec_field_names})
        if any(key in rewards for key in BETA_KEYS):
            defaults = EnvSpec().task_rew_betas
            spec_values["task_rew_betas"] = (
                rewards.get(OBJ_POS_BETA_KEY, defaults[0]),
                rewards.get(OBJ_ROT_BETA_KEY, defaults[1]),
                rewards.get(OBJ_ARTI_BETA_KEY, defaults[2]),
            )
    parse_clip_string(spec_values["clip"])
    return replace(EnvSpec(), **spec_values), {} if rewards is None else dict(rewards)


def reconstruct_spec(name: str, wandb_config: Mapping[str, Any]) -> tuple[EnvSpec, dict[str, Any], dict[str, Any]]:
    """The run's EnvSpec, its reward overrides, and the spec fields (with values) that took the EnvSpec default."""
    env = cast(Mapping[str, Any], wandb_config["env"])
    check_logger_copies(name, wandb_config)
    check_env_keys(name, env)
    hydra_kwargs = {**env, "max_epochs": LAB_MAX_EPOCHS}
    spec, reward_overrides = lab_spec(int(env["num_train_envs"]), hydra_kwargs, int(env["seed"]), str(env["env_name"]))
    spec = replace(spec, action_mode=LAB_ACTION_MODE)
    set_fields = spec_fields_set_by(env) | CLIP_DERIVED_SPEC_FIELDS
    values = cast(dict[str, Any], json.loads(spec.to_json()))
    defaulted = {field: value for field, value in sorted(values.items()) if field not in set_fields}
    return spec, reward_overrides, defaulted


def derived_from_clip(spec: EnvSpec) -> dict[str, Any]:
    """What dexmachina sets from the clip at env construction, and what the spec file holds for it instead.

    The spec keeps the EnvSpec defaults for these fields, as every spec FlashSAC saves does, so a reconstructed spec
    and a fresh run of the same task share a task hash.
    """
    obj_name, frame_start, frame_end, *_ = parse_clip_string(spec.clip)
    return {
        "clip": spec.clip,
        "arctic_object": obj_name,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "spec_file_holds": {field: getattr(spec, field) for field in sorted(CLIP_DERIVED_SPEC_FIELDS)},
    }


def checkpoint_step(wandb_config: Mapping[str, Any]) -> int:
    """The last step that the run saved: the saving interval times the number of intervals that fit in the run."""
    interval = int(wandb_config["save_checkpoint_per_interaction_step"])
    return int(wandb_config["num_interaction_steps"]) // interval * interval


def write_spec_file(run_dir: Path, spec: EnvSpec, reward_overrides: Mapping[str, Any]) -> None:
    """env_spec.json through `save_run_spec`, with the lab's dexmachina commit instead of the checkout's."""
    path = save_run_spec(run_dir, spec, reward_overrides)
    content = cast(dict[str, Any], json.loads(path.read_text()))
    content[SHA_KEY] = LAB_DEXMACHINA_SHA
    content[DIRTY_KEY] = True
    content[DIFF_KEY] = None
    path.write_text(json.dumps(content, sort_keys=True, indent=2) + "\n")
    saved = load_run_spec(run_dir)
    if saved.spec != spec or saved.dexmachina_sha != LAB_DEXMACHINA_SHA:
        raise RuntimeError(f"{run_dir / RUN_SPEC_FILENAME} does not read back as the spec that was written")


def link_checkpoint(source: Path, step_dir: Path) -> None:
    actor = source / ACTOR_FILENAME
    if not actor.is_file():
        raise FileNotFoundError(
            f"{actor} does not exist. The checkpoint root must hold one directory per lab run, "
            f"with the checkpoint files in its {LAB_CHECKPOINT_SUBDIR} directory. Point --checkpoint-root at it."
        )
    if actor.resolve().parent.name != step_dir.name:
        raise ValueError(
            f"{actor} resolves into {actor.resolve().parent.name!r}, but the run's last saved step is "
            f"{step_dir.name!r}. The checkpoint would be evaluated under the wrong step, and the env steps computed "
            f"from it would be wrong. Link {LAB_CHECKPOINT_SUBDIR} to the {step_dir.name} directory of the run."
        )
    step_dir.mkdir(parents=True)
    for file in sorted(source.glob("*.pt")):
        (step_dir / file.name).symlink_to(file.resolve())


def known_differences_from_training(run: LabRun) -> list[str]:
    update = UPDATE_VARIANTS.get(run.flashsac_commit)
    if update is None:
        raise ValueError(
            f"The FlashSAC commit {run.flashsac_commit} of this run is not one of {sorted(UPDATE_VARIANTS)}. Its "
            "update.py is unknown, so the differences from training cannot be listed. Check what the commit changed "
            "and add it to UPDATE_VARIANTS."
        )
    return [
        VELOCITY_OBSERVATION_DIFFERENCE,
        f"FlashSAC commit {run.flashsac_commit} (the commit of the run's eval record, not of training): {update}.",
    ]


def write_run_files(name: str, run: LabRun, checkpoint_root: Path, out_dir: Path) -> Path:
    """Writes one run's files and returns its step directory, the argument of `evaluate_benchmark`."""
    wandb_config = run.config
    config = hydra_config(name, wandb_config)
    spec, reward_overrides, defaulted = reconstruct_spec(name, wandb_config)
    run_dir = out_dir / name
    write_spec_file(run_dir, spec, reward_overrides)
    OmegaConf.save(OmegaConf.create(config), run_dir / RUN_CONFIG_FILENAME, resolve=True)
    marker = {
        "reconstructed_from": "the W&B config of the run, not saved by the run itself",
        "dexmachina_sha_of_lab_code": LAB_DEXMACHINA_SHA,
        "set_by_lab_code": {"action_mode": LAB_ACTION_MODE, "max_epochs": LAB_MAX_EPOCHS},
        "derived_from_clip": derived_from_clip(spec),
        "dexmachina_dirty": f"unknown: {DIRTY_UNKNOWN_REASON}",
        "known_differences_from_training": known_differences_from_training(run),
        "filled_from_envspec_defaults": defaulted,
    }
    (run_dir / RECONSTRUCTED_FILENAME).write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n")
    step_dir = run_dir / CHECKPOINT_DIRNAME.format(step=checkpoint_step(wandb_config))
    link_checkpoint(checkpoint_root / name / LAB_CHECKPOINT_SUBDIR, step_dir)
    return step_dir


def run_names(checkpoint_root: Path) -> list[str]:
    names = sorted(path.name for path in checkpoint_root.iterdir() if path.is_dir())
    if not names:
        raise FileNotFoundError(f"{checkpoint_root} has no run directories. Point --checkpoint-root at the lab runs.")
    return names


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True, help="W&B export JSON (export_wandb_runs.py)")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError(f"{args.out_dir} exists already; choose a new --out-dir.")
    configs = lab_runs(args.export)
    names = run_names(args.checkpoint_root)
    missing = [name for name in names if name not in configs]
    if missing:
        raise ValueError(
            f"No {LAB_GROUP} run in {args.export} is named {missing}. Their W&B configs are needed to rebuild the "
            "files; export the project again, or remove these directories from the checkpoint root."
        )
    for name in names:
        step_dir = write_run_files(name, configs[name], args.checkpoint_root, args.out_dir)
        print(f"{name}: wrote {step_dir}")


if __name__ == "__main__":
    main()

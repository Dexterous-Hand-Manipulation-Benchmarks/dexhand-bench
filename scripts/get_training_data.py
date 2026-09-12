#!/usr/bin/env python3
"""
Full progression data exporter for DexMachina + FlashSAC runs.
Uses argument-free scan_history() with client-side scalar extraction
to ensure zero-loss capture of all physical and RL telemetry.
"""

import wandb
import pandas as pd

ENTITY = "schrapsi-kit"
PROJECT = "dexmachina_clean"
GROUP = "FlashSAC"

def parse_run_number(name: str) -> int:
    try:
        return int(name.split("-")[-1])
    except (ValueError, IndexError):
        return -1

def extract_cfg(config: dict, key: str, default=None):
    if key in config:
        return config[key]
    bench_key = f"benchmark/{key}"
    if bench_key in config:
        return config[bench_key]
    if "env" in config and isinstance(config["env"], dict):
        if key in config["env"]:
            return config["env"][key]
    return default

def fetch_all_progression_data(entity: str, project: str, group: str) -> pd.DataFrame:
    api = wandb.Api(timeout=120)
    runs = api.runs(f"{entity}/{project}", filters={"group": group})

    print(f"[INFO] Found {len(runs)} runs in group '{group}'. Starting export...")

    all_records = []
    processed_runs = 0
    last_skipped = False

    for run in runs:
        run_number = parse_run_number(run.name)
        # Skip outdated debug runs prior to milestone 100
        if run_number < 100:
            if not last_skipped:
                print(f"[SKIP] Skipping run(s): ({run.name})", end="")
                last_skipped = True
                continue
            print(f", ({run.name})", end="")
            last_skipped = True
            continue
        print("")
        last_skipped = False
        # Skip runs killed within initial 2 minutes
        runtime = run.summary.get("_runtime", getattr(run, "runtime", 0))
        if getattr(run, "state", "") == "killed" or runtime < 120:
            print(f"[SKIP] Short/killed run: {run.name}")
            continue

        # Extract metadata
        hand = extract_cfg(run.config, "hand", "unknown")
        clip = extract_cfg(run.config, "clip", "unknown")
        seed = extract_cfg(run.config, "seed", "unknown")
        actor_blocks = extract_cfg(run.config, "actor_blocks", None)
        critic_blocks = extract_cfg(run.config, "critic_blocks", None)
        actor_dim = extract_cfg(run.config, "actor_hidden_dim", None)
        critic_dim = extract_cfg(run.config, "critic_hidden_dim", None)
        gamma = extract_cfg(run.config, "gamma", None)

        tags = run.tags if run.tags else []
        param_group = "128d_2blk_params" if ("original_params" in tags or actor_blocks == 2) else "512d_6blk_params"

        print(f"[FETCHING] {run.name:<22} | Hand: {hand:<12} | Clip: {clip:<12} | Seed: {seed:<4} | Group: {param_group}")

        history_rows = []
        try:
            # Crucial: Call scan_history() with NO arguments to fetch all raw logged records
            for row in run.scan_history():
                clean_row = {
                    k: v for k, v in row.items()
                    if not isinstance(v, dict) and not str(k).endswith("video")
                }
                clean_row["run_id"] = run.id
                clean_row["run_name"] = run.name
                clean_row["hand"] = hand
                clean_row["clip"] = clip
                clean_row["seed"] = seed
                clean_row["param_group"] = param_group
                clean_row["actor_blocks"] = actor_blocks
                clean_row["critic_blocks"] = critic_blocks
                clean_row["actor_hidden_dim"] = actor_dim
                clean_row["critic_hidden_dim"] = critic_dim
                clean_row["gamma"] = gamma
                history_rows.append(clean_row)
        except Exception as e:
            print(f"  -> [ERROR] Failed reading history for {run.name}: {e}")
            continue

        if history_rows:
            df_run = pd.DataFrame(history_rows)
            all_records.append(df_run)
            processed_runs += 1
            print(f"  -> Successfully extracted {len(df_run)} time-series steps.")
        else:
            print(f"  -> [WARNING] No records found for {run.name}.")

    if not all_records:
        raise ValueError("No progression records retrieved across runs.")

    print(f"\n[INFO] Concatenating records across {processed_runs} valid runs...")
    combined_df = pd.concat(all_records, ignore_index=True)
    return combined_df

if __name__ == "__main__":
    df_progression = fetch_all_progression_data(ENTITY, PROJECT, GROUP)
    output_filename = "flashsac_dexmachina_complete_progression.csv"
    df_progression.to_csv(output_filename, index=False)
    print(f"\n[SUCCESS] Saved master dataset to '{output_filename}' ({df_progression.shape[0]} rows, {df_progression.shape[1]} columns).")
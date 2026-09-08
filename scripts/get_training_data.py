#!/usr/bin/env python3
"""
Extracts time-series progression metrics from Weights & Biases for FlashSAC runs.
Outputs a structured CSV containing training dynamics curves across seeds.
"""

import wandb
import pandas as pd
import numpy as np
from typing import List, Dict

ENTITY = "schrapsi-kit"
PROJECT = "dexmachina_clean"
GROUP = "FlashSAC"

# Metrics essential to defend training stability and convergence
TARGET_KEYS = [
    "_step",
    "global_env_steps",
    "train/critic_loss",
    "train/actor_loss",
    "train/q_min",
    "train/q_mean",
    "train/alpha",
    "train/entropy",
    "train/grad_norm_critic",
    "train/grad_norm_actor",
    "train/reward_scale",
    "eval/success_rate",
    "eval/add_score",
    "eval/auc_add",
    "curriculum/stage",
]


def fetch_progression_data(entity: str, project: str, group: str) -> pd.DataFrame:
    api = wandb.Api(timeout=60)
    runs = api.runs(f"{entity}/{project}", filters={"group": group})

    print(f"[INFO] Found {len(runs)} runs in group '{group}'. Extracting histories...")

    all_records = []


    for i, run in enumerate(runs):
        tags = run.config.get("tags", [])
        curriculum_metric = "benchmark/curriculum"
        is_curriculum = run.config.get(curriculum_metric, False)
        if not is_curriculum:
            print(f"Skipping run {run.name}, as curriculum was found or is False, meaning it is outdated. {curriculum_metric}: {is_curriculum}.")
            continue
        run_name = run.name
        run_number = int(run_name.split("-")[-1])
        if run_number < 100:
            print(f"Skipping run {run_name}, as it is outdated. (Non-outdated runs end with a number > 100. This run ended in {run_number}")
            continue
        seed = run.config.get("seed", "unknown")
        hand = run.config.get("env", {}).get("hand", "unknown")
        clip = run.config.get("env", {}).get("clip", "unknown")

        print(f"[FETCHING] Run: {run_name} (Hand: {hand}, Clip: {clip}, Seed: {seed})")

        # Scan full history without the 500-sample limit of run.history()
        history_data = []
        for row in run.scan_history(keys=TARGET_KEYS):
            row["run_id"] = run.id
            row["run_name"] = run_name
            row["hand"] = hand
            row["clip"] = clip
            row["seed"] = seed
            history_data.append(row)

        if history_data:
            df_run = pd.DataFrame(history_data)
            all_records.append(df_run)
            print(f"  -> Extracted {len(df_run)} progression data points.")
        else:
            print(f"  -> [WARNING] No history metrics found for {run_name}.")
            print(f"  -> full config for run {run_name}: {run.config}")

    if not all_records:
        raise ValueError("No progression records retrieved. Check project and group names.")

    combined_df = pd.concat(all_records, ignore_index=True)
    return combined_df


if __name__ == "__main__":
    df_progression = fetch_progression_data(ENTITY, PROJECT, GROUP)

    output_filename = "flashsac_dexmachina_progression.csv"
    df_progression.to_csv(output_filename, index=False)
    print(f"\n[SUCCESS] Saved complete progression dataset to '{output_filename}'.")
    print(f"Dataset shape: {df_progression.shape}")
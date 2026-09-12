#!/usr/bin/env python3
"""
Robust progression data exporter for DexMachina + FlashSAC WandB runs.

Features:
- Dual-strategy extraction: uses scan_history() for lossless finished runs and
  automatically falls back to run.history() for active/running runs or if scan_history()
  stops at <= 1 rows due to cloud sync delays or artifact errors.
- Robust scalar filtering: strictly filters out dicts, lists, wandb.Table, and media objects.
- CLI options to target specific runs (--run), newest run (--newest), or limit batch size.
"""

import argparse
import sys
import pandas as pd
import wandb

ENTITY = "schrapsi-kit"
PROJECT = "dexmachina_clean"
GROUP = "FlashSAC"


def parse_run_number(name: str) -> int:
    try:
        return int(name.split("-")[-1])
    except (ValueError, IndexError):
        return -1


def extract_cfg(config: dict, key: str, default=None):
    if not isinstance(config, dict):
        return default
    if key in config:
        return config[key]
    bench_key = f"benchmark/{key}"
    if bench_key in config:
        return config[bench_key]
    if "env" in config and isinstance(config["env"], dict):
        if key in config["env"]:
            return config["env"][key]
    return default


def is_scalar(v):
    if v is None:
        return True
    if isinstance(v, (int, float, str, bool)):
        return True
    type_name = type(v).__name__
    if any(t in type_name for t in ("int", "float", "bool", "str")):
        return True
    return False


def clean_history_row(row: dict) -> dict:
    clean = {}
    for k, v in row.items():
        k_str = str(k)
        # Skip media, video, and table artifacts
        if k_str.endswith(("video", "table", "media", "images", "image")):
            continue
        if any(skip in k_str for skip in ("update_table", "video", "artifact")):
            continue
        if is_scalar(v):
            clean[k] = v
    return clean


def extract_run_history(run, max_samples: int = 100000) -> list[dict]:
    """
    Extracts time-series progression history from a run.
    Uses scan_history() first. If that yields <= 1 rows (standard behavior for
    active/running runs whose wandb-history.jsonl has not yet flushed to storage,
    or runs encountering table deserialization issues), it falls back to run.history().
    """
    history_rows = []

    # Strategy 1: scan_history() with client-side row cleaning
    try:
        for row in run.scan_history(page_size=2000):
            try:
                c_row = clean_history_row(row)
                if c_row:
                    history_rows.append(c_row)
            except Exception:
                continue
    except Exception as e:
        print(f"    [INFO] scan_history notice for {run.name}: {e}")

    # Strategy 2: Fallback to run.history() if scan_history returned <= 1 row
    if len(history_rows) <= 1:
        print(f"    [INFO] scan_history returned {len(history_rows)} row(s). Falling back to run.history(samples={max_samples})...")
        try:
            records = run.history(samples=max_samples, pandas=False)
            fallback_rows = []
            for row in records:
                try:
                    c_row = clean_history_row(row)
                    if c_row:
                        fallback_rows.append(c_row)
                except Exception:
                    continue
            if len(fallback_rows) > len(history_rows):
                history_rows = fallback_rows
                print(f"    [SUCCESS] Fallback retrieved {len(history_rows)} steps from active history.")
        except Exception as e:
            print(f"    [WARN] run.history fallback failed: {e}")

    return history_rows


def fetch_progression_data(
    entity: str = ENTITY,
    project: str = PROJECT,
    group: str = GROUP,
    target_run: str = None,
    newest_only: bool = False,
    limit: int = None,
    min_runtime: int = 30,
    include_killed: bool = False,
) -> pd.DataFrame:
    api = wandb.Api(timeout=120)

    if target_run:
        print(f"[INFO] Looking for specific run '{target_run}'...")
        runs = api.runs(f"{entity}/{project}", filters={"display_name": target_run})
        if not runs:
            runs = [r for r in api.runs(f"{entity}/{project}") if r.name == target_run or r.id == target_run]
        if not runs:
            raise ValueError(f"Run '{target_run}' not found in {entity}/{project}")
        runs_to_process = runs[:1]
    else:
        filters = {}
        if group:
            filters["group"] = group
        runs = api.runs(f"{entity}/{project}", filters=filters, order="-created_at")
        print(f"[INFO] Found {len(runs)} runs matching filters={filters}. Starting export...")

        if newest_only:
            runs_to_process = runs[:1]
        elif limit:
            runs_to_process = runs[:limit]
        else:
            runs_to_process = runs

    all_records = []
    processed_runs = 0

    for run in runs_to_process:
        runtime = run.summary.get("_runtime", getattr(run, "runtime", 0))
        state = getattr(run, "state", "unknown")

        # Skip short killed runs only if not explicitly targeting a run/newest
        if not target_run and not newest_only and not include_killed:
            if state == "killed" and runtime < min_runtime:
                print(f"[SKIP] Short killed run: {run.name} ({runtime:.1f}s)")
                continue

        # Extract config metadata
        config = run.config or {}
        hand = extract_cfg(config, "hand", "unknown")
        clip = extract_cfg(config, "clip", "unknown")
        seed = extract_cfg(config, "seed", "unknown")
        actor_blocks = extract_cfg(config, "actor_blocks", None)
        critic_blocks = extract_cfg(config, "critic_blocks", None)
        actor_dim = extract_cfg(config, "actor_hidden_dim", None)
        critic_dim = extract_cfg(config, "critic_hidden_dim", None)
        gamma = extract_cfg(config, "gamma", None)

        tags = run.tags if run.tags else []
        param_group = "128d_2blk_params" if ("original_params" in tags or actor_blocks == 2) else "512d_6blk_params"

        print(f"\n[FETCHING] {run.name:<22} | State: {state:<9} | Runtime: {runtime:<6.0f}s | Hand: {hand:<12} | Clip: {clip}")

        history_rows = extract_run_history(run)

        if not history_rows:
            print(f"  -> [WARNING] No history records retrieved for {run.name}.")
            continue

        # Attach metadata to all rows
        metadata = {
            "run_id": run.id,
            "run_name": run.name,
            "hand": hand,
            "clip": clip,
            "seed": seed,
            "param_group": param_group,
            "actor_blocks": actor_blocks,
            "critic_blocks": critic_blocks,
            "actor_hidden_dim": actor_dim,
            "critic_hidden_dim": critic_dim,
            "gamma": gamma,
        }
        for r in history_rows:
            r.update(metadata)

        df_run = pd.DataFrame(history_rows)
        all_records.append(df_run)
        processed_runs += 1
        print(f"  -> Successfully extracted {len(df_run)} progression steps.")

    if not all_records:
        raise ValueError("No progression records retrieved across runs.")

    print(f"\n[INFO] Concatenating records across {processed_runs} valid runs...")
    combined_df = pd.concat(all_records, ignore_index=True)
    return combined_df


def main():
    parser = argparse.ArgumentParser(description="Export progression telemetry from WandB for DexMachina runs.")
    parser.add_argument("--entity", default=ENTITY, help="WandB entity (default: %(default)s)")
    parser.add_argument("--project", default=PROJECT, help="WandB project (default: %(default)s)")
    parser.add_argument("--group", default=GROUP, help="WandB group (default: %(default)s)")
    parser.add_argument("--run", type=str, default=None, help="Target specific run name or ID (e.g. 'efficient-eon-178')")
    parser.add_argument("--newest", action="store_true", help="Extract only the single newest run")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of runs to extract")
    parser.add_argument("--min-runtime", type=int, default=30, help="Minimum runtime in seconds to include (default: 30)")
    parser.add_argument("--include-killed", action="store_true", help="Include short/killed runs")
    parser.add_argument(
        "--output",
        type=str,
        default="flashsac_dexmachina_complete_progression.csv",
        help="Path to save output CSV (default: %(default)s)",
    )
    args = parser.parse_args()

    df_progression = fetch_progression_data(
        entity=args.entity,
        project=args.project,
        group=args.group,
        target_run=args.run,
        newest_only=args.newest,
        limit=args.limit,
        min_runtime=args.min_runtime,
        include_killed=args.include_killed,
    )

    df_progression.to_csv(args.output, index=False)
    print(f"\n[SUCCESS] Saved master dataset to '{args.output}' ({df_progression.shape[0]} rows, {df_progression.shape[1]} columns).")


if __name__ == "__main__":
    main()
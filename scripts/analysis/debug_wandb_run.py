import sys

import wandb
import pandas as pd

from dexhand_bench.config import load_config, wandb_settings

_wandb = wandb_settings(load_config())
GROUP = "FlashSAC"
RUN_ID_OR_NAME = sys.argv[1] if len(sys.argv) > 1 else "jumping-bee-137"

api = wandb.Api()
ENTITY = _wandb["WANDB_ENTITY"] or api.default_entity
PROJECT = _wandb["WANDB_PROJECT"]

print(f"[*] Querying run: {RUN_ID_OR_NAME} in {ENTITY}/{PROJECT}...")
runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": RUN_ID_OR_NAME})

if not runs:
    print(f"[!] No run found with display_name: {RUN_ID_OR_NAME}")
    print("[*] Attempting fallback search across all runs matching name...")
    runs = [r for r in api.runs(f"{ENTITY}/{PROJECT}") if r.name == RUN_ID_OR_NAME]

if not runs:
    print(f"[ERROR] Could not locate run '{RUN_ID_OR_NAME}'. Check ENTITY and PROJECT names.")
    exit(1)

run = runs[0]
print(f"[+] Run found: ID={run.id}, State={run.state}")

def debug_get_wandb_run():
    print("\n--- 1. Summary Keys ---")
    summary_keys = list(run.summary.keys())
    print(f"Total summary keys: {len(summary_keys)}")
    print(f"Summary keys preview: {summary_keys[:20]}")

    print("\n--- 2. Raw History Sample (No Key Filtering) ---")
    try:
        raw_history = run.history(nrows=5)
        print(f"Default history() row count: {len(raw_history)}")
        if not raw_history.empty:
            print("Available columns in raw history:")
            for col in sorted(raw_history.columns):
                print(f"  - {col}")
            print("\nFirst row snippet:")
            print(raw_history.iloc[0].dropna().to_dict())
        else:
            print("[!] Default history() returned an empty DataFrame.")
    except Exception as e:
        print(f"[ERROR] run.history() failed: {e}")

    print("\n--- 3. scan_history() Check ---")
    try:
        scan_iter = run.scan_history()
        first_row = next(iter(scan_iter), None)
        if first_row is not None:
            print(f"scan_history() retrieved records successfully.")
            print(f"Sample keys in scan_history row: {list(first_row.keys())}")
        else:
            print("[!] scan_history() produced 0 records.")
    except Exception as e:
        print(f"[ERROR] run.scan_history() failed: {e}")

def test_extract_single_run():
    # Test 1: Direct wandb-history.jsonl file download
    print("\n--- TEST 1: Direct File Download ('wandb-history.jsonl') ---")
    import tempfile, json, gzip
    file_records = []
    try:
        target_f = None
        for f in run.files():
            if f.name in ("wandb-history.jsonl", "wandb-history.jsonl.gz"):
                target_f = f
                break
        if target_f:
            print(f"Found {target_f.name} ({target_f.size / 1024:.1f} KB). Downloading...")
            with tempfile.TemporaryDirectory() as tmpdir:
                dl = target_f.download(root=tmpdir, replace=True)
                dl_path = dl.name
                dl.close()
                opener = gzip.open if target_f.name.endswith(".gz") else open
                with opener(dl_path, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            file_records.append(json.loads(line.strip()))
            print(f"Result: {len(file_records)} records yielded via file download.")
        else:
            print("[!] No wandb-history.jsonl file found on run.")
    except Exception as e:
        print(f"[ERROR] File download failed: {e}")

    # Test 2: run.history(samples=5000, pandas=False)
    print("\n--- TEST 2: run.history(samples=5000) [GraphQL single request] ---")
    try:
        hist = run.history(samples=5000, pandas=False)
        print(f"Result: {len(hist)} records yielded via run.history().")
    except Exception as e:
        print(f"[ERROR] run.history() failed: {e}")

    # Test 3: Client-side filtering of scalar metrics
    print("\n--- TEST 3: Client-side scalar extraction ---")
    records_to_use = file_records if file_records else run.history(samples=5000, pandas=False)
    records = []
    for row in records_to_use:
        clean_row = {
            k: v for k, v in row.items()
            if not isinstance(v, dict) and not str(k).endswith(("video", "table", "media", "image"))
        }
        records.append(clean_row)

    df = pd.DataFrame(records)
    print(f"Result: Extracted DataFrame with shape {df.shape}")

    print("\n--- Sample Extracted Telemetry (First 5 Non-Empty Rows) ---")
    display_cols = [
        c for c in [
            "_step", "global/env_step", "eval/return", "eval/auc_add",
            "eval/success_2cm", "eval/success_5cm", "env/kp", "env/pos_dist",
            "train/actor_loss", "train/critic_loss"
        ] if c in df.columns
    ]

    # Display non-null rows
    if display_cols:
        sample_df = df[display_cols].dropna(how="all").head(5)
        print(sample_df.to_string(index=False))

    print("\n" + "=" * 60)
    if len(df) > 0:
        print("[UNIT TEST PASSED] History data is fully accessible.")
    else:
        print("[UNIT TEST FAILED] Unable to extract history rows.")
    print("=" * 60)

if __name__ == "__main__":
    debug_get_wandb_run()
    test_extract_single_run()
import wandb
import pandas as pd

ENTITY = "schrapsi-kit"
PROJECT = "dexmachina_clean"
GROUP = "FlashSAC"
RUN_ID_OR_NAME = "jumping-bee-137"

api = wandb.Api()

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
    # Test 1: scan_history() without arguments
    print("\n--- TEST 1: scan_history() [No Arguments] ---")
    count_test1 = 0
    for row in run.scan_history():
        count_test1 += 1
    print(f"Result: {count_test1} records yielded.")

    # Test 2: scan_history(keys=...) with media keys included
    print("\n--- TEST 2: scan_history(keys=...) [Including 'eval/video'] ---")
    keys_with_media = ["_step", "global/env_step", "eval/return", "eval/video"]
    count_test2 = 0
    for row in run.scan_history(keys=keys_with_media):
        count_test2 += 1
    print(f"Result: {count_test2} records yielded.")

    # Test 3: Client-side filtering of scalar metrics
    print("\n--- TEST 3: scan_history() with client-side scalar extraction ---")
    records = []
    for row in run.scan_history():
        # Remove complex media dictionaries (videos/images) from row
        clean_row = {
            k: v for k, v in row.items()
            if not isinstance(v, dict) and not k.endswith("video")
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
    sample_df = df[display_cols].dropna(how="all").head(5)
    print(sample_df.to_string(index=False))

    print("\n" + "=" * 60)
    if count_test1 > 0 and len(df) > 0:
        print("[UNIT TEST PASSED] History data is fully accessible via client-side filtering.")
    else:
        print("[UNIT TEST FAILED] Unable to extract history rows.")
    print("=" * 60)

if __name__ == "__main__":
    test_extract_single_run()
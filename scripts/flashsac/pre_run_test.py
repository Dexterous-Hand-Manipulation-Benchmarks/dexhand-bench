import os
import sys
import subprocess
import shutil

from dexhand_bench.config import ROOT

FLASHSAC_DIR = os.path.join(ROOT, "algorithms", "FlashSAC")
DEXMACHINA_ASSETS = os.path.join(ROOT, "environments", "dexmachina", "dexmachina", "assets")
# Run with the FlashSAC environment: uv run --exact --extra flashsac python scripts/flashsac/pre_run_test.py
PYTHON_EXEC = sys.executable

HANDS = ["allegro_hand", "dex3_hand", "inspire_hand", "xhand"]
CLIPS = ["box-30-230", "mixer-0-150"]

print("=" * 70)
print(" PRE-FLIGHT COMPREHENSIVE SMOKE TEST (FlashSAC + Dexmachina)")
print(f" [*] Using Python: {PYTHON_EXEC}")
print("=" * 70)

results = []

for hand in HANDS:
    for clip in CLIPS:
        config_name = f"{hand} | {clip}"
        obj_name = clip.split("-")[0]
        data_file = os.path.join(DEXMACHINA_ASSETS, f"contact_retarget/{hand}/s01/{obj_name}_use_01.npy")

        # 1. Check Asset File Existence
        if not os.path.exists(data_file):
            print(f"[-] {config_name:32} : [SKIPPED - No Retarget Asset]")
            results.append((config_name, "SKIPPED (No Asset)"))
            continue

        print(f"[*] Testing: {config_name:28} ... ", end="", flush=True)

        # 2. Run Miniature Headless Pipeline (Physics + NaN Guard + SAC Loss Update + Checkpoint Save)
        cmd = [
            PYTHON_EXEC,
            "train.py",
            "--config_name", "flashSAC_base",
            "--overrides", "seed=0",
            "--overrides", "logger_type=tensorboard",
            "--overrides", "env=dexmachina",
            "--overrides", f"env.hand={hand}",
            "--overrides", f"env.clip={clip}",
            "--overrides", "env.retarget_name=para",
            "--overrides", "num_train_envs=16",
            "--overrides", "num_env_steps=800",
            "--overrides", "num_eval_episodes=0",
            "--overrides", "num_record_episodes=0",
            "--overrides", "agent.buffer_min_length=32",
            "--overrides", "agent.sample_batch_size=16",
            "--overrides", "agent.actor_hidden_dim=64",
            "--overrides", "agent.critic_hidden_dim=64",
            "--overrides", "agent.actor_num_blocks=1",
            "--overrides", "agent.critic_num_blocks=1",
            "--overrides", "save_checkpoint_per_interaction_step=25",
        ]

        try:
            res = subprocess.run(
                cmd,
                cwd=FLASHSAC_DIR,
                capture_output=True,
                text=True,
                timeout=120
            )

            if res.returncode == 0:
                print("[PASS]")
                results.append((config_name, "PASS"))
            else:
                print("[FAILED]")
                print("\n" + "-" * 30 + " ERROR TRACEBACK " + "-" * 30)
                # Print the most relevant error lines
                err_output = res.stderr if res.stderr else res.stdout
                print("\n".join(err_output.strip().split("\n")[-15:]))
                print("-" * 77 + "\n")
                results.append((config_name, "FAIL"))
        except subprocess.TimeoutExpired:
            print("[TIMEOUT (>120s)]")
            results.append((config_name, "TIMEOUT"))
        except Exception as e:
            print(f"[ERROR: {e}]")
            results.append((config_name, "ERROR"))

print("\n" + "=" * 70)
print(" 📋 SMOKE TEST SUMMARY")
print("=" * 70)
all_passed = True
for name, status in results:
    badge = "PASS" if status == "PASS" else ("Skipped" if "SKIPPED" in status else "")
    print(f" {badge} {name:34} : {status}")
    if status in ["FAIL", "TIMEOUT", "ERROR"]:
        all_passed = False
print("=" * 70)

if all_passed:
    print("ALL COMPATIBLE BENCHMARK CONFIGS PASSED!")
    sys.exit(0)
else:
    print("ISSUES DETECTED. Review failing configurations above.")
    sys.exit(1)
import os
import glob
import subprocess
import wandb

# --- W&B Target Configuration ---
ENTITY = "schrapsi-kit"
PROJECT = "dexmachina"
GROUP_NAME = "FlashSAC"

# --- Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELS_BASE_DIR = os.path.join(PROJECT_ROOT, "FlashSAC", "models", "FlashSAC")
EVAL_SCRIPT = os.path.join(PROJECT_ROOT, "envs", "FlashSAC", "eval_and_render.py")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "FlashSAC", "configs")


def get_config_val(run_config, key_path, default=None):
    """Safely extracts nested or dot-notated parameters from W&B run config."""
    if key_path in run_config:
        return run_config[key_path]
    parts = key_path.split(".")
    curr = run_config
    for part in parts:
        if isinstance(curr, dict) and part in curr:
            curr = curr[part]
        else:
            return default
    return curr if curr is not None else default


def main():
    api = wandb.Api()
    print(f"[INFO] Fetching runs from W&B ({ENTITY}/{PROJECT}) filtered by group '{GROUP_NAME}'...")

    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"config.group_name": GROUP_NAME})
    print(f"[INFO] Found {len(runs)} runs matching criteria.")

    runs = reversed(runs)

    for run in runs:
        exp_name = get_config_val(run.config, "exp_name")
        hand = get_config_val(run.config, "env.hand") or get_config_val(run.config, "env_kwargs.env_cfg.hand")
        clip = get_config_val(run.config, "env.clip") or get_config_val(run.config, "env_kwargs.env_cfg.clip")

        if not exp_name or not hand or not clip:
            print(f"[WARNING] Run {run.name} ({run.id}) is missing required config keys. Skipping.")
            continue

        run_dir = os.path.join(MODELS_BASE_DIR, exp_name, "dexmachina", hand)

        if not os.path.exists(run_dir):
            print(f"[SKIP] Local model directory not found for {run.name}: {run_dir}")
            continue

        ts_folders = sorted([d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))])
        if not ts_folders:
            print(f"[SKIP] No timestamp folders found in {run_dir}")
            continue

        latest_ts_folder = os.path.join(run_dir, ts_folders[-1])

        step_dirs = glob.glob(os.path.join(latest_ts_folder, "step*"))
        if not step_dirs:
            print(f"[SKIP] No 'step*' folders found in {latest_ts_folder}.")
            continue

        step_dirs = sorted(step_dirs, key=lambda x: int(os.path.basename(x).replace("step", "")))

        # --- Dynamically extract neural network architecture parameters ---
        actor_dim = get_config_val(run.config, "agent.actor_hidden_dim", 128)
        critic_dim = get_config_val(run.config, "agent.critic_hidden_dim", 256)
        actor_blocks = get_config_val(run.config, "agent.actor_num_blocks", 2)
        critic_blocks = get_config_val(run.config, "agent.critic_num_blocks", 2)

        print("\n" + "=" * 60)
        print(f"[INFO] Processing Run: {run.name} | W&B ID: {run.id}")
        print(f"[INFO] Hand: {hand} | Clip: {clip}")
        print(f"[INFO] Configured Architecture -> Actor: {actor_dim}d ({actor_blocks}b) | Critic: {critic_dim}d ({critic_blocks}b)")
        print(f"[INFO] Found {len(step_dirs)} saved checkpoints.")
        print("=" * 60)

        for step_path in step_dirs:
            if not os.path.exists(os.path.join(step_path, "actor.pt")):
                print(f"  -> [SKIP] Checkpoint {os.path.basename(step_path)} is missing 'actor.pt'.")
                continue

            step_num = int(os.path.basename(step_path).replace("step", ""))
            print(f"  -> [EVAL] Evaluating and Rendering Step {step_num}...")

            cmd = [
                "python", EVAL_SCRIPT,
                "--config_path", CONFIG_DIR,
                "--checkpoint_path", step_path,
                "--wandb_run_id", run.id,
                "--step", str(step_num),
                "--num_episodes", "10",
                "--overrides", "env=dexmachina",
                "--overrides", f"env.hand={hand}",
                "--overrides", f"env.clip={clip}",
                "--overrides", "env.retarget_name=para",
                "--overrides", "env.num_record_envs=1",
                "--overrides", "agent.asymmetric_observation=false",
                "--overrides", f"agent.actor_hidden_dim={actor_dim}",
                "--overrides", f"agent.critic_hidden_dim={critic_dim}",
                "--overrides", f"agent.actor_num_blocks={actor_blocks}",
                "--overrides", f"agent.critic_num_blocks={critic_blocks}",
            ]

            try:
                subprocess.run(cmd, check=True, cwd=os.path.join(PROJECT_ROOT, "FlashSAC"))
            except subprocess.CalledProcessError as e:
                print(f"  -> [ERROR] Evaluation crashed for {step_path}: {e}")

        print(f"[SUCCESS] Finished processing {run.name}")


if __name__ == "__main__":
    main()
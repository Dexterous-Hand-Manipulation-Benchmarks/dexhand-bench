#!/bin/bash
set -e

# --- 1. Workspace & Path Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate Conda environment
CONDA_ENV="dexmachina_FlashSAC"
if [ -d "$HOME/miniconda3/envs/$CONDA_ENV" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate $CONDA_ENV
fi

# Base paths
MODELS_DIR="${PROJECT_ROOT}/FlashSAC/models/FlashSAC"
PYTHON_SCRIPT="${PROJECT_ROOT}/envs/FlashSAC/eval_and_render.py"

# --- 2. Evaluation Settings ---
# Choose mode: "AUTO" (evaluates all runs in MODELS_DIR) or "SPECIFIC" (uses SPECIFIC_RUNS array)
EVAL_MODE="SPECIFIC"

# Used if EVAL_MODE="SPECIFIC"
SPECIFIC_RUNS=(
    "allegro_hand_box-30-230_seed0"
    "allegro_hand_box-30-230_seed1000"
    "allegro_hand_box-30-230_seed2000"
)

# Optional: Map run names to existing WandB run IDs if videos are supposed to be uploaded to the specific runs
declare -A WANDB_RUN_IDS=(
    ["allegro_hand_box-30-230_seed0"]="36ctg6w7"
    ["allegro_hand_box-30-230_seed1000"]="rqq98vl5"
    ["allegro_hand_box-30-230_seed2000"]="tovswiem"
)

# Environment configuration defaults
HAND="allegro_hand"
CLIP="box-30-230"
RETARGET="para"
NUM_EPISODES=10

# --- 3. Determine Runs to Process ---
if [ "$EVAL_MODE" == "AUTO" ]; then
    echo "[INFO] Auto-discovering all runs in: ${MODELS_DIR}"
    RUN_FOLDERS=()
    for dir in "${MODELS_DIR}"/*; do
        if [ -d "$dir" ]; then
            RUN_FOLDERS+=("$(basename "$dir")")
        fi
    done
else
    echo "[INFO] Running evaluation for specified runs list."
    RUN_FOLDERS=("${SPECIFIC_RUNS[@]}")
fi

echo "[INFO] Found ${#RUN_FOLDERS[@]} run(s) to evaluate."
echo "===================================================="

# Navigate to FlashSAC root so Hydra configs and relative imports load correctly
cd "${PROJECT_ROOT}/FlashSAC"

# --- 4. Loop Through Runs ---
for RUN_NAME in "${RUN_FOLDERS[@]}"; do
    RUN_PATH="${MODELS_DIR}/${RUN_NAME}"

    if [ ! -d "$RUN_PATH" ]; then
        echo "[WARNING] Directory not found: ${RUN_PATH}. Skipping..."
        continue
    fi

    echo "[INFO] Processing run: ${RUN_NAME}"

    # Auto-detect the latest step directory (e.g., step48829) inside the nested subfolders
    STEP_DIR=$(find "$RUN_PATH" -type d -name "step*" | sort -V | tail -n 1)

    if [ -z "$STEP_DIR" ]; then
        echo "[WARNING] No 'step*' checkpoint directory found in ${RUN_PATH}. Skipping..."
        continue
    fi

    echo "[INFO] Using Checkpoint: ${STEP_DIR}"

    # Check for mapped WandB Run ID
    WANDB_ID="${WANDB_RUN_IDS[$RUN_NAME]}"
    WANDB_ARG=""
    if [ -n "$WANDB_ID" ]; then
        WANDB_ARG="--wandb_run_id ${WANDB_ID}"
        echo "[INFO] Syncing output to WandB Run ID: ${WANDB_ID}"
    fi

        # Run the evaluation and rendering script
    python "${PYTHON_SCRIPT}" \
        --config_path "${PROJECT_ROOT}/FlashSAC/configs" \
        --checkpoint_path "${STEP_DIR}" \
        ${WANDB_ARG} \
        --num_episodes ${NUM_EPISODES} \
        --overrides env=dexmachina \
        --overrides env.hand="${HAND}" \
        --overrides env.clip="${CLIP}" \
        --overrides env.retarget_name="${RETARGET}" \
        --overrides env.num_record_envs=1 \
        --overrides agent.asymmetric_observation=false

    echo "[INFO] Completed evaluation for ${RUN_NAME}."
    echo "===================================================="
done

echo "[SUCCESS] All evaluations completed!"
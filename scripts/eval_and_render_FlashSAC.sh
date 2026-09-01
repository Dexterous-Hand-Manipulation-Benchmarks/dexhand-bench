#!/bin/bash
set -e

# --- 1. Workspace & Path Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate Conda environment
eval "$(conda shell.bash hook)"
if [ "$CONDA_DEFAULT_ENV" != "dexmachina_FlashSAC" ]; then
    echo "[INFO] Activating Conda environment 'dexmachina_FlashSAC'..."
    conda activate dexmachina_FlashSAC
fi

MODELS_DIR="${PROJECT_ROOT}/FlashSAC/models/FlashSAC"
PYTHON_SCRIPT="${PROJECT_ROOT}/envs/FlashSAC/eval_and_render.py"
WANDB_PROJECT="dexmachina_clean"
WANDB_ENTITY="schrapsi-kit"

echo "[INFO] Scanning for completed runs in: ${MODELS_DIR}"
cd "${PROJECT_ROOT}/FlashSAC"

# --- 2. Iterate Over All Model Folders Dynamically ---
for RUN_PATH in "${MODELS_DIR}"/*; do
    if [ ! -d "$RUN_PATH" ]; then
        continue
    fi

    RUN_NAME=$(basename "$RUN_PATH")

    # Locate the latest checkpoint folder that actually contains actor.pt
    STEP_DIR=$(find "$RUN_PATH" -type f -name "actor.pt" -exec dirname {} + 2>/dev/null | sort -V | tail -n 1)

    if [ -z "$STEP_DIR" ]; then
        echo "[SKIP] No valid checkpoint containing actor.pt found in: ${RUN_NAME}"
        continue
    fi

    # Extract clean integer step number (e.g., step48820 -> 48820)
    STEP_NUM=$(basename "$STEP_DIR" | grep -oP '[0-9]+' || echo "0")

    # Parse Hand, Clip, and Seed from RUN_NAME (Format: {hand}_{clip}_seed{seed})
    SEED=$(echo "$RUN_NAME" | grep -oP 'seed\K[0-9]+' || echo "0")
    NAME_WITHOUT_SEED=$(echo "$RUN_NAME" | sed -E 's/_seed[0-9]+//')

    if [[ "$NAME_WITHOUT_SEED" =~ ^allegro_hand ]]; then
        HAND="allegro_hand"
    elif [[ "$NAME_WITHOUT_SEED" =~ ^inspire_hand ]]; then
        HAND="inspire_hand"
    elif [[ "$NAME_WITHOUT_SEED" =~ ^dex3_hand ]]; then
        HAND="dex3_hand"
    elif [[ "$NAME_WITHOUT_SEED" =~ ^xhand ]]; then
        HAND="xhand"
    else
        echo "[WARNING] Unknown hand in ${RUN_NAME}. Skipping..."
        continue
    fi

    CLIP=${NAME_WITHOUT_SEED#"${HAND}_"}

    echo "===================================================="
    echo "[PROCESSING] Run:  ${RUN_NAME}"
    echo "  -> Hand:         ${HAND}"
    echo "  -> Clip:         ${CLIP}"
    echo "  -> Seed:         ${SEED}"
    echo "  -> Checkpoint:   ${STEP_DIR} (Step ${STEP_NUM})"

    # --- 3. Dynamic WandB Run ID Lookup ---
    WANDB_RUN_ID=$(python3 -c "
import wandb
try:
    api = wandb.Api()
    runs = api.runs('${WANDB_ENTITY}/${WANDB_PROJECT}')
    matched = [r.id for r in runs if r.config.get('benchmark/hand') == '${HAND}' and r.config.get('benchmark/clip') == '${CLIP}' and int(r.config.get('benchmark/seed', -1)) == int('${SEED}')]
    if not matched:
        matched = [r.id for r in runs if r.name == '${RUN_NAME}' or r.config.get('exp_name') == '${RUN_NAME}']
    print(matched[0] if matched else '')
except Exception:
    print('')
")

    WANDB_ARG=""
    if [ -n "$WANDB_RUN_ID" ]; then
        echo "  -> WandB Run ID: ${WANDB_RUN_ID} (Matched)"
        WANDB_ARG="--wandb_run_id ${WANDB_RUN_ID}"
    else
        echo "  -> WandB Run ID: None found (Will log locally)"
    fi

    # --- 4. Execute Full Evaluation, ADD Metric Calculation & Video Rendering ---
    python "${PYTHON_SCRIPT}" \
        --config_path "configs" \
        --checkpoint_path "${STEP_DIR}" \
        ${WANDB_ARG} \
        --project "${WANDB_PROJECT}" \
        --entity "${WANDB_ENTITY}" \
        --step "${STEP_NUM}" \
        --num_episodes 1 \
        --overrides env=dexmachina \
        --overrides env.hand="${HAND}" \
        --overrides env.clip="${CLIP}" \
        --overrides env.retarget_name="para" \
        --overrides env.num_record_envs=1 \
        --overrides agent.asymmetric_observation=false \
        --overrides agent.actor_hidden_dim=512 \
        --overrides agent.critic_hidden_dim=512 \
        --overrides agent.actor_num_blocks=6 \
        --overrides agent.critic_num_blocks=6

    echo "[COMPLETED] Evaluation finished for: ${RUN_NAME}"
    echo "===================================================="
done

echo "[SUCCESS] All retroactive evaluations and video uploads complete!"
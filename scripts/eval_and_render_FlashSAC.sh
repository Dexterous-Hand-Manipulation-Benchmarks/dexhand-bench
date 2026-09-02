#!/usr/bin/env bash

# Auto-re-execute with bash if run via sh/dash
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

set -e

FORCE=false
if [ "$1" == "--force" ] || [ "$2" == "--force" ]; then
    FORCE=true
    echo "[INFO] Running in FORCE mode: existing evaluations will be re-rendered."
fi

# --- 1. Workspace & Path Setup ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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

# --- 2. Iterate Over All Model Folders ---
for RUN_PATH in "${MODELS_DIR}"/*; do
    if [ ! -d "$RUN_PATH" ]; then
        continue
    fi

    RUN_NAME=$(basename "$RUN_PATH")

    # Locate the LATEST timestamp run directory (e.g., seed0-0901-185646)
    LATEST_TS_DIR=$(find "$RUN_PATH" -type d -name "seed*" 2>/dev/null | sort -V | tail -n 1)

    if [ -z "$LATEST_TS_DIR" ]; then
        echo "[SKIP] No seed directories found in: ${RUN_NAME}"
        continue
    fi

    # Safeguard: Verify the run has finished training (step48820 exists)
    if [ ! -d "${LATEST_TS_DIR}/step48820" ]; then
        echo "[SKIP] Run is currently still training (step48820 not yet found in ${LATEST_TS_DIR}). Skipping..."
        continue
    fi

    # Find checkpoints strictly belonging to this latest run
    ALL_STEP_DIRS=($(find "$LATEST_TS_DIR" -maxdepth 1 -type d -name "step*" 2>/dev/null | sort -V))
    TOTAL_CKPTS=${#ALL_STEP_DIRS[@]}

    if [ "$TOTAL_CKPTS" -eq 0 ]; then
        echo "[SKIP] No checkpoint directories found in: ${LATEST_TS_DIR}"
        continue
    fi
    FINAL_VIDEO="${LATEST_TS_DIR}/step48820/step48820_results/eval_video.mp4"
    if [ "$FORCE" = false ] && [ "$TOTAL_CKPTS" -eq 1 ] && [ -f "$FINAL_VIDEO" ]; then
        echo "[SKIP] Run ${RUN_NAME} already fully evaluated (final video & metrics present). Skipping..."
        continue
    fi

    # Parse Hand, Clip, and Seed from RUN_NAME
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
    echo "[PROCESSING] Run: ${RUN_NAME} (${TOTAL_CKPTS} checkpoint(s) in latest session)"
    echo "  -> Hand:         ${HAND}"
    echo "  -> Clip:         ${CLIP}"
    echo "  -> Seed:         ${SEED}"
    echo "  -> Directory:    ${LATEST_TS_DIR}"

    # --- 3. Dynamic WandB Run ID Lookup (Targets Newest / Curriculum Runs) ---
    WANDB_RUN_ID=$(python3 -c "
import wandb
try:
    api = wandb.Api()
    runs = sorted(api.runs('${WANDB_ENTITY}/${WANDB_PROJECT}'), key=lambda r: getattr(r, 'created_at', ''), reverse=True)

    # Priority 1: Match Hand, Clip, Seed AND curriculum=True / 'curriculum' tag
    matched = [
        r.id for r in runs
        if (r.config.get('benchmark/hand') == '${HAND}' or r.config.get('hand') == '${HAND}')
        and (r.config.get('benchmark/clip') == '${CLIP}' or r.config.get('clip') == '${CLIP}')
        and int(r.config.get('benchmark/seed', r.config.get('seed', -1))) == int('${SEED}')
        and (r.config.get('curriculum') is True or r.config.get('benchmark/curriculum') is True or 'curriculum' in (r.tags or []))
    ]

    # Priority 2: Match newest Hand, Clip, Seed (Fallback)
    if not matched:
        matched = [
            r.id for r in runs
            if (r.config.get('benchmark/hand') == '${HAND}' or r.config.get('hand') == '${HAND}')
            and (r.config.get('benchmark/clip') == '${CLIP}' or r.config.get('clip') == '${CLIP}')
            and int(r.config.get('benchmark/seed', r.config.get('seed', -1))) == int('${SEED}')
        ]

    print(matched[0] if matched else '')
except Exception as e:
    print('')
")

    WANDB_ARG=""
    if [ -n "$WANDB_RUN_ID" ]; then
        echo "  -> Matched WandB Run ID: ${WANDB_RUN_ID}"
        WANDB_ARG="--wandb_run_id ${WANDB_RUN_ID}"
    else
        echo "  -> WandB Run ID: None found (Logging locally only)"
    fi

    # --- 4. Loop Through All Milestones (10% to 100%) ---
    CKPT_IDX=0
    for STEP_DIR in "${ALL_STEP_DIRS[@]}"; do
        CKPT_IDX=$((CKPT_IDX + 1))
        STEP_NUM=$(basename "$STEP_DIR" | grep -oP '[0-9]+' || echo "0")
        CKPT_VIDEO="${STEP_DIR}/step${STEP_NUM}_results/eval_video.mp4"

        # Checkpoint-level skip check
        if [ "$FORCE" = false ] && [ -f "$CKPT_VIDEO" ]; then
            echo "[SKIP] Milestone step${STEP_NUM} already rendered. Skipping rollout..."
        else
            echo "[INFO] [${CKPT_IDX}/${TOTAL_CKPTS}] Evaluating & Rendering Checkpoint: step${STEP_NUM}"

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
        fi
        # Auto-cleanup intermediate checkpoints to free disk space
        if [ "$CKPT_IDX" -lt "$TOTAL_CKPTS" ]; then
            echo "[CLEANUP] Removing rendered intermediate milestone: ${STEP_DIR}"
            rm -rf "${STEP_DIR}"
        else
            echo "[PRESERVE] Keeping final checkpoint: ${STEP_DIR}"
        fi
    done

    echo "[COMPLETED] All milestones finished for: ${RUN_NAME}"
    echo "===================================================="
done

echo "[SUCCESS] All evaluations and cleanup complete!"
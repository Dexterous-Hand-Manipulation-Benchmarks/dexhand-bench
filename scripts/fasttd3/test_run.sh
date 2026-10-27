#!/bin/bash
##################################################################################
# Dexmachina (Genesis GPU Simulator + FastTD3 Smoke Test)
# Runs a quick 200-step training + evaluation rollout to verify:
#   1. UniversalLogger metric interception & WandB syncing
#   2. Checkpoint creation
#   3. Automated ADD / AUC-ADD computation
#   4. Video rendering and upload to WandB
##################################################################################

set -e

# --- Auto-activate Conda Environment ---
DHB_EXTRA=fasttd3
source "$(dirname "$0")/../common.sh"

cd "${DHB_ROOT}/algorithms/FastTD3" || exit 1

echo "[INFO] Running from directory: $(pwd)"

hand="allegro_hand"
clip="box-30-230"
seed=0
run_exp_name="smoke_test_${hand}_${clip}_seed${seed}"

echo "========================================================"
echo "[START] Running FastTD3 Smoke Test: $run_exp_name"
echo "========================================================"

# 1. RUN QUICK 200-STEP TRAINING
python fast_td3/train.py \
    --env_name "dexmachina|${hand}|${clip}" \
    --exp_name "${run_exp_name}" \
    --project "${WANDB_PROJECT}" \
    --group_name "FastTD3_Test" \
    --seed "${seed}" \
    --num_envs 64 \
    --learning_starts 10 \
    --batch_size 2048 \
    --buffer_size 1024 \
    --num_updates 1 \
    --std_min 0.05 \
    --std_max 0.6 \
    --policy_noise 0.1 \
    --gamma 0.99 \
    --v_min 0.0 \
    --v_max 250.0 \
    --disable_bootstrap \
    --save_interval 100 \
    --render_interval 100 \
    --eval_interval 100 \
    --total_timesteps 200 \
    --max_epochs 1000

echo "[SUCCESS] Smoke test training completed for $run_exp_name"

# 2. RUN EVALUATION & RENDERING FOR SAVED CHECKPOINTS
echo "[START] Evaluating & Rendering checkpoints for: $run_exp_name"

# Auto-detect the exact WandB Run ID created by the training script
LATEST_WANDB_DIR=$(ls -td wandb/*run-* 2>/dev/null | head -n 1)
if [ -n "$LATEST_WANDB_DIR" ]; then
    WANDB_RUN_ID=$(basename "$LATEST_WANDB_DIR" | awk -F'-' '{print $NF}')
    WANDB_ARG="--wandb_run_id ${WANDB_RUN_ID}"
    echo "[INFO] Linking videos & metrics to WandB Run ID: ${WANDB_RUN_ID}"
else
    WANDB_ARG=""
    echo "[WARNING] Could not detect recent WandB run. Videos will not sync to the cloud."
fi

# Target ONLY the latest timestamp directory from this test run
RUN_PATH="models/FastTD3/${run_exp_name}"
LATEST_TS_DIR=$(ls -td "${RUN_PATH}"/*/ 2>/dev/null | head -n 1)

if [ -z "$LATEST_TS_DIR" ]; then
    echo "[ERROR] No checkpoint directory found in ${RUN_PATH}!"
    exit 1
fi

echo "[INFO] Evaluating checkpoints from: ${LATEST_TS_DIR}"

# 1. Evaluate intermediate step checkpoints
STEP_CKPTS=$(find "$LATEST_TS_DIR" -maxdepth 1 -type f -name "step*.pt" 2>/dev/null | sort -V)

for CKPT in $STEP_CKPTS; do
    STEP_NUM=$(basename "$CKPT" .pt | sed 's/step//')
    ENV_STEP=$(( STEP_NUM * 64 ))
    echo "[INFO] Evaluating checkpoint: ${CKPT} (Env Step: ${ENV_STEP})"

    python fast_td3/eval_dexmachina.py \
        --checkpoint "$CKPT" \
        --num_envs 20 \
        --env_name "dexmachina|${hand}|${clip}" \
        --record_video \
        ${WANDB_ARG} \
        --step "${ENV_STEP}"
done

# 2. Evaluate final checkpoint
if [ -f "${LATEST_TS_DIR}/final.pt" ]; then
    FINAL_CKPT="${LATEST_TS_DIR}/final.pt"
    TOTAL_ENV_STEPS=$(( 200 * 64 ))
    echo "[INFO] Evaluating final checkpoint: ${FINAL_CKPT} (Env Step: ${TOTAL_ENV_STEPS})"

    python fast_td3/eval_dexmachina.py \
        --checkpoint "$FINAL_CKPT" \
        --num_envs 20 \
        --env_name "dexmachina|${hand}|${clip}" \
        --record_video \
        ${WANDB_ARG} \
        --step "${TOTAL_ENV_STEPS}"
fi

echo "========================================================"
echo "[SUCCESS] FastTD3 Smoke Test completed successfully!"
echo "Check your W&B dashboard under project 'dexmachina', group 'FastTD3_Test'."
echo "========================================================"

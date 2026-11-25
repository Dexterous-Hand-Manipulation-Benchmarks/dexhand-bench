#!/bin/bash
##################################################################################
# Dexmachina (Genesis GPU Simulator + FastTD3 Smoke Test)
# Runs a quick 200-step training and the benchmark evaluation of its checkpoints, to verify:
#   1. UniversalLogger metric interception & WandB syncing
#   2. Checkpoint creation, with env_spec.json and wandb_run.json next to them
#   3. The shared benchmark evaluation, logged into the training run
##################################################################################

set -e

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
    --eval_interval 100 \
    --total_timesteps 200

echo "[SUCCESS] Smoke test training completed for $run_exp_name"

# 2. EVALUATE THE SAVED CHECKPOINTS ON THE SHARED BENCHMARK PROTOCOL
echo "[START] Evaluating the checkpoints of: $run_exp_name"

# Target ONLY the latest timestamp directory from this test run
RUN_PATH="models/FastTD3/${run_exp_name}"
LATEST_TS_DIR=$(ls -td "${RUN_PATH}"/*/ 2>/dev/null | head -n 1)

if [ -z "$LATEST_TS_DIR" ]; then
    echo "[ERROR] No checkpoint directory found in ${RUN_PATH}!"
    exit 1
fi

echo "[INFO] Evaluating checkpoints from: ${LATEST_TS_DIR}"

# Step checkpoints in training order, then final.pt
STEP_CKPTS=$(find "$LATEST_TS_DIR" -maxdepth 1 -type f -name "step*.pt" 2>/dev/null | sort -V)
for CKPT in $STEP_CKPTS "${LATEST_TS_DIR}/final.pt"; do
    echo "[INFO] Evaluating checkpoint: ${CKPT}"
    python fast_td3/eval_dexmachina.py --checkpoint "$CKPT" --wandb
done

echo "========================================================"
echo "[SUCCESS] FastTD3 Smoke Test completed successfully!"
echo "Check your W&B dashboard under project 'dexmachina', group 'FastTD3_Test'."
echo "========================================================"

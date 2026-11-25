#!/bin/bash
# Evaluates the final checkpoint of a finished FlashSAC training with the shared benchmark protocol, logs the result into
# the training's W&B run, and deletes the run's other checkpoints (only after the evaluation succeeded).
# Usage, from algorithms/FlashSAC with `python` the method's uv python (scripts/common.sh):
#   evaluate_run.sh <run directory> <W&B project> <device>
set -e

RUN_DIR="${1:?usage: evaluate_run.sh <run directory> <W&B project> <device>}"
PROJECT_NAME="${2:?usage: evaluate_run.sh <run directory> <W&B project> <device>}"
DEVICE="${3:?usage: evaluate_run.sh <run directory> <W&B project> <device>}"

# flash_rl.envs.dexmachina.WANDB_RUN_ID_FILENAME
WANDB_RUN_ID_FILE="${RUN_DIR}/wandb_run_id.txt"
if [ ! -f "$WANDB_RUN_ID_FILE" ]; then
    echo "[ERROR] ${WANDB_RUN_ID_FILE} does not exist: train.py saves it when the run starts, so the evaluation" >&2
    echo "        cannot tell which W&B run to log into. Check that training used logger_type=wandb." >&2
    exit 1
fi
WANDB_RUN_ID=$(cat "$WANDB_RUN_ID_FILE")

FINAL_STEP_DIR=$(find "$RUN_DIR" -maxdepth 1 -type d -name "step*" | sort -V | tail -n 1)
if [ -z "$FINAL_STEP_DIR" ]; then
    echo "[ERROR] No step directory in ${RUN_DIR}: the finished training saved no checkpoint." >&2
    exit 1
fi

echo "[START] Evaluating ${FINAL_STEP_DIR}"
python eval_dexmachina.py \
    --checkpoint_path "${FINAL_STEP_DIR}" \
    --wandb_run_id "${WANDB_RUN_ID}" \
    --project "${PROJECT_NAME}" \
    --device "${DEVICE}"

find "$RUN_DIR" -maxdepth 1 -type d -name "step*" ! -path "${FINAL_STEP_DIR}" -exec rm -rf {} +
echo "[PRESERVE] Kept the final checkpoint: ${FINAL_STEP_DIR}"

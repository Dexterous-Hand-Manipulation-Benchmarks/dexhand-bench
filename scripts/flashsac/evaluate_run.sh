#!/bin/bash
# Evaluates the final checkpoint of a finished FlashSAC training with the shared benchmark protocol, logs the result into
# the training's W&B run, and deletes the run's other checkpoints (only after the evaluation succeeded).
# Usage, from algorithms/FlashSAC with `python` the method's uv python (scripts/common.sh):
#   evaluate_run.sh <run directory> <W&B project> <device> <final interaction step>
set -e

USAGE="usage: evaluate_run.sh <run directory> <W&B project> <device> <final interaction step>"
RUN_DIR="${1:?$USAGE}"
PROJECT_NAME="${2:?$USAGE}"
DEVICE="${3:?$USAGE}"
FINAL_STEP="${4:?$USAGE}"

# flash_rl.envs.dexmachina.TRAINING_FINISHED_FILENAME
if [ ! -f "${RUN_DIR}/training_finished" ]; then
    echo "[ERROR] ${RUN_DIR}/training_finished does not exist: train.py writes it only after the last step, its checks and" >&2
    echo "        the final evaluation, so this training did not finish and its checkpoints are not the reported ones." >&2
    echo "        Look at the training's error, then delete the run directory and train again." >&2
    exit 1
fi

# flash_rl.envs.dexmachina.WANDB_RUN_ID_FILENAME
WANDB_RUN_ID_FILE="${RUN_DIR}/wandb_run_id.txt"
if [ ! -f "$WANDB_RUN_ID_FILE" ]; then
    echo "[ERROR] ${WANDB_RUN_ID_FILE} does not exist: train.py saves it when the run starts, so the evaluation" >&2
    echo "        cannot tell which W&B run to log into. Check that training used logger_type=wandb." >&2
    exit 1
fi
WANDB_RUN_ID=$(cat "$WANDB_RUN_ID_FILE")

FINAL_STEP_DIR="${RUN_DIR}/step${FINAL_STEP}"
if [ ! -d "$FINAL_STEP_DIR" ]; then
    echo "[ERROR] ${FINAL_STEP_DIR} does not exist: the finished training saved no checkpoint at its final step." >&2
    echo "        Check the final step number passed to this script against the training's num_env_steps." >&2
    exit 1
fi

echo "[START] Evaluating ${FINAL_STEP_DIR}"
python eval_dexmachina.py \
    --checkpoint_path "${FINAL_STEP_DIR}" \
    --wandb_run_id "${WANDB_RUN_ID}" \
    --project "${PROJECT_NAME}" \
    --device "${DEVICE}"

find "$RUN_DIR" -maxdepth 1 -type d -name "step*" ! -name "step${FINAL_STEP}" -exec rm -rf {} +
echo "[PRESERVE] Kept the final checkpoint: ${FINAL_STEP_DIR}"

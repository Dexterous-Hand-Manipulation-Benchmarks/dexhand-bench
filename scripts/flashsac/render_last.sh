#!/bin/bash
set -e

# 1. Use the FlashSAC uv environment
DHB_EXTRA=flashsac
source "$(dirname "$0")/../common.sh"

# 2. Navigate to the FlashSAC directory
cd "${DHB_ROOT}/algorithms/FlashSAC" || exit 1
echo "[INFO] Working directory is now: $(pwd)"

# 3. Target the correct run path
RUN_PATH="models/FlashSAC/allegro_hand_box-30-230_seed0/dexmachina/allegro_hand"
LATEST_RUN_DIR=$(ls -td ${RUN_PATH}/*/ 2>/dev/null | head -n 1)

if [ -z "$LATEST_RUN_DIR" ]; then
    echo "[ERROR] Could not find any timestamp directories in $RUN_PATH"
    exit 1
fi

echo "[INFO] Rendering checkpoints from: $LATEST_RUN_DIR"

for STEP_DIR in $(find "$LATEST_RUN_DIR" -maxdepth 1 -type d -name "step*" | sort -V); do
    STEP_NUM=$(basename "$STEP_DIR" | sed 's/step//')
    echo "Rendering $STEP_DIR..."
    python eval_dexmachina.py \
        --config_path "configs" \
        --checkpoint_path "${STEP_DIR}" \
        --step "${STEP_NUM}" \
        --num_episodes 10 \
        --overrides env=dexmachina \
        --overrides env.hand="allegro_hand" \
        --overrides env.clip="box-30-230" \
        --overrides env.retarget_name="para" \
        --overrides env.num_record_envs=1 \
        --overrides agent.asymmetric_observation=false \
        --overrides agent.actor_hidden_dim=512 \
        --overrides agent.critic_hidden_dim=512 \
        --overrides agent.actor_num_blocks=6 \
        --overrides agent.critic_num_blocks=6
done
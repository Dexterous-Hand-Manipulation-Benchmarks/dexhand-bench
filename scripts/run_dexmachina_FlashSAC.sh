#!/bin/bash
##################################################################################
# Dexmachina (Genesis GPU Simulator + FlashSAC)
##################################################################################

set -e

# --- Auto-activate Conda Environment ---
eval "$(conda shell.bash hook)"
if [ "$CONDA_DEFAULT_ENV" != "dexmachina_FlashSAC" ]; then
    echo "[INFO] Activating Conda environment 'dexmachina_FlashSAC'..."
    conda activate dexmachina_FlashSAC
fi

cd "$(dirname "$0")/../FlashSAC" || exit 1

echo "[INFO] Using Python from: $(which python)"
echo "[INFO] Running from directory: $(pwd)"

# --- Define hands and clips to benchmark ---
hands=(
    "allegro_hand"
    "dex3_hand"
    "inspire_hand"
    "xhand"
)

clips=(
    "box-30-230"
    "mixer-0-150"
)

seeds=( 0 1000 2000 )

for seed in "${seeds[@]}"; do
    for hand in "${hands[@]}"; do
        for clip in "${clips[@]}"; do

            # Extract the object name to safely check if retarget data exists
            obj_name=$(echo "$clip" | cut -d'-' -f1)
            data_file="../dexmachina/dexmachina/assets/contact_retarget/${hand}/s01/${obj_name}_use_01.npy"

            if [ ! -f "$data_file" ]; then
                echo "[SKIP] Retarget data not found: ${hand} + ${obj_name}. Moving to next..."
                continue
            fi

            run_exp_name="${hand}_${clip}_seed${seed}"
            echo "========================================================"
            echo "[START] Training: $run_exp_name"
            echo "========================================================"

            # 1. RUN HEADLESS TRAINING AT MAX SPEED
            python train.py \
                --config_name flashSAC_base \
                --overrides seed=${seed} \
                `#=== Logging (WandB) ===#` \
                --overrides logger_type='wandb' \
                --overrides project_name='dexmachina' \
                --overrides entity_name='schrapsi-kit' \
                --overrides group_name='FlashSAC' \
                --overrides exp_name=${run_exp_name} \
                `#=== Environment (Dexmachina GPU sim) ===#` \
                --overrides env=dexmachina \
                --overrides env.hand=${hand} \
                --overrides env.clip=${clip} \
                --overrides env.retarget_name='para' \
                --overrides num_env_steps=50_000_896 \
                --overrides num_train_envs=1024 \
                --overrides num_eval_envs=null \
                --overrides num_record_envs=null \
                --overrides num_eval_episodes=1024 \
                --overrides num_record_episodes=0 \
                `#=== Checkpointing (Save 10 times during training) ===#` \
                --overrides save_checkpoint_per_interaction_step=4882 \
                `#=== Reward Shaping ===#` \
                --overrides ++env.reward_cfg.force_penalty=2.0 \
                --overrides ++env.reward_cfg.action_penalty=0.05 \
                --overrides ++env.reward_cfg.imi_rew_weight=1.0 \
                --overrides ++env.reward_cfg.contact_rew_weight=1.0 \
                `#=== Agent (Expanded Architecture) ===#` \
                --overrides agent=flashSAC \
                --overrides agent.actor_hidden_dim=512 \
                --overrides agent.critic_hidden_dim=512 \
                --overrides agent.actor_num_blocks=6 \
                --overrides agent.critic_num_blocks=6 \
                --overrides agent.buffer_max_length=2_000_000 \
                --overrides agent.buffer_min_length=100_000 \
                --overrides agent.buffer_device_type='cuda' \
                --overrides agent.sample_batch_size=2048 \
                --overrides agent.use_amp=true \
                --overrides updates_per_interaction_step=2 \
                `#=== Benchmark default ===#` \
                --overrides agent.asymmetric_observation=false \
                --overrides gamma=0.99 \
                --overrides n_step=1

            echo "[SUCCESS] Training completed for $run_exp_name"

            # 2. RUN EVALUATION & RENDERING FOR ALL SAVED CHECKPOINTS
            echo "[START] Evaluating & Rendering all step milestones for: $run_exp_name"

            # Auto-detect the exact WandB Run ID created by the training script
            LATEST_WANDB_DIR=$(ls -td wandb/*run-* 2>/dev/null | head -n 1)
            if [ -n "$LATEST_WANDB_DIR" ]; then
                WANDB_RUN_ID=$(basename "$LATEST_WANDB_DIR" | awk -F'-' '{print $NF}')
                WANDB_ARG="--wandb_run_id ${WANDB_RUN_ID}"
                echo "[INFO] Linking videos to WandB Run ID: ${WANDB_RUN_ID}"
            else
                WANDB_ARG=""
                echo "[WARNING] Could not detect recent WandB run. Videos will not sync to the cloud."
            fi

            # Find ALL step directories sorted by step number
            RUN_PATH="models/FlashSAC/${run_exp_name}/dexmachina/${hand}"
            STEP_DIRS=$(find "$RUN_PATH" -type d -name "step*" 2>/dev/null | sort -V)

            if [ -z "$STEP_DIRS" ]; then
                echo "[ERROR] No checkpoint directories found in ${RUN_PATH}. Skipping evaluation."
                continue
            fi

            # Loop through EVERY checkpoint saved (10%, 20%, ..., 100%) and render a video
            for STEP_DIR in $STEP_DIRS; do
                echo "[INFO] Rendering evaluation video for checkpoint: ${STEP_DIR}"

                python ../envs/FlashSAC/eval_and_render.py \
                    --config_path "configs" \
                    --checkpoint_path "${STEP_DIR}" \
                    ${WANDB_ARG} \
                    --num_episodes 10 \
                    --overrides env=dexmachina \
                    --overrides env.hand="${hand}" \
                    --overrides env.clip="${clip}" \
                    --overrides env.retarget_name="para" \
                    --overrides env.num_record_envs=1 \
                    --overrides agent.asymmetric_observation=false
            done

            echo "[SUCCESS] All milestone evaluations completed for $run_exp_name"
            echo "--------------------------------------------------------"

        done
    done
done

echo "[FINISHED] All automated training and evaluations are complete!"
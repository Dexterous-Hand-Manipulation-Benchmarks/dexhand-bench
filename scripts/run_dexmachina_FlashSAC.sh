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

            run_exp_name="${hand}_${clip}_seed${seed}"

            # Skip only the first specific run
            #if [ "$hand" = "allegro_hand" ] && [ "$clip" = "box-30-230" ] && [ "$seed" -eq 0 ]; then
            #    echo "[SKIP] Skipping completed run: ${run_exp_name}"
            #    continue
            #fi

            # Extract the object name to safely check if retarget data exists
            obj_name=$(echo "$clip" | cut -d'-' -f1)
            data_file="../dexmachina/dexmachina/assets/contact_retarget/${hand}/s01/${obj_name}_use_01.npy"

            if [ ! -f "$data_file" ]; then
                echo "[SKIP] Retarget data not found: ${hand} + ${obj_name}. Moving to next..."
                continue
            fi

            echo "========================================================"
            echo "[START] Training: $run_exp_name"
            echo "========================================================"

            # 1. RUN HEADLESS TRAINING AT MAX SPEED
            python train.py \
                --config_name flashSAC_base \
                --overrides seed=${seed} \
                `#=== Logging (WandB) ===#` \
                --overrides logger_type='wandb' \
                --overrides project_name='dexmachina_clean' \
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
                `#=== Checkpointing (Save 10 milestone steps) ===#` \
                --overrides save_checkpoint_per_interaction_step=4882 \
                `#=== Reward Shaping (Full Dexmachina Match) ===#` \
                --overrides ++env.reward_cfg.force_penalty=0.1 \
                --overrides ++env.reward_cfg.action_penalty=0.01 \
                --overrides ++env.reward_cfg.imi_rew_weight=0.3 \
                --overrides ++env.reward_cfg.imi_wrist_weight=0.5 \
                --overrides ++env.reward_cfg.contact_rew_weight=3.0 \
                --overrides ++env.reward_cfg.contact_beta=10.0 \
                --overrides ++env.reward_cfg.bc_rew_weight=0.3 \
                --overrides ++env.reward_cfg.bc_beta=500.0 \
                --overrides ++env.reward_cfg.use_retarget_contact=true \
                --overrides ++env.reward_cfg.obj_pos_beta=10.0 \
                --overrides ++env.reward_cfg.obj_rot_beta=1.0 \
                --overrides ++env.reward_cfg.obj_arti_beta=5.0 \
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

            # 2. RUN EVALUATION & RENDERING WITH AUTOMATIC DISK CLEANUP
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

            RUN_PATH="models/FlashSAC/${run_exp_name}/dexmachina/${hand}"
            LATEST_TS_DIR=$(ls -td "${RUN_PATH}"/*/ 2>/dev/null | head -n 1)

            if [ -z "$LATEST_TS_DIR" ]; then
                echo "[ERROR] No checkpoint directories found in ${RUN_PATH}. Skipping evaluation."
                continue
            fi

            # Find and sort all checkpoint milestone directories
            STEP_DIRS=$(find "$LATEST_TS_DIR" -maxdepth 1 -type d -name "step*" 2>/dev/null | sort -V)
            TOTAL_CHECKPOINTS=$(echo "$STEP_DIRS" | wc -w)
            CHECKPOINT_COUNTER=0

            for STEP_DIR in $STEP_DIRS; do
                CHECKPOINT_COUNTER=$((CHECKPOINT_COUNTER + 1))
                STEP_NUM=$(basename "$STEP_DIR" | sed 's/step//')
                echo "[INFO] [${CHECKPOINT_COUNTER}/${TOTAL_CHECKPOINTS}] Rendering video for: ${STEP_DIR}"

                python ../envs/FlashSAC/eval_and_render.py \
                    --config_path "configs" \
                    --checkpoint_path "${STEP_DIR}" \
                    ${WANDB_ARG} \
                    --project "dexmachina_clean" \
                    --step "${STEP_NUM}" \
                    --num_episodes 1 \
                    --overrides env=dexmachina \
                    --overrides env.hand="${hand}" \
                    --overrides env.clip="${clip}" \
                    --overrides env.retarget_name="para" \
                    --overrides env.num_record_envs=1 \
                    --overrides agent.asymmetric_observation=false \
                    --overrides agent.actor_hidden_dim=512 \
                    --overrides agent.critic_hidden_dim=512 \
                    --overrides agent.actor_num_blocks=6 \
                    --overrides agent.critic_num_blocks=6

                # --- Remove intermediate checkpoints after rendering ---
                if [ "$CHECKPOINT_COUNTER" -lt "$TOTAL_CHECKPOINTS" ]; then
                    echo "[CLEANUP] Deleted rendered intermediate checkpoint: ${STEP_DIR}"
                    rm -rf "${STEP_DIR}"
                else
                    echo "[PRESERVE] Keeping final checkpoint: ${STEP_DIR}"
                fi
            done

            echo "[SUCCESS] Milestone evaluations & cleanup completed for $run_exp_name"
            echo "--------------------------------------------------------"

        done
    done
done

echo "[FINISHED] All automated training and evaluations are complete!"
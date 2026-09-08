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


# --- 1. WandB & Hydra Paths ---
PROJECT_NAME="dexmachina_clean"
ENTITY_NAME="schrapsi-kit"
GROUP_NAME="FlashSAC"
CONFIG_PATH="configs"
CONFIG_NAME="flashSAC_base"

# --- 2. Shared Agent Architecture --- (A: Appendix , T: Table from FlashSAC paper)
AGENT_NAME="flashSAC"
ACTOR_HIDDEN_DIM=128    # AT9
CRITIC_HIDDEN_DIM=256   # AT9
ACTOR_NUM_BLOCKS=2      # AT9
CRITIC_NUM_BLOCKS=2     # AT9
ASYMMETRIC_OBS="false"

# --- 3. Shared Environment Identity ---
ENV_TYPE="dexmachina"
RETARGET_NAME="para"

# --- 4. Training-Specific Hyperparameters ---
NUM_ENV_STEPS=50_000_896
NUM_TRAIN_ENVS=1024 # AT9
CHECKPOINT_INTERVAL=4882
USE_CURRICULUM="true"


# Auto-detect available CUDA devices
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "[INFO] Detected $NUM_GPUS GPUs. Distributing simulation to cuda:0 and buffer to cuda:1."
    SIM_DEVICE="cuda:0"
    BUFFER_DEVICE="cuda:1"
else
    echo "[INFO] Detected $NUM_GPUS GPU. Running both simulation and buffer on cuda:0."
    SIM_DEVICE="cuda:0"
    BUFFER_DEVICE="cuda:0"
fi

BUFFER_MAX_LENGTH=10_000_000   # AT9
BUFFER_MIN_LENGTH=100_000
BUFFER_DEVICE="cuda"
SAMPLE_BATCH_SIZE=2048  # AT9
USE_AMP="true"
UPDATES_PER_STEP=2  # AT9
GAMMA=0.99          # FlashSACs run_isaaclab (using dex hand tasks)
N_STEP=1      # AT9

# --- 5. Benchmark Loops ---
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

# Execution of runs
for seed in "${seeds[@]}"; do
    for hand in "${hands[@]}"; do
        for clip in "${clips[@]}"; do

            run_exp_name="${hand}_${clip}_seed${seed}_10M_official"

            obj_name=$(echo "$clip" | cut -d'-' -f1)
            data_file="../dexmachina/dexmachina/assets/contact_retarget/${hand}/s01/${obj_name}_use_01.npy"

            if [ ! -f "$data_file" ]; then
                echo "[SKIP] Retarget data not found: ${hand} + ${obj_name}. Moving to next..."
                continue
            fi

            SHARED_AGENT_OVERRIDES=(
                --overrides "agent=${AGENT_NAME}"
                --overrides "agent.actor_hidden_dim=${ACTOR_HIDDEN_DIM}"
                --overrides "agent.critic_hidden_dim=${CRITIC_HIDDEN_DIM}"
                --overrides "agent.actor_num_blocks=${ACTOR_NUM_BLOCKS}"
                --overrides "agent.critic_num_blocks=${CRITIC_NUM_BLOCKS}"
                --overrides "agent.asymmetric_observation=${ASYMMETRIC_OBS}"
            )

            SHARED_ENV_OVERRIDES=(
                --overrides "env=${ENV_TYPE}"
                --overrides "env.hand=${hand}"
                --overrides "env.clip=${clip}"
                --overrides "env.retarget_name=${RETARGET_NAME}"
            )

            echo "========================================================"
            echo "[START] Training: $run_exp_name"
            echo "========================================================"

            # 1. RUN HEADLESS TRAINING
            python train.py \
                --config_path "${CONFIG_PATH}" \
                --config_name "${CONFIG_NAME}" \
                --overrides "seed=${seed}" \
                `#=== Logging (WandB) ===#` \
                --overrides "logger_type=wandb" \
                --overrides "project_name=${PROJECT_NAME}" \
                --overrides "entity_name=${ENTITY_NAME}" \
                --overrides "group_name=${GROUP_NAME}" \
                --overrides "exp_name=${run_exp_name}" \
                `#=== Architecture & Environment Overrides ===#` \
                "${SHARED_ENV_OVERRIDES[@]}" \
                "${SHARED_AGENT_OVERRIDES[@]}" \
                `#=== Device & Scaling Setup ===#` \
                --overrides "env.rl_device=${SIM_DEVICE}" \
                --overrides "agent.device=${BUFFER_DEVICE}" \
                --overrides "agent.buffer_device_type=${BUFFER_DEVICE}" \
                --overrides "agent.buffer_max_length=${BUFFER_MAX_LENGTH}" \
                --overrides "agent.buffer_min_length=${BUFFER_MIN_LENGTH}" \
                --overrides "agent.sample_batch_size=${SAMPLE_BATCH_SIZE}" \
                --overrides "agent.use_amp=${USE_AMP}" \
                --overrides "updates_per_interaction_step=${UPDATES_PER_STEP}" \
                --overrides "gamma=${GAMMA}" \
                --overrides "n_step=${N_STEP}" \
                `#=== Simulation Steps & Checkpointing ===#` \
                --overrides "env.env_name=dexmachina/${hand}" \
                --overrides "num_env_steps=${NUM_ENV_STEPS}" \
                --overrides "num_train_envs=${NUM_TRAIN_ENVS}" \
                --overrides "num_eval_envs=null" \
                --overrides "num_record_envs=null" \
                --overrides "num_eval_episodes=${NUM_TRAIN_ENVS}" \
                --overrides "num_record_episodes=0" \
                --overrides "++env.curriculum=${USE_CURRICULUM}" \
                --overrides "save_checkpoint_per_interaction_step=${CHECKPOINT_INTERVAL}" \
                `#=== Reward Shaping Weights ===#` \
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
                --overrides ++env.reward_cfg.obj_arti_beta=5.0

            echo "[SUCCESS] Training completed for $run_exp_name"

            # 2. EVALUATION & RENDERING
            echo "[START] Evaluating & Rendering checkpoints for: $run_exp_name"

            LATEST_WANDB_DIR=$(ls -td wandb/*run-* 2>/dev/null | head -n 1)
            if [ -n "$LATEST_WANDB_DIR" ]; then
                WANDB_RUN_ID=$(basename "$LATEST_WANDB_DIR" | awk -F'-' '{print $NF}')
                WANDB_ARG="--wandb_run_id ${WANDB_RUN_ID}"
            else
                WANDB_ARG=""
            fi

            LATEST_TS_DIR=$(find "models/FlashSAC/${run_exp_name}" -mindepth 2 -maxdepth 3 -type d -name "seed*" 2>/dev/null | sort -V | tail -n 1)

            if [ -z "$LATEST_TS_DIR" ]; then
                echo "[ERROR] No checkpoint directories found for ${run_exp_name}. Skipping evaluation."
                continue
            fi

            STEP_DIRS=$(find "$LATEST_TS_DIR" -maxdepth 1 -type d -name "step*" 2>/dev/null | sort -V)
            TOTAL_CHECKPOINTS=$(echo "$STEP_DIRS" | wc -w)
            CHECKPOINT_COUNTER=0

            for STEP_DIR in $STEP_DIRS; do
                CHECKPOINT_COUNTER=$((CHECKPOINT_COUNTER + 1))
                STEP_NUM=$(basename "$STEP_DIR" | sed 's/step//')

                python ../envs/FlashSAC/eval_and_render.py \
                    --config_path "${CONFIG_PATH}" \
                    --config_name "${CONFIG_NAME}" \
                    --checkpoint_path "${STEP_DIR}" \
                    ${WANDB_ARG} \
                    --project "${PROJECT_NAME}" \
                    --entity "${ENTITY_NAME}" \
                    --step "${STEP_NUM}" \
                    --num_episodes 1 \
                    "${SHARED_ENV_OVERRIDES[@]}" \
                    "${SHARED_AGENT_OVERRIDES[@]}" \
                    --overrides "env.num_record_envs=1"

                if [ "$CHECKPOINT_COUNTER" -lt "$TOTAL_CHECKPOINTS" ]; then
                    rm -rf "${STEP_DIR}"
                else
                    echo "[PRESERVE] Keeping final milestone checkpoint: ${STEP_DIR}"
                fi
            done

            echo "[SUCCESS] Milestone evaluations completed for $run_exp_name"
            echo "--------------------------------------------------------"

        done
    done
done

echo "[FINISHED] 10M baseline benchmark run is complete!"
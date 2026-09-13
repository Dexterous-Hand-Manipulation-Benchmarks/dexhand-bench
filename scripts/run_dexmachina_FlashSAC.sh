#!/bin/bash
##################################################################################
# Dexmachina (Genesis GPU Simulator + FlashSAC)
##################################################################################

set -e

# --- Auto-activate Conda Environment ---
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

eval "$(conda shell.bash hook 2>/dev/null || true)"
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

# --- 2. Shared Agent Architecture (Table 9 Baseline) ---
AGENT_NAME="flashSAC"
ACTOR_HIDDEN_DIM=256    # Scaled to match SimbaV2 capacity
CRITIC_HIDDEN_DIM=512   # Scaled to match SimbaV2 capacity
ACTOR_NUM_BLOCKS=2
CRITIC_NUM_BLOCKS=2
ASYMMETRIC_OBS="false"

# --- 3. Shared Environment Identity ---
ENV_TYPE="dexmachina"
RETARGET_NAME="para"

# --- 4. Training-Specific Hyperparameters ---
NUM_ENV_STEPS=50_000_896
NUM_TRAIN_ENVS=1024 # AT9
CHECKPOINT_INTERVAL=4882
USE_CURRICULUM="true"
SAMPLE_BATCH_SIZE=4096
USE_AMP="true"
UPDATES_PER_STEP=4
TEMP_TARGET_SIGMA=0.35
GAMMA=0.99
N_STEP=3

# --- 4b. Dexmachina Ablations & Enhancements ---
BOOTSTRAP_ON_TIMEOUT="true"
USE_EFFECTIVE_GAMMA="true"
ACTOR_INIT_STD=0.3
INDEPENDENT_ENV_NOISE="true"
TARGET_ENTROPY_MODE="sac_heuristic"
NORMALIZE_OBSERVATION="true"

# --- 5. Hardware Device & Buffer Allocation Strategy ---
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)

# STRATEGY CHOICE:
# Set to "10M_CPU" for full 10M buffer in System RAM with GPU training (Recommended)
# Set to "GPU_MAX" for pure VRAM buffer (6M on Dual GPU, 5M on Single GPU)
BUFFER_MODE="10M_CPU"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "[INFO] Detected $NUM_GPUS GPUs. Simulation on cuda:0 | Training on cuda:1."
    SIM_DEVICE="cuda:0"
    AGENT_DEVICE="cuda:1"
    GPU_BUFFER_MAX=6_000_000
else
    echo "[INFO] Detected $NUM_GPUS GPU. Simulation and Training on cuda:0."
    SIM_DEVICE="cuda:0"
    AGENT_DEVICE="cuda:0"
    GPU_BUFFER_MAX=5_000_000
fi

if [ "$BUFFER_MODE" = "10M_CPU" ]; then
    echo "[INFO] Buffer Strategy: 10M transitions stored in Host RAM (CPU Buffer) with GPU training."
    BUFFER_DEVICE="cpu"
    BUFFER_MAX_LENGTH=10_000_000
else
    echo "[INFO] Buffer Strategy: Pure VRAM buffer ($GPU_BUFFER_MAX transitions on ${AGENT_DEVICE})."
    BUFFER_DEVICE="${AGENT_DEVICE}"
    BUFFER_MAX_LENGTH=${GPU_BUFFER_MAX}
fi

BUFFER_MIN_LENGTH=100_000

# --- 6. Benchmark Loops ---
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
                --overrides "agent.bootstrap_on_timeout=${BOOTSTRAP_ON_TIMEOUT}"
                --overrides "agent.use_effective_gamma=${USE_EFFECTIVE_GAMMA}"
                --overrides "agent.actor_init_std=${ACTOR_INIT_STD}"
                --overrides "agent.independent_env_noise=${INDEPENDENT_ENV_NOISE}"
                --overrides "agent.target_entropy_mode=${TARGET_ENTROPY_MODE}"
                --overrides "agent.normalize_observation=${NORMALIZE_OBSERVATION}"
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
            if [[ "${run_exp_name}" != "allegro_hand_box-30-230_seed0_10M_official" ]]; then
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
                  --overrides "agent.device_type=${AGENT_DEVICE}" \
                  --overrides "agent.buffer_device_type=${BUFFER_DEVICE}" \
                  --overrides "agent.buffer_max_length=${BUFFER_MAX_LENGTH}" \
                  --overrides "agent.buffer_min_length=${BUFFER_MIN_LENGTH}" \
                  --overrides "agent.sample_batch_size=${SAMPLE_BATCH_SIZE}" \
                  --overrides "agent.use_amp=${USE_AMP}" \
                  --overrides "agent.temp_target_sigma=${TEMP_TARGET_SIGMA}" \
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
                  `#=== Reward Shaping Weights (Identical to FastTD3) ===#` \
                  --overrides ++env.reward_cfg.force_penalty=0.1 \
                  --overrides ++env.reward_cfg.action_penalty=0.01 \
                  --overrides ++env.reward_cfg.task_rew_weight=1.0 \
                  --overrides ++env.reward_cfg.obj_pos_beta=10.0 \
                  --overrides ++env.reward_cfg.obj_rot_beta=1.0 \
                  --overrides ++env.reward_cfg.obj_arti_beta=5.0 \
                  --overrides ++env.reward_cfg.imi_rew_weight=0.3 \
                  --overrides ++env.reward_cfg.imi_wrist_weight=0.5 \
                  --overrides ++env.reward_cfg.contact_rew_weight=3.0 \
                  --overrides ++env.reward_cfg.contact_beta=10.0 \
                  --overrides ++env.reward_cfg.bc_rew_weight=0.3 \
                  --overrides ++env.reward_cfg.bc_beta=500.0 \
                  --overrides ++env.reward_cfg.use_retarget_contact=true

              echo "[SUCCESS] Training completed for $run_exp_name"
            fi
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

echo "[FINISHED] Benchmark execution complete!"
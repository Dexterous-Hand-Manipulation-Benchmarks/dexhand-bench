#!/bin/bash
##################################################################################
# Dexmachina (Genesis GPU Simulator + FlashSAC)
##################################################################################

set -e

DHB_EXTRA=flashsac
source "$(dirname "$0")/../common.sh"
cd "${DHB_ROOT}/algorithms/FlashSAC" || exit 1

echo "[INFO] Running from directory: $(pwd)"

# --- 1. WandB & Hydra Paths ---
PROJECT_NAME="${WANDB_PROJECT}"
GROUP_NAME="FlashSAC"
CONFIG_PATH="configs"
CONFIG_NAME="flashSAC_base"

# --- 2. Shared Agent Architecture (Table 9 Baseline) ---
AGENT_NAME="flashSAC"
ACTOR_HIDDEN_DIM=256
CRITIC_HIDDEN_DIM=512
ACTOR_NUM_BLOCKS=2
CRITIC_NUM_BLOCKS=2
ASYMMETRIC_OBS="false"

# --- 3. Shared Environment Identity ---
ENV_TYPE="dexmachina"
RETARGET_NAME="para"

# --- 4. Training-Specific Hyperparameters ---
NUM_ENV_STEPS=50_000_896
NUM_TRAIN_ENVS=1024
# The eval env (dexmachina make_eval_env) starts every episode at the first frame, without assistance, and the
# policy acts deterministically, so more eval episodes would repeat the same rollout.
NUM_EVAL_EPISODES=1
CHECKPOINT_INTERVAL=4882
USE_CURRICULUM="true"
SAMPLE_BATCH_SIZE=4096
USE_AMP="true"
UPDATES_PER_STEP=4
GAMMA=0.99  # taken from run_isaaclab.sh
N_STEP=3    # taken from run_isaaclab.sh

# --- 4b. Dexmachina Ablations & Enhancements ---
BOOTSTRAP_ON_TIMEOUT="true"
USE_EFFECTIVE_GAMMA="true"    # correct dynamic discounting for early termination (like FastTD3)
ACTOR_INIT_STD=0.3  # prevent deterministic initialization stall
INDEPENDENT_ENV_NOISE="true"
TARGET_ENTROPY_MODE="sac_heuristic"
NORMALIZE_OBSERVATION="true"

# --- 5. Hardware Device & Buffer Allocation Strategy ---
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())") || {
    echo "[ERROR] Could not count the GPUs: importing torch in the ${DHB_EXTRA} uv environment failed (error above)." >&2
    echo "        Fix the environment (uv sync --extra ${DHB_EXTRA}); guessing one GPU would run on the wrong devices." >&2
    exit 1
}

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

# --- 6. Explicit Benchmark Execution Queue ---
# One run per line: seed, hand, clip. A run whose benchmark evaluation file already exists is skipped.
RUN_QUEUE=(
    "1000 xhand box-30-230"
    "2000 allegro_hand box-30-230"
    "2000 allegro_hand mixer-0-150"
    "2000 dex3_hand box-30-230"
    "2000 inspire_hand box-30-230"
    "2000 inspire_hand mixer-0-150"
    "2000 xhand box-30-230"
)

# train.py saves the final checkpoint as step<number of interaction steps> (NUM_ENV_STEPS without its underscores).
FINAL_INTERACTION_STEP=$(( ${NUM_ENV_STEPS//_/} / NUM_TRAIN_ENVS ))

# Execution of runs
for run_item in "${RUN_QUEUE[@]}"; do
    read -r seed hand clip <<< "$run_item"

    run_exp_name="${hand}_${clip}_seed${seed}_10M_official"
    env_name="${ENV_TYPE}/${hand}"
    exp_dir="models/${GROUP_NAME}/${run_exp_name}"

    obj_name=$(echo "$clip" | cut -d'-' -f1)
    data_file="${DEXMACHINA_ASSETS}/contact_retarget/${hand}/s01/${obj_name}_use_01.npy"

    if [ ! -f "$data_file" ]; then
        echo "[SKIP] Retarget data not found: ${hand} + ${obj_name}. Moving to next..."
        continue
    fi

    # The benchmark evaluation (eval_dexmachina.py, TRAJECTORY_FILENAME) saves this file next to the evaluated checkpoint.
    COMPLETED_EVALUATION=$(find "$exp_dir" -mindepth 3 -maxdepth 5 -type f -name "benchmark_trajectory_*.npz" 2>/dev/null | head -n 1)
    if [ -n "$COMPLETED_EVALUATION" ]; then
        echo "[SKIP] Experiment ${run_exp_name} already completed (${COMPLETED_EVALUATION} found). Moving to next..."
        continue
    fi

    # A training that finished (train.py wrote training_finished) but has no evaluation (it failed earlier) is evaluated,
    # not retrained.
    FINISHED_STEP_DIRS=""
    for step_dir in $(find "$exp_dir" -maxdepth 5 -type d -name "step${FINAL_INTERACTION_STEP}" 2>/dev/null | sort); do
        if [ -f "$(dirname "$step_dir")/training_finished" ]; then
            FINISHED_STEP_DIRS+="${step_dir}"$'\n'
        fi
    done
    FINISHED_STEP_DIRS="${FINISHED_STEP_DIRS%$'\n'}"
    if [ -n "$FINISHED_STEP_DIRS" ] && [ "$(echo "$FINISHED_STEP_DIRS" | wc -l)" -gt 1 ]; then
        echo "[ERROR] ${exp_dir} has several finished trainings without an evaluation:" >&2
        echo "$FINISHED_STEP_DIRS" >&2
        echo "        Delete the failed ones, so that the evaluation does not have to guess which run to score." >&2
        exit 1
    fi

    if [ -n "$FINISHED_STEP_DIRS" ]; then
        RUN_DIR=$(dirname "$FINISHED_STEP_DIRS")
        echo "[INFO] Training of ${run_exp_name} is finished (${RUN_DIR}); only its evaluation is missing."
    else
        RUN_DIR="${exp_dir}/${env_name}/seed${seed}-$(date +%Y%m%d-%H%M%S)"

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

        python train.py \
            --config_path "${CONFIG_PATH}" \
            --config_name "${CONFIG_NAME}" \
            --overrides "seed=${seed}" \
            `#=== Logging (WandB) ===#` \
            --overrides "logger_type=wandb" \
            --overrides "project_name=${PROJECT_NAME}" \
            --overrides "group_name=${GROUP_NAME}" \
            --overrides "exp_name=${run_exp_name}" \
            --overrides "save_path=${RUN_DIR}" \
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
            --overrides "updates_per_interaction_step=${UPDATES_PER_STEP}" \
            --overrides "gamma=${GAMMA}" \
            --overrides "n_step=${N_STEP}" \
            `#=== Simulation Steps & Checkpointing ===#` \
            --overrides "env.env_name=${env_name}" \
            --overrides "num_env_steps=${NUM_ENV_STEPS}" \
            --overrides "num_train_envs=${NUM_TRAIN_ENVS}" \
            --overrides "num_eval_envs=null" \
            --overrides "num_record_envs=null" \
            --overrides "num_eval_episodes=${NUM_EVAL_EPISODES}" \
            --overrides "num_record_episodes=0" \
            --overrides "++env.curriculum=${USE_CURRICULUM}" \
            --overrides "save_checkpoint_per_interaction_step=${CHECKPOINT_INTERVAL}"

        echo "[SUCCESS] Training completed for $run_exp_name"
    fi

    # The benchmark evaluation of the final checkpoint (the protocol shared by all methods).
    "${DHB_ROOT}/scripts/flashsac/evaluate_run.sh" "$RUN_DIR" "$PROJECT_NAME" "$SIM_DEVICE" "$FINAL_INTERACTION_STEP"
    echo "[SUCCESS] Evaluation completed for $run_exp_name"
    echo "--------------------------------------------------------"
done

echo "[FINISHED] Benchmark execution complete!"

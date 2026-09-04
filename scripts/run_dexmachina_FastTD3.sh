#!/bin/bash
##################################################################################
# Dexmachina (Genesis GPU Simulator + FastTD3)
##################################################################################

set -e

# --- Auto-activate Conda Environment ---
eval "$(conda shell.bash hook)"
if [ "$CONDA_DEFAULT_ENV" != "dexmachina" ]; then
    echo "[INFO] Activating Conda environment 'dexmachina'..."
    conda activate dexmachina
fi

cd "$(dirname "$0")/../FastTD3" || exit 1

echo "[INFO] Using Python from: $(which python)"
echo "[INFO] Running from directory: $(pwd)"

# --- Algorithm Selection ---
# Choose: "fasttd3_simbav2" (FastTD3 + SimbaV2, recommended) or "fasttd3" (plain FastTD3)
AGENT="fasttd3_simbav2"

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
#clips=("box-30-230")

#seeds=( 0 1000 2000 )
seeds=( 0 )

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
            NUM_TRAIN_ENVS=640
            TOTAL_TIMESTEPS=$((50000000 / NUM_TRAIN_ENVS)) #65104
            MAX_EPOCHS=10000
            SAVE_INTERVAL=$((TOTAL_TIMESTEPS / 5))
            RENDER_INTERVAL=$((TOTAL_TIMESTEPS / 10))
            EVAL_INTERVAL=$((TOTAL_TIMESTEPS / 10))
            BUFFER_SIZE=1024

            # Configure agent-specific parameters
            AGENT_ARGS=( "--agent" "${AGENT}" )
            if [ "$AGENT" = "fasttd3_simbav2" ]; then
                GROUP_NAME="FastTD3_SimbaV2"
                AGENT_ARGS+=(
                    "--actor_hidden_dim" "256"
                    "--critic_hidden_dim" "512"
                    "--actor_num_blocks" "1"
                    "--critic_num_blocks" "2"
                    "--actor_learning_rate_end" "3e-5"
                    "--critic_learning_rate_end" "3e-5"
                    "--weight_decay" "0.0"
                )
            else
                GROUP_NAME="FastTD3"
                AGENT_ARGS+=(
                    "--actor_hidden_dim" "512"
                    "--critic_hidden_dim" "1024"
                )
            fi

            echo "========================================================"
            echo "[START] Training ${GROUP_NAME}: $run_exp_name (Agent: ${AGENT})"
            echo "========================================================"

            # 1. RUN TRAINING
            python fast_td3/train.py \
                --env_name "dexmachina|${hand}|${clip}" \
                --exp_name "${run_exp_name}" \
                --project "dexmachina" \
                --entity "schrapsi-kit" \
                --group_name "${GROUP_NAME}" \
                "${AGENT_ARGS[@]}" \
                --seed "${seed}" \
                --num_envs ${NUM_TRAIN_ENVS} \
                --learning_starts 50 \
                --batch_size 16384 \
                --buffer_size ${BUFFER_SIZE} \
                --num_updates 1 \
                --num_steps 3 \
                --no_use_cdq \
                --std_min 0.05 \
                --std_max 0.6 \
                --policy_noise 0.1 \
                --gamma 0.99 \
                --v_min 0.0 \
                --v_max 250.0 \
                --disable_bootstrap \
                --save_interval ${SAVE_INTERVAL} \
                --render_interval ${RENDER_INTERVAL} \
                --eval_interval ${EVAL_INTERVAL} \
                --total_timesteps ${TOTAL_TIMESTEPS} \
                --max_epochs ${MAX_EPOCHS}

            echo "[SUCCESS] Training completed for $run_exp_name"

            # 2. RUN EVALUATION & RENDERING FOR ALL SAVED CHECKPOINTS
            echo "[START] Evaluating & Rendering all step milestones for: $run_exp_name"

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

            # Target ONLY the latest timestamp directory from this specific run
            RUN_PATH="models/FastTD3/${run_exp_name}"
            LATEST_TS_DIR=$(ls -td "${RUN_PATH}"/*/ 2>/dev/null | head -n 1)

            if [ -z "$LATEST_TS_DIR" ]; then
                echo "[WARNING] No checkpoint directory found in ${RUN_PATH}. Skipping evaluation."
                continue
            fi

            echo "[INFO] Evaluating checkpoints from: ${LATEST_TS_DIR}"

            # 1. Evaluate intermediate step checkpoints sorted in chronological order
            STEP_CKPTS=$(find "$LATEST_TS_DIR" -maxdepth 1 -type f -name "step*.pt" 2>/dev/null | sort -V)

            for CKPT in $STEP_CKPTS; do
                STEP_NUM=$(basename "$CKPT" .pt | sed 's/step//')
                ENV_STEP=$(( STEP_NUM * NUM_TRAIN_ENVS ))
                echo "[INFO] Evaluating intermediate checkpoint: ${CKPT} (Env Step: ${ENV_STEP})"

                python ../envs/FastTD3/eval_fast_td3.py \
                    --checkpoint "$CKPT" \
                    --num_envs 20 \
                    --env_name "dexmachina|${hand}|${clip}" \
                    --record_video \
                    ${WANDB_ARG} \
                    --step "${ENV_STEP}"
            done

            # 2. Evaluate best checkpoint saved during training
            if [ -f "${LATEST_TS_DIR}/best.pt" ]; then
                BEST_CKPT="${LATEST_TS_DIR}/best.pt"
                echo "[INFO] Evaluating best checkpoint: ${BEST_CKPT} (Reading true saved step from metadata)"

                python ../envs/FastTD3/eval_fast_td3.py \
                    --checkpoint "$BEST_CKPT" \
                    --num_envs 20 \
                    --env_name "dexmachina|${hand}|${clip}" \
                    --record_video \
                    ${WANDB_ARG}
            elif [ -f "${LATEST_TS_DIR}/final.pt" ]; then
                FINAL_CKPT="${LATEST_TS_DIR}/final.pt"
                TOTAL_ENV_STEPS=$(( TOTAL_TIMESTEPS * NUM_TRAIN_ENVS ))
                echo "[INFO] Evaluating fallback final checkpoint: ${FINAL_CKPT} (Env Step: ${TOTAL_ENV_STEPS})"

                python ../envs/FastTD3/eval_fast_td3.py \
                    --checkpoint "$FINAL_CKPT" \
                    --num_envs 20 \
                    --env_name "dexmachina|${hand}|${clip}" \
                    --record_video \
                    ${WANDB_ARG} \
                    --step "${TOTAL_ENV_STEPS}"
            fi

            echo "[SUCCESS] All milestone evaluations completed for $run_exp_name"
            echo "--------------------------------------------------------"

        done
    done
done

echo "[FINISHED] All automated FastTD3 training and evaluations are complete!"
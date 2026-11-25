#!/bin/bash
##################################################################################
# Dexmachina (Genesis GPU Simulator + FastTD3)
##################################################################################

set -e
DHB_EXTRA=fasttd3
source "$(dirname "$0")/../common.sh"
cd "${DHB_ROOT}/algorithms/FastTD3" || exit 1

echo "[INFO] Running from directory: $(pwd)"

# Scores one checkpoint on the shared benchmark protocol and logs it into the run's W&B run. A failure stops the
# script: a reported number must not go missing silently.
evaluate_checkpoint() {
    echo "[INFO] Evaluating checkpoint: $1"
    python fast_td3/eval_dexmachina.py --checkpoint "$1" --wandb || {
        echo "[ERROR] Evaluating $1 failed. To evaluate only it again, from algorithms/FastTD3 (first delete the" >&2
        echo "        benchmark_trajectory_* file next to it if this attempt wrote one):" >&2
        echo "        python fast_td3/eval_dexmachina.py --checkpoint $1 --wandb" >&2
        exit 1
    }
}

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

seeds=( 0 1000 2000 )
#seeds=( 0 )

for seed in "${seeds[@]}"; do
    for hand in "${hands[@]}"; do
        for clip in "${clips[@]}"; do

            # Extract the object name to safely check if retarget data exists
            obj_name=$(echo "$clip" | cut -d'-' -f1)
            data_file="${DEXMACHINA_ASSETS}/contact_retarget/${hand}/s01/${obj_name}_use_01.npy"

            if [ ! -f "$data_file" ]; then
                echo "[SKIP] Retarget data not found: ${hand} + ${obj_name}. Moving to next..."
                continue
            fi

            run_exp_name="${hand}_${clip}_seed${seed}"
            NUM_TRAIN_ENVS=2048
            TOTAL_TIMESTEPS=$((50000000 / NUM_TRAIN_ENVS)) # 24414 iterations
            SAVE_INTERVAL=$((TOTAL_TIMESTEPS / 2))
            EVAL_INTERVAL=$((TOTAL_TIMESTEPS / 4))
            BUFFER_SIZE=8192

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
                --project "${WANDB_PROJECT}" \
                --group_name "${GROUP_NAME}" \
                "${AGENT_ARGS[@]}" \
                --seed "${seed}" \
                --num_envs ${NUM_TRAIN_ENVS} \
                --learning_starts 50 \
                --batch_size 32768 \
                --buffer_size ${BUFFER_SIZE} \
                --num_updates 8 \
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
                --eval_interval ${EVAL_INTERVAL} \
                --total_timesteps ${TOTAL_TIMESTEPS}

            echo "[SUCCESS] Training completed for $run_exp_name"

            # 2. EVALUATE EVERY CHECKPOINT OF THIS RUN ON THE SHARED BENCHMARK PROTOCOL
            echo "[START] Evaluating the checkpoints of: $run_exp_name"

            # Target ONLY the latest timestamp directory from this specific run
            RUN_PATH="models/FastTD3/${run_exp_name}"
            LATEST_TS_DIR=$(ls -td "${RUN_PATH}"/*/ 2>/dev/null | head -n 1)

            if [ -z "$LATEST_TS_DIR" ]; then
                echo "[ERROR] No checkpoint directory found in ${RUN_PATH}: training of ${run_exp_name} wrote none." >&2
                exit 1
            fi
            FINAL_CKPT="${LATEST_TS_DIR}/final.pt"
            if [ ! -f "$FINAL_CKPT" ]; then
                echo "[ERROR] ${FINAL_CKPT} is missing: training of ${run_exp_name} did not finish." >&2
                exit 1
            fi

            echo "[INFO] Evaluating checkpoints from: ${LATEST_TS_DIR}"

            # 1. Intermediate step checkpoints, in training order
            STEP_CKPTS=$(find "$LATEST_TS_DIR" -maxdepth 1 -type f -name "step*.pt" 2>/dev/null | sort -V)
            for CKPT in $STEP_CKPTS; do
                evaluate_checkpoint "$CKPT"
            done

            # 2. The final checkpoint: the reported score.
            evaluate_checkpoint "$FINAL_CKPT"

            echo "[SUCCESS] All milestone evaluations completed for $run_exp_name"
            echo "--------------------------------------------------------"

        done
    done
done

echo "[FINISHED] All automated FastTD3 training and evaluations are complete!"

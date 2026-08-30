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
            echo "[START] Training FastTD3: $run_exp_name"
            echo "========================================================"

            # 1. RUN TRAINING
            python fast_td3/train.py \
                --env_name "dexmachina|${hand}|${clip}" \
                --exp_name "${run_exp_name}" \
                --project "dexmachina" \
                --entity "schrapsi-kit" \
                --group_name "FastTD3" \
                --seed "${seed}" \
                --num_envs 768 \
                --learning_starts 50 \
                --batch_size 16384 \
                --buffer_size 1024 \
                --num_updates 1 \
                --std_min 0.05 \
                --std_max 0.6 \
                --policy_noise 0.1 \
                --gamma 0.99 \
                --v_min 0.0 \
                --v_max 250.0 \
                --disable_bootstrap \
                --save_interval 15000 \
                --render_interval 5000 \
                --eval_interval 5000 \
                --total_timesteps 150000 \
                --max_epochs 10000

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

            # Find all saved checkpoints for this run
            CKPTS=$(ls -t models/*/*${run_exp_name}*.pt 2>/dev/null | sort -V)

            if [ -z "$CKPTS" ]; then
                echo "[WARNING] No checkpoints found for ${run_exp_name}. Skipping milestone evaluation."
                continue
            fi

            for CKPT in $CKPTS; do
                STEP_NUM=$(basename "$CKPT" .pt | grep -o '[0-9]\+$' || echo "")
                echo "[INFO] Evaluating checkpoint: ${CKPT} (Step: ${STEP_NUM:-final})"

                STEP_ARG=""
                if [ -n "$STEP_NUM" ]; then
                    ENV_STEP=$(( STEP_NUM * 768 ))
                    STEP_ARG="--step ${ENV_STEP}"
                fi

                python ../envs/FastTD3/eval_fast_td3.py \
                    --checkpoint "$CKPT" \
                    --num_envs 20 \
                    --env_name "dexmachina|${hand}|${clip}" \
                    --record_video \
                    ${WANDB_ARG} \
                    ${STEP_ARG}
            done

            echo "[SUCCESS] All milestone evaluations completed for $run_exp_name"
            echo "--------------------------------------------------------"

        done
    done
done

echo "[FINISHED] All automated FastTD3 training and evaluations are complete!"
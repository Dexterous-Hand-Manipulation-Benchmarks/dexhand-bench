#!/bin/bash

set -e

# --- Auto-activate Conda Environment ---
eval "$(conda shell.bash hook)"
if [ "$CONDA_DEFAULT_ENV" != "dexmachina" ]; then
    echo "[INFO] Activating Conda environment 'dexmachina'..."
    conda activate dexmachina
fi

cd "$(dirname "$0")/../FastTD3" || exit 1

hands=("allegro_hand")
clips=("box-30-230")
seeds=(0)

for seed in "${seeds[@]}"; do
    for hand in "${hands[@]}"; do
        for clip in "${clips[@]}"; do

            run_exp_name="FastTD3_${hand}_${clip}_seed${seed}"
            echo "Running: $run_exp_name"

            python fast_td3/train.py \
                --env_name "dexmachina|${hand}|${clip}" \
                --exp_name ${run_exp_name} \
                --project "dexmachina" \
                --seed ${seed} \
                --learning_starts 50 \
                --num_envs 768 \
                --batch_size 16384 \
                --buffer_size 1024 \
                --num_updates 1 \
                --std_min 0.05 \
                --std_max 0.6 \
                --policy_noise 0.1 \
                --gamma 0.99 \
                --v_min 0.0 \
                --v_max 250.0 \
                --disable_bootstrap  \
                --render_interval 5000 \
                --eval_interval 5000 \
                --max_epochs 10000 \
                --total_timesteps 150000
                
            # Find the most recently created final checkpoint for this specific run
            LATEST_MODEL=$(ls -t models/*/*${run_exp_name}*final.pt 2>/dev/null | head -n 1)
            
            if [ -n "$LATEST_MODEL" ]; then
                echo "[INFO] Found final checkpoint: $LATEST_MODEL. Running evaluation..."
                
                # 1. Run the FastTD3 custom evaluation script and record a video
                python ../envs/FastTD3/eval_fast_td3.py \
                    --checkpoint "$LATEST_MODEL" \
                    --num_envs 20 \
                    --env_name "dexmachina|${hand}|${clip}" \
                    --record_video
                
                # The eval data is saved in a folder named after the checkpoint
                CKPT_NAME=$(basename "$LATEST_MODEL" .pt)
                CKPT_DIR=$(dirname "$LATEST_MODEL")
                EVAL_FILE="${CKPT_DIR}/${CKPT_NAME}_eval/eval_ep0.npy"
                
                # Extract object name from clip (e.g., box-30-230 -> box)
                OBJ_NAME=$(echo "$clip" | cut -d'-' -f1)
                
                # 2. Run the DexMachina compute_add.py scoring script
                echo "[INFO] Running standardized ADD-AUC metric evaluation..."
                # Temporarily cd to root to avoid the namespace package shadowing bug!
                cd ..
                python -m dexmachina.eval.compute_add --input "FastTD3/${EVAL_FILE}" --obj_name "$OBJ_NAME"
                cd FastTD3
                
                echo "[INFO] Evaluation complete! Check FastTD3/${CKPT_DIR}/${CKPT_NAME}_eval/add_stats.json for your final scores!"
            else
                echo "[WARNING] Could not find a final checkpoint matching models/*${run_exp_name}*150000.pt!"
            fi
        done
    done
done
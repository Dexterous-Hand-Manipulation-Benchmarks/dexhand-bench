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
seeds=(0 1000)

for seed in "${seeds[@]}"; do
    for hand in "${hands[@]}"; do
        for clip in "${clips[@]}"; do

            run_exp_name="FastTD3_${hand}_${clip}_seed${seed}"
            echo "Running: $run_exp_name"

            python fast_td3/train.py \
                --env_name "dexmachina|${hand}|${clip}" \
                --exp_name ${run_exp_name} \
                --num_envs 1024 \
                --batch_size 32768 \
                --buffer_size 1024 \
                --seed ${seed} \
                --render_interval 0 \
                --project "dexmachina" \
                --render_interval 5000
        done
    done
done
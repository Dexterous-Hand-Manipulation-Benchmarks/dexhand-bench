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
                --num_envs 896 \
                --batch_size 16384 \
                --buffer_size 1024 \
                --num_updates 2 \
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
        done
    done
done
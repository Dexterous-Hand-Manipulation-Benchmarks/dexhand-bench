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
# This ensures train.py and Hydra configs are found correctly
cd "$(dirname "$0")/../FlashSAC" || exit 1

# Confirm Python path
echo "[INFO] Using Python from: $(which python)"
echo "[INFO] Running from directory: $(pwd)"


# --- Define hands and clips to benchmark ---
hands=(
    "allegro_hand"
)

clips=(
    "box-30-230"
)

seeds=( 0 1000 2000 )

for seed in "${seeds[@]}"; do
    for hand in "${hands[@]}"; do
        for clip in "${clips[@]}"; do

            run_exp_name="${hand}_${clip}_seed${seed}"
            echo "Running: $run_exp_name"

            python train.py \
                --config_name flashSAC_base \
                --overrides seed=${seed} \
                `#=== Logging (WandB) ===#` \
                --overrides logger_type='wandb' \
                --overrides project_name='dexmachina' \
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
                `#=== Agent (GPU sim defaults) ===#` \
                --overrides agent=flashSAC \
                --overrides agent.buffer_max_length=2_000_000 \
                --overrides agent.buffer_min_length=100_000 \
                --overrides agent.buffer_device_type='cuda' \
                --overrides agent.sample_batch_size=2048 \
                --overrides agent.use_amp=true \
                --overrides updates_per_interaction_step=2 \
                `#=== Benchmark default ===#` \
                --overrides agent.asymmetric_observation=false \
                --overrides gamma=0.95 \
                --overrides n_step=1
        done
    done
done
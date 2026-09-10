#!/usr/bin/env bash

set -e

# --- Auto-activate Conda Environment ---
eval "$(conda shell.bash hook)"
if [ "$CONDA_DEFAULT_ENV" != "dexmachina" ]; then
    echo "[INFO] Activating Conda environment 'dexmachina'..."
    conda activate dexmachina
fi

cd "$(dirname "$0")/../reppo" || exit 1

echo "[INFO] Using Python from: $(which python)"
echo "[INFO] Running from directory: $(pwd)"

WANDB_PROJECT="${WANDB_PROJECT:-dexmachina}"
WANDB_ENTITY="${WANDB_ENTITY:-schrapsi-kit}"

echo "[INFO] WANDB CONFIG"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "WANDB_ENTITY: ${WANDB_ENTITY}"

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
            echo "[START] Training: $run_exp_name"
            echo "========================================================"

            # 1. RUN HEADLESS TRAINING AT MAX SPEED
            python src/jaxrl/reppo.py \
                env=dexmachina \
                seed=${seed} \
                env.hand=${hand} \
                env.clip=${clip} \
                ++exp_name="${run_exp_name}" \
                ++env.name="dexmachina" \
                ++wandb.project="${WANDB_PROJECT}" \
                ++wandb.entity="${WANDB_ENTITY}" \
                ++env.retarget_name="para" \
                ++hyperparameters.total_time_steps=262_144 \
                ++hyperparameters.num_envs=512 \
                ++hyperparameters.num_eval=2 \
                ++num_trials=1 \
                ++env.reward_cfg.force_penalty=0.1 \
                ++env.reward_cfg.action_penalty=0.01 \
                ++env.reward_cfg.imi_rew_weight=0.3 \
                ++env.reward_cfg.imi_wrist_weight=0.5 \
                ++env.reward_cfg.contact_rew_weight=3.0 \
                ++env.reward_cfg.contact_beta=10.0 \
                ++env.reward_cfg.bc_rew_weight=0.3 \
                ++env.reward_cfg.bc_beta=500.0 \
                ++env.reward_cfg.use_retarget_contact=true \
                ++env.reward_cfg.obj_pos_beta=10.0 \
                ++env.reward_cfg.obj_rot_beta=1.0 \
                ++env.reward_cfg.obj_arti_beta=5.0 \
                ++env.curriculum=true \
                ++use_rl_games=true


#           hyperparameters.total_time_steps=50_000_896 \

            echo "[SUCCESS] Training completed for $run_exp_name"

            # 2. RUN EVALUATION & RENDERING FOR ALL SAVED CHECKPOINTS
            echo "[START] Evaluating & Rendering all step milestones for: $run_exp_name"

            # Hydra automatically outputs to outputs/YYYY-MM-DD/HH-MM-SS.
            # We grab the most recently created directory.
            LATEST_HYDRA_DIR=$(ls -td outputs/*/* 2>/dev/null | head -n 1)
            CKPT_DIR="${LATEST_HYDRA_DIR}/checkpoints"

            if [ ! -d "$CKPT_DIR" ]; then
                echo "[ERROR] No checkpoint directory found at ${CKPT_DIR}. Skipping evaluation."
                continue
            fi

            echo "[INFO] Handing off to debug_render.py targeting ${CKPT_DIR}"
            
            # debug_render.py natively handles the config loading, WandB resumption, and the checkpoint loop!
            python debug_render.py --ckpt_dir "${CKPT_DIR}" --wandb_project "${WANDB_PROJECT}"

            echo "[SUCCESS] All milestone evaluations completed for $run_exp_name"
            echo "--------------------------------------------------------"

        done
    done
done

echo "[FINISHED] All automated training and evaluations are complete!"
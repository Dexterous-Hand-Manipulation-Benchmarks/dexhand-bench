#!/bin/bash
#SBATCH --job-name=dexmachina_ppo
#SBATCH --output=logs/slurm_ppo_%j.out
#SBATCH --error=logs/slurm_ppo_%j.err
#SBATCH --gres=gpu:1                        # Genesis requires a GPU
#SBATCH --ntasks=1
# Partition, account, time and CPUs come from configs/ via scripts/slurm/submit.sh.

# Queues the original DexMachina PPO (rl_games) implementation.
# Usage:
#   scripts/slurm/submit.sh ppo
#   HAND=dex3_hand CLIP=mixer-0-150 SEED=1000 scripts/slurm/submit.sh ppo

set -e

# --- Run configuration (override via environment variables) ---
HAND="${HAND:-allegro_hand}"
CLIP="${CLIP:-box-30-230}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_EPOCHS="${MAX_EPOCHS:-5000}"
EXP_NAME="${EXP_NAME:-ppo_${HAND}_${CLIP}_seed${SEED}}"

# 1. Environment: the ppo uv extra (rl_games) and W&B settings from configs/
cd "${SLURM_SUBMIT_DIR}"   # submit.sh submits from the repository root
mkdir -p logs
DHB_EXTRA=ppo
source scripts/common.sh

# 2. Move to the dexmachina package
cd "${DHB_ROOT}/environments/dexmachina"

echo "[INFO] Job started on $(hostname) at $(date)"
echo "[INFO] hand=${HAND} clip=${CLIP} seed=${SEED} num_envs=${NUM_ENVS}"

# 4. Train
python dexmachina/rl/train_rl_games.py -B ${NUM_ENVS} -obf -obt --max_epochs ${MAX_EPOCHS} \
    --hand ${HAND} --clip ${CLIP} --seed ${SEED} -exp ${EXP_NAME} --wandb_project ${WANDB_PROJECT} \
    --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all --curr_schedule uniform \
    --wait_epochs 100 --learning_rate 0.0003 --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
    --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow --action_penalty 0.01 \
    --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact \
    --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 -am hybrid --hybrid_scales 0.1 1.0 \
    --kp_init 80 --kv_init 5 -imi 0.3 -bc 0.3 -con 3 -ert 0.6

echo "[INFO] Job finished at $(date)"

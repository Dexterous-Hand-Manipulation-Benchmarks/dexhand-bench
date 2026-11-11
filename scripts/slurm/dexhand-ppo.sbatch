#!/bin/bash
#SBATCH --job-name=dexhand_ppo
#SBATCH --output=logs/slurm_dexhand_ppo_%j.out
#SBATCH --error=logs/slurm_dexhand_ppo_%j.err
#SBATCH --gres=gpu:1                        # Genesis requires a GPU
#SBATCH --ntasks=1
# Partition, account, time and CPUs come from configs/ via scripts/slurm/submit.sh.

# Queues one dexhand-ppo run (algorithms/dexhand-ppo).
# Usage:
#   scripts/slurm/submit.sh dexhand-ppo
#   PRESET=reppo_budget HAND=dex3_hand SEED=1000 scripts/slurm/submit.sh dexhand-ppo

set -e
cd "${SLURM_SUBMIT_DIR}"   # submit.sh submits from the repository root
mkdir -p logs

echo "[INFO] Job started on $(hostname) at $(date)"
./scripts/dexhand_ppo/train.sh
echo "[INFO] Job finished at $(date)"

#!/bin/bash
#SBATCH --job-name=dexmachina_reppo
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --gres=gpu:1                        # Genesis/JAX require a GPU
#SBATCH --ntasks=1
# Partition, account, time and CPUs come from configs/ via scripts/slurm/submit.sh:
#   scripts/slurm/submit.sh reppo

set -e
cd "${SLURM_SUBMIT_DIR}"   # submit.sh submits from the repository root
mkdir -p logs

echo "[INFO] Job started on $(hostname) at $(date)"
./scripts/reppo/run_and_eval.sh
echo "[INFO] Job finished at $(date)"

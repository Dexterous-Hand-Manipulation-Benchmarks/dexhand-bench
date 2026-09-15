#!/bin/bash
#SBATCH --job-name=dexmachina_reppo
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=16:00:00                     # Time limit (30 mins max for dev queues)
#SBATCH --partition=gpu_a100_il       # Fast development GPU partition (change for production)
#SBATCH --gres=gpu:1                        # Request 1 GPU (Genesis/JAX require GPU)
#SBATCH --cpus-per-task=16                  # Recommended CPU count per GPU task
#SBATCH --ntasks=1

# 1. Initialize environment (adjust path to your conda installation if needed)
source ~/.bashrc
conda activate dexmachina

# 2. Move to project directory
cd ~/github/dexmachina_sac

# 3. Create logs directory if it doesn't exist
mkdir -p logs

echo "[INFO] Job started on $(hostname) at $(date)"

# 4. Execute your run script
./scripts/run_and_eval_reppo.sh

echo "[INFO] Job finished at $(date)"

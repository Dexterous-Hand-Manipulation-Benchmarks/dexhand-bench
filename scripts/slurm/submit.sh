#!/bin/bash
# Submit a SLURM job with partition/account/time/... from configs/.
# Usage: scripts/slurm/submit.sh <job> [extra sbatch flags]    (job: reppo, ppo)
# Run variables pass through, e.g.: HAND=xhand SEED=1 scripts/slurm/submit.sh ppo
set -e
JOB="${1:?usage: scripts/slurm/submit.sh <job> [sbatch flags]}"; shift
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="${ROOT}/scripts/slurm/${JOB}.sbatch"
[ -f "${SCRIPT}" ] || { echo "[ERROR] no job script ${SCRIPT}"; exit 1; }

# Read the slurm section with the job's own uv extra, so the run environment is not re-synced.
ARGS=$(uv run --project "${ROOT}" --exact --extra "${JOB}" dexhand-bench sbatch-args "${JOB}")
cd "${ROOT}"
echo "[INFO] sbatch ${ARGS} $* ${SCRIPT}"
eval sbatch ${ARGS} '"$@"' '"${SCRIPT}"'

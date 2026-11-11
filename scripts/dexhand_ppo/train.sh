#!/bin/bash
# One dexhand-ppo training run on DexMachina (benchmark protocol of the lab report).
# Run variables (environment): HAND, CLIP, SEED, PRESET (default | baseline | reppo_budget), WANDB (1/0).
# Further Hydra overrides are passed through, e.g.:
#   PRESET=reppo_budget SEED=1000 scripts/dexhand_ppo/train.sh ppo.kl_target=0.01
# Smoke test on CPU without W&B:
#   WANDB=0 scripts/dexhand_ppo/train.sh device=cpu env.num_envs=2 ppo.num_minibatches=2 total_steps=128 \
#       env.dexmachina.num_eval_envs=1
set -e
DHB_EXTRA=dexhand-ppo
source "$(dirname "$0")/../common.sh"

HAND="${HAND:-allegro_hand}"
CLIP="${CLIP:-box-30-230}"
SEED="${SEED:-0}"
PRESET="${PRESET:-default}"
WANDB="${WANDB:-1}"
EXP_NAME="dexhand_ppo_${PRESET}_${HAND}_${CLIP}_seed${SEED}"
RUN_DIR="${DHB_ROOT}/logs/dexhand_ppo/${EXP_NAME}/$(date +%Y-%m-%d_%H-%M-%S)"

WANDB_ARGS=("log.wandb=false")
if [ "${WANDB}" = "1" ]; then
    WANDB_ARGS=("log.wandb=true" "log.project=${WANDB_PROJECT}" "log.group=dexhand_ppo_${PRESET}")
    if [ -n "${WANDB_ENTITY}" ]; then WANDB_ARGS+=("log.entity=${WANDB_ENTITY}"); fi
fi

echo "[INFO] ${EXP_NAME} -> ${RUN_DIR}"
python -m dexhand_ppo.train "preset=${PRESET}" "seed=${SEED}" "env.dexmachina.hand=${HAND}" \
    "env.dexmachina.clip=${CLIP}" "exp_name=${EXP_NAME}" "hydra.run.dir=${RUN_DIR}" "${WANDB_ARGS[@]}" "$@"

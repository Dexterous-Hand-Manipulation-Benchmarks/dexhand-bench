# Shared setup for the run scripts. Set DHB_EXTRA to the method's uv extra
# (flashsac, fasttd3, reppo or ppo), then source this file.
: "${DHB_EXTRA:?set DHB_EXTRA before sourcing common.sh}"

DHB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEXMACHINA_ASSETS="${DHB_ROOT}/environments/dexmachina/dexmachina/assets"

# Every `python` call runs in the method's locked uv environment, from any directory.
python() {
    uv run --project "${DHB_ROOT}" --exact --extra "${DHB_EXTRA}" python "$@"
}
python3() { python "$@"; }
export -f python python3
export DHB_ROOT DHB_EXTRA

# W&B entity/project from configs/default.yaml + configs/local.yaml.
# WANDB_ENTITY / WANDB_PROJECT already set in the environment take precedence.
eval "$(uv run --project "${DHB_ROOT}" --exact --extra "${DHB_EXTRA}" dexhand-bench env)"
echo "[INFO] uv environment: ${DHB_EXTRA} | W&B: ${WANDB_ENTITY:-<default entity>}/${WANDB_PROJECT}"

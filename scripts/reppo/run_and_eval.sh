#!/usr/bin/env bash

set -e

DHB_EXTRA=reppo
source "$(dirname "$0")/../common.sh"
cd "${DHB_ROOT}/algorithms/reppo" || exit 1

echo "[INFO] Running from directory: $(pwd)"

# =====================================================================
# TEST RUN -- 30M steps.
#
# Curriculum epochs are now a NORMALISED progress counter (see
# env.curriculum_max_epochs in reppo/config/env/dexmachina.yaml), so they no
# longer scale with total_time_steps. wait_epochs / num_zero_epoch / zero_epoch
# mean the same thing at 10M as at 50M and must NOT be overridden per-budget
# any more. Only total_time_steps and num_eval change between test and full run.
#
#   curriculum epochs = 0 .. 10000 regardless of budget
#   zero_epoch        = 10000 - 1000 = 9000  (last 10% unassisted)
#   first decay       >= wait_epochs(100), i.e. 1% into the run
#
# For the real run: total_time_steps=50_000_000 and restore the
# full hand/clip/seed sweep below.
# =====================================================================

# How many checkpoints to render. ~40s each (200 frames @ 500x500 + encode +
# upload), so this dominates the job tail. 0 = all.
MAX_VIDEOS="${MAX_VIDEOS:-6}"

# debug_render.py has its own hardcoded default entity, so always pass one explicitly.
if [ -z "${WANDB_ENTITY}" ]; then
    WANDB_ENTITY="$(python -c 'import wandb; print(wandb.Api().default_entity or "")')"
    export WANDB_ENTITY
fi

echo "[INFO] WANDB CONFIG"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "WANDB_ENTITY: ${WANDB_ENTITY}"

hands=( "allegro_hand" "dex3_hand" "inspire_hand" "xhand" )
clips=( "box-30-230" )
#clips=( "box-30-230" "mixer-0-150" )
seeds=( 0 1000 2000 )

for seed in "${seeds[@]}"; do
    for hand in "${hands[@]}"; do
        for clip in "${clips[@]}"; do

            # Extract the object name to safely check if retarget data exists
            obj_name=$(echo "$clip" | cut -d'-' -f1)
            data_file="${DEXMACHINA_ASSETS}/contact_retarget/${hand}/s01/${obj_name}_use_01.npy"

            if [ ! -f "$data_file" ]; then
                echo "[SKIP] Retarget data not found: ${hand} + ${obj_name}. Moving to next..."
                continue
            fi

            run_exp_name="${hand}_${clip}_seed${seed}"
            echo "========================================================"
            echo "[START] Training: $run_exp_name"
            echo "========================================================"

            # 1. RUN HEADLESS TRAINING AT MAX SPEED
	    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
            python src/jaxrl/reppo.py \
                env=dexmachina \
                seed=${seed} \
                env.hand=${hand} \
                env.clip=${clip} \
                ++exp_name="${run_exp_name}" \
                ++env.name="dexmachina" \
                ++wandb.project="${WANDB_PROJECT}" \
                ++wandb.entity="${WANDB_ENTITY:-null}" \
                ++env.retarget_name="para" \
                ++hyperparameters.total_time_steps=50_000_896 \
                ++hyperparameters.num_envs=8192 \
                ++hyperparameters.num_eval=16 \
		        ++hyperparameters.num_steps=32 \
		        ++hyperparameters.num_mini_batches=512 \
                ++hyperparameters.num_bins=301 \
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
                ++use_rl_games=true \
                ++hyperparameters.actor_min_std=0.05 \


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
            # --voc_mode zero: evaluate with the assistive object controller off,
            # which is what dexmachina/rl/eval_rl_games.py does and what the
            # paper's ADD/AUC/success numbers are measured under. Add a second
            # pass with --voc_mode replay if you also want assisted videos.
	    python debug_render.py --ckpt_dir "${CKPT_DIR}" --wandb_project "${WANDB_PROJECT}" --wandb_entity "${WANDB_ENTITY}" \
                --voc_mode zero --dump_eval_npy --max_videos ${MAX_VIDEOS}

            # Canonical offline ADD / AUC from the dumped eval_ep0.npy files.
            # Writes add.npy + add_stats.json next to each checkpoint.
            echo "[START] Computing ADD/AUC for: $run_exp_name"
            # --obj_name is required: compute_add's find_config_path() walks up
            # looking for a bare config.yaml, but hydra writes .hydra/config.yaml,
            # so inference always fails here. $obj_name is already parsed from $clip.
            # compute_add builds a fresh 200-env Genesis scene per eval file and
            # leaks them, so it dies partway and costs ~60s each. Run it on the
            # FINAL checkpoint only -- the online eval/add* curves already cover
            # the trajectory; this is just the paper-comparable spot check.
            LAST_CKPT=$(ls -d "${CKPT_DIR}"/checkpoint_seed_*_step_* 2>/dev/null \
                | sed -E 's/.*step_([0-9]+)$/\1 &/' | sort -n | tail -1 | cut -d" " -f2-)
            if [ -n "$LAST_CKPT" ]; then
                echo "[START] Computing ADD/AUC for final checkpoint: $(basename "$LAST_CKPT")"
                python -m dexmachina.eval.compute_add --input "${LAST_CKPT}" --overwrite \
                    --obj_name "${obj_name}" \
                    || echo "[WARN] compute_add failed for ${run_exp_name}"
            fi

            echo "[SUCCESS] All milestone evaluations completed for $run_exp_name"
            echo "--------------------------------------------------------"

        done
    done
done

echo "[FINISHED] All automated training and evaluations are complete!"

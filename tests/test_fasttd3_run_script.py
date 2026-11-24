"""scripts/fasttd3/train.sh must report the final checkpoint, never best.pt.

The real script runs against a temporary copy of the layout it needs, with a fake ``uv`` first on PATH: the fake
"training" writes the checkpoints train.py would write, and the fake "evaluation" records its arguments.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
SEEDS = 3
NUM_TRAIN_ENVS = 2048
TOTAL_TIMESTEPS = 50_000_000 // NUM_TRAIN_ENVS
STEP_CHECKPOINT = f"step{TOTAL_TIMESTEPS // 2}.pt"
WANDB_RUN_ID = "fakerun1"

FAKE_UV = r"""#!/bin/bash
set -e
if [ "$1 $2 $3 $4 $5 $6" != "run --project $FAKE_ROOT --exact --extra fasttd3" ]; then
    echo "fake uv: unexpected call: $*" >&2
    exit 99
fi
shift 6
case "$*" in
    "dexhand-bench env")
        echo "export WANDB_PROJECT=fake-project" ;;
    "python fast_td3/train.py "*)
        exp_name=$(echo "$*" | sed -E 's/.*--exp_name ([^ ]+).*/\1/')
        echo "train $exp_name" >> "$FAKE_LOG"
        ckpt_dir="models/FastTD3/${exp_name}/20261124_000000"
        mkdir -p "$ckpt_dir"
        touch "$ckpt_dir/$FAKE_STEP_CHECKPOINT" "$ckpt_dir/best.pt"
        [ -n "$FAKE_NO_FINAL" ] || touch "$ckpt_dir/final.pt" ;;
    "python fast_td3/eval_dexmachina.py "*)
        shift 2
        [ -f "$2" ] || { echo "fake eval: no checkpoint $2" >&2; exit 1; }
        echo "eval $*" >> "$FAKE_LOG" ;;
    *)
        echo "fake uv: unexpected command: $*" >&2
        exit 99 ;;
esac
"""


def run_train_script(tmp_path: Path, *, write_final: bool) -> tuple[subprocess.CompletedProcess, list[str]]:
    root = tmp_path / "repo"
    (root / "scripts/fasttd3").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts/common.sh", root / "scripts/common.sh")
    shutil.copy(REPO_ROOT / "scripts/fasttd3/train.sh", root / "scripts/fasttd3/train.sh")
    (root / f"algorithms/FastTD3/wandb/run-20261124_000000-{WANDB_RUN_ID}").mkdir(parents=True)
    retarget = root / "environments/dexmachina/dexmachina/assets/contact_retarget/allegro_hand/s01/box_use_01.npy"
    retarget.parent.mkdir(parents=True)
    retarget.touch()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(FAKE_UV)
    (bin_dir / "uv").chmod(0o755)
    log = tmp_path / "calls.log"
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / "venv"),
        "FAKE_ROOT": str(root),
        "FAKE_LOG": str(log),
        "FAKE_STEP_CHECKPOINT": STEP_CHECKPOINT,
    }
    if not write_final:
        env["FAKE_NO_FINAL"] = "1"
    result = subprocess.run(
        ["bash", str(root / "scripts/fasttd3/train.sh")], env=env, capture_output=True, text=True, timeout=60
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


def evaluations_per_run(calls: list[str]) -> list[list[str]]:
    runs: list[list[str]] = []
    for call in calls:
        if call.startswith("train "):
            runs.append([])
        else:
            runs[-1].append(call.removeprefix("eval "))
    return runs


def argument(call: str, flag: str) -> str | None:
    words = call.split()
    return words[words.index(flag) + 1] if flag in words else None


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_each_run_evaluates_the_step_checkpoint_and_the_final_checkpoint(tmp_path):
    result, calls = run_train_script(tmp_path, write_final=True)
    assert result.returncode == 0, result.stderr

    runs = evaluations_per_run(calls)
    assert len(runs) == SEEDS
    for evaluations in runs:
        assert [Path(argument(e, "--checkpoint") or "").name for e in evaluations] == [STEP_CHECKPOINT, "final.pt"]
        assert argument(evaluations[-1], "--step") == str(TOTAL_TIMESTEPS * NUM_TRAIN_ENVS)
        assert all(argument(e, "--wandb_run_id") == WANDB_RUN_ID for e in evaluations)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_missing_final_checkpoint_stops_the_script(tmp_path):
    result, calls = run_train_script(tmp_path, write_final=False)
    assert result.returncode != 0
    assert len(evaluations_per_run(calls)) == 1
    assert "final.pt is missing" in result.stderr

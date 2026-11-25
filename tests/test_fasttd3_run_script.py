"""scripts/fasttd3/train.sh and test_run.sh evaluate every checkpoint on the benchmark protocol, never best.pt.

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
        echo "eval $*" >> "$FAKE_LOG"
        [ -z "$FAKE_EVAL_FAILS" ] || exit 1 ;;
    *)
        echo "fake uv: unexpected command: $*" >&2
        exit 99 ;;
esac
"""


def run_train_script(
    tmp_path: Path, *, write_final: bool = True, eval_fails: bool = False, script: str = "train.sh"
) -> tuple[subprocess.CompletedProcess, list[str]]:
    root = tmp_path / "repo"
    (root / "scripts/fasttd3").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts/common.sh", root / "scripts/common.sh")
    shutil.copy(REPO_ROOT / f"scripts/fasttd3/{script}", root / f"scripts/fasttd3/{script}")
    (root / "algorithms/FastTD3").mkdir(parents=True)
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
    if eval_fails:
        env["FAKE_EVAL_FAILS"] = "1"
    result = subprocess.run(
        ["bash", str(root / f"scripts/fasttd3/{script}")], env=env, capture_output=True, text=True, timeout=60
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


def evaluated_checkpoints(evaluations: list[str]) -> list[str]:
    """The checkpoint names, after checking every call is exactly `--checkpoint <path> --wandb`."""
    words = [evaluation.split() for evaluation in evaluations]
    assert all(len(w) == 3 and w[0] == "--checkpoint" and w[2] == "--wandb" for w in words), evaluations
    return [Path(w[1]).name for w in words]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_each_run_evaluates_the_step_checkpoint_and_the_final_checkpoint(tmp_path):
    result, calls = run_train_script(tmp_path)
    assert result.returncode == 0, result.stderr

    runs = evaluations_per_run(calls)
    assert len(runs) == SEEDS
    for evaluations in runs:
        assert evaluated_checkpoints(evaluations) == [STEP_CHECKPOINT, "final.pt"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_failing_evaluation_stops_the_script_and_names_the_checkpoint(tmp_path):
    result, calls = run_train_script(tmp_path, eval_fails=True)
    assert result.returncode != 0
    ((evaluation,),) = evaluations_per_run(calls)
    (checkpoint,) = evaluated_checkpoints([evaluation])
    assert checkpoint == STEP_CHECKPOINT
    assert f"Evaluating {evaluation.split()[1]} failed" in result.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_smoke_test_evaluates_its_checkpoints_the_same_way(tmp_path):
    result, calls = run_train_script(tmp_path, script="test_run.sh")
    assert result.returncode == 0, result.stderr
    (evaluations,) = evaluations_per_run(calls)
    assert evaluated_checkpoints(evaluations) == [STEP_CHECKPOINT, "final.pt"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_missing_final_checkpoint_stops_the_script(tmp_path):
    result, calls = run_train_script(tmp_path, write_final=False)
    assert result.returncode != 0
    assert len(evaluations_per_run(calls)) == 1
    assert "final.pt is missing" in result.stderr

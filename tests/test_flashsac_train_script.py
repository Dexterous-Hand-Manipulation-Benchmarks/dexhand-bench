"""scripts/flashsac/train.sh and evaluate_run.sh run against a temporary copy of the layout, with a fake `uv`/`python`.

The fake "training" writes what train.py writes (checkpoints, run id file); the fake "evaluation" writes the benchmark
trajectory next to the checkpoint it is given, like eval_dexmachina.py.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")

REPO_ROOT = Path(__file__).parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts/flashsac/train.sh"
EVALUATE_SCRIPT = REPO_ROOT / "scripts/flashsac/evaluate_run.sh"

_eval_spec = importlib.util.spec_from_file_location(
    "flashsac_eval_dexmachina", REPO_ROOT / "algorithms/FlashSAC/eval_dexmachina.py"
)
assert _eval_spec is not None and _eval_spec.loader is not None
eval_dexmachina = sys.modules.get(_eval_spec.name) or importlib.util.module_from_spec(_eval_spec)
if _eval_spec.name not in sys.modules:
    sys.modules[_eval_spec.name] = eval_dexmachina
    _eval_spec.loader.exec_module(eval_dexmachina)

WANDB_RUN_ID = "run1234"
FINAL_STEP = 50_000_896 // 1024  # NUM_ENV_STEPS // NUM_TRAIN_ENVS in train.sh
INTERVAL_STEP = 4882  # CHECKPOINT_INTERVAL in train.sh
FINISHED_RUN = "seed2000-20260101-000000"
EXPERIMENT = "allegro_hand_box-30-230_seed2000_10M_official"
TRAJECTORY = eval_dexmachina.TRAJECTORY_FILENAME.format(sha="abc123")
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def make_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


@needs_bash
@pytest.mark.parametrize("script", [TRAIN_SCRIPT, EVALUATE_SCRIPT])
def test_the_scripts_are_valid_bash(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def run_evaluate_script(
    tmp_path: Path,
    steps: list[int],
    *,
    python_exit_code: int,
    final_step: int | None = 12,
    write_run_id: bool = True,
    write_marker: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    run_dir = tmp_path / "run"
    for step in steps:
        (run_dir / f"step{step}").mkdir(parents=True)
    run_dir.mkdir(exist_ok=True)
    if write_run_id:
        (run_dir / flash_dex.WANDB_RUN_ID_FILENAME).write_text(f"{WANDB_RUN_ID}\n")
    if write_marker:
        (run_dir / flash_dex.TRAINING_FINISHED_FILENAME).touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_log = tmp_path / "python.log"
    make_executable(bin_dir / "python", f'#!/bin/bash\necho "$*" >> "{python_log}"\nexit {python_exit_code}\n')
    final_step_args = [] if final_step is None else [str(final_step)]
    result = subprocess.run(
        ["bash", str(EVALUATE_SCRIPT), str(run_dir), "fake-project", "cuda:0", *final_step_args],
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    return result, run_dir


def remaining_steps(run_dir: Path) -> list[str]:
    return sorted(path.name for path in run_dir.glob("step*"))


@needs_bash
def test_a_successful_evaluation_keeps_only_the_final_checkpoint(tmp_path: Path) -> None:
    result, run_dir = run_evaluate_script(tmp_path, [5, 10, 12], python_exit_code=0)
    assert result.returncode == 0, result.stderr
    assert remaining_steps(run_dir) == ["step12"]
    command = (tmp_path / "python.log").read_text()
    assert f"--checkpoint_path {run_dir}/step12 --wandb_run_id {WANDB_RUN_ID} --project fake-project" in command


@needs_bash
def test_a_failed_evaluation_deletes_no_checkpoint(tmp_path: Path) -> None:
    result, run_dir = run_evaluate_script(tmp_path, [5, 10, 12], python_exit_code=1)
    assert result.returncode == 1
    assert remaining_steps(run_dir) == ["step10", "step12", "step5"]


@needs_bash
def test_a_run_without_the_run_id_file_is_an_error(tmp_path: Path) -> None:
    result, run_dir = run_evaluate_script(tmp_path, [5, 12], python_exit_code=0, write_run_id=False)
    assert result.returncode == 1
    assert flash_dex.WANDB_RUN_ID_FILENAME in result.stderr
    assert remaining_steps(run_dir) == ["step12", "step5"]
    assert not (tmp_path / "python.log").exists()


@needs_bash
def test_a_run_without_the_final_checkpoint_is_an_error(tmp_path: Path) -> None:
    result, run_dir = run_evaluate_script(tmp_path, [5, 10], python_exit_code=0)
    assert result.returncode == 1
    assert "step12 does not exist" in result.stderr
    assert remaining_steps(run_dir) == ["step10", "step5"]
    assert not (tmp_path / "python.log").exists()


@needs_bash
def test_the_evaluation_takes_the_given_final_step_not_the_highest_one(tmp_path: Path) -> None:
    result, run_dir = run_evaluate_script(tmp_path, [5, 12, 20], python_exit_code=0)
    assert result.returncode == 0, result.stderr
    assert remaining_steps(run_dir) == ["step12"]
    assert f"--checkpoint_path {run_dir}/step12 " in (tmp_path / "python.log").read_text()


@needs_bash
def test_a_missing_final_step_argument_is_an_error(tmp_path: Path) -> None:
    result, run_dir = run_evaluate_script(tmp_path, [5, 12], python_exit_code=0, final_step=None)
    assert result.returncode == 1
    assert "usage: evaluate_run.sh" in result.stderr
    assert remaining_steps(run_dir) == ["step12", "step5"]
    assert not (tmp_path / "python.log").exists()


@needs_bash
def test_a_run_without_the_completion_marker_is_not_evaluated(tmp_path: Path) -> None:
    result, run_dir = run_evaluate_script(tmp_path, [5, 12], python_exit_code=0, write_marker=False)
    assert result.returncode == 1
    assert flash_dex.TRAINING_FINISHED_FILENAME in result.stderr
    assert remaining_steps(run_dir) == ["step12", "step5"]
    assert not (tmp_path / "python.log").exists()


FAKE_UV = r"""#!/bin/bash
set -e
shift 6  # run --project <root> --exact --extra flashsac
case "$*" in
    "dexhand-bench env")
        echo "export WANDB_PROJECT=fake-project" ;;
    "python -c "*)
        echo 1 ;;
    "python train.py "*)
        save_path=$(echo "$*" | sed -E 's/.*save_path=([^ ]+).*/\1/')
        echo "train $save_path" >> "$FAKE_LOG"
        mkdir -p "$save_path/step$FAKE_INTERVAL_STEP" "$save_path/step$FAKE_FINAL_STEP"
        echo run1234 > "$save_path/$FAKE_RUN_ID_FILENAME"
        touch "$save_path/$FAKE_MARKER_FILENAME" ;;
    "python eval_dexmachina.py "*)
        echo "eval $*" >> "$FAKE_LOG"
        checkpoint=$(echo "$*" | sed -E 's/.*--checkpoint_path ([^ ]+).*/\1/')
        [ "$FAKE_EVAL_EXIT" = 0 ] || exit "$FAKE_EVAL_EXIT"
        touch "$checkpoint/$FAKE_TRAJECTORY" ;;
    *)
        echo "fake uv: unexpected command: $*" >&2
        exit 99 ;;
esac
"""


def run_train_script(
    tmp_path: Path,
    *,
    existing_steps: tuple[int, ...] = (),
    existing_trajectory: bool = False,
    finished_runs: tuple[str, ...] = (FINISHED_RUN,),
    unfinished_runs: tuple[str, ...] = (),
    eval_exit_code: int = 0,
):
    """Runs train.sh with data for the allegro box run only, so exactly that run of the queue is considered.

    With existing steps, every earlier run in `finished_runs` and `unfinished_runs` has them; only the finished ones
    have the completion marker."""
    root = tmp_path / "repo"
    (root / "scripts/flashsac").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts/common.sh", root / "scripts/common.sh")
    for script in (TRAIN_SCRIPT, EVALUATE_SCRIPT):
        shutil.copy(script, root / "scripts/flashsac" / script.name)
    experiment_dir = root / "algorithms/FlashSAC/models/FlashSAC" / EXPERIMENT
    for name in (*finished_runs, *unfinished_runs) if existing_steps else ():
        previous_run = experiment_dir / "dexmachina/allegro_hand" / name
        for step in existing_steps:
            (previous_run / f"step{step}").mkdir(parents=True)
        (previous_run / flash_dex.WANDB_RUN_ID_FILENAME).write_text(f"{WANDB_RUN_ID}\n")
        if name in finished_runs:
            (previous_run / flash_dex.TRAINING_FINISHED_FILENAME).touch()
        if existing_trajectory:
            trajectory = previous_run / f"step{FINAL_STEP}" / TRAJECTORY
            trajectory.parent.mkdir(parents=True, exist_ok=True)
            trajectory.touch()
    retarget = root / "environments/dexmachina/dexmachina/assets/contact_retarget/allegro_hand/s01/box_use_01.npy"
    retarget.parent.mkdir(parents=True)
    retarget.touch()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_executable(bin_dir / "uv", FAKE_UV)
    log = tmp_path / "calls.log"
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / "venv"),
        "FAKE_LOG": str(log),
        "FAKE_INTERVAL_STEP": str(INTERVAL_STEP),
        "FAKE_FINAL_STEP": str(FINAL_STEP),
        "FAKE_RUN_ID_FILENAME": flash_dex.WANDB_RUN_ID_FILENAME,
        "FAKE_MARKER_FILENAME": flash_dex.TRAINING_FINISHED_FILENAME,
        "FAKE_EVAL_EXIT": str(eval_exit_code),
        "FAKE_TRAJECTORY": TRAJECTORY,
    }
    (root / "algorithms/FlashSAC").mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["bash", str(root / "scripts/flashsac/train.sh")], env=env, capture_output=True, text=True, timeout=60
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls, experiment_dir


@needs_bash
def test_a_new_run_trains_into_a_directory_the_script_names_and_evaluates_it(tmp_path: Path) -> None:
    result, calls, experiment_dir = run_train_script(tmp_path)
    assert result.returncode == 0, result.stderr
    assert [call.split()[0] for call in calls] == ["train", "eval"]
    run_dirs = list((experiment_dir / "dexmachina/allegro_hand").iterdir())
    assert len(run_dirs) == 1
    assert run_dirs[0].name.startswith("seed2000-20")
    run_relative = f"models/FlashSAC/{EXPERIMENT}/dexmachina/allegro_hand/{run_dirs[0].name}"
    assert calls[0] == f"train {run_relative}"
    assert f"--checkpoint_path {run_relative}/step{FINAL_STEP}" in calls[1]
    assert remaining_steps(run_dirs[0]) == [f"step{FINAL_STEP}"]


@needs_bash
def test_a_finished_training_without_an_evaluation_is_only_evaluated(tmp_path: Path) -> None:
    result, calls, experiment_dir = run_train_script(tmp_path, existing_steps=(INTERVAL_STEP, FINAL_STEP))
    assert result.returncode == 0, result.stderr
    assert [call.split()[0] for call in calls] == ["eval"]
    assert remaining_steps(experiment_dir / "dexmachina/allegro_hand" / FINISHED_RUN) == [f"step{FINAL_STEP}"]


@needs_bash
def test_a_run_with_an_evaluation_is_skipped(tmp_path: Path) -> None:
    result, calls, _ = run_train_script(tmp_path, existing_steps=(FINAL_STEP,), existing_trajectory=True)
    assert result.returncode == 0, result.stderr
    assert calls == []


@needs_bash
def test_a_final_checkpoint_without_the_completion_marker_is_retrained(tmp_path: Path) -> None:
    unfinished = "seed2000-20260101-000000"
    result, calls, experiment_dir = run_train_script(
        tmp_path, existing_steps=(INTERVAL_STEP, FINAL_STEP), finished_runs=(), unfinished_runs=(unfinished,)
    )
    assert result.returncode == 0, result.stderr
    assert [call.split()[0] for call in calls] == ["train", "eval"]
    assert remaining_steps(experiment_dir / "dexmachina/allegro_hand" / unfinished) == [
        f"step{INTERVAL_STEP}",
        f"step{FINAL_STEP}",
    ]


@needs_bash
def test_several_finished_trainings_without_an_evaluation_are_an_error(tmp_path: Path) -> None:
    result, calls, _ = run_train_script(
        tmp_path,
        existing_steps=(FINAL_STEP,),
        finished_runs=("seed2000-20260101-000000", "seed2000-20260102-000000"),
    )
    assert result.returncode == 1
    assert "several finished trainings" in result.stderr
    assert calls == []


@needs_bash
def test_a_failing_evaluation_in_train_sh_keeps_every_checkpoint(tmp_path: Path) -> None:
    result, calls, experiment_dir = run_train_script(
        tmp_path, existing_steps=(INTERVAL_STEP, FINAL_STEP), eval_exit_code=1
    )
    assert result.returncode == 1
    assert [call.split()[0] for call in calls] == ["eval"]
    assert remaining_steps(experiment_dir / "dexmachina/allegro_hand" / FINISHED_RUN) == [
        f"step{INTERVAL_STEP}",
        f"step{FINAL_STEP}",
    ]

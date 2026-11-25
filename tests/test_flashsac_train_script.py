"""train.sh evaluates with the shared protocol; the checks read the script text, because running it needs GPUs."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/flashsac/train.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def evaluation_command(script_text: str) -> str:
    match = re.search(r"python eval_dexmachina\.py(?:[^\n]*\\\n)*[^\n]*\n", script_text)
    assert match is not None, "train.sh runs no eval_dexmachina.py"
    return match.group(0)


def test_the_evaluation_uses_the_shared_protocol_on_the_run_it_trained(script_text):
    command = evaluation_command(script_text)
    assert "--old-protocol" not in script_text
    assert "--checkpoint_path" in command
    assert '--wandb_run_id "${WANDB_RUN_ID}"' in command


def test_the_wandb_run_id_comes_from_the_run_directory_not_from_the_newest_wandb_folder(script_text):
    assert "wandb_run_id.txt" in script_text
    assert "ls -td wandb" not in script_text


def test_a_run_is_skipped_when_its_benchmark_trajectory_exists(script_text):
    assert "benchmark_trajectory_*.npz" in script_text

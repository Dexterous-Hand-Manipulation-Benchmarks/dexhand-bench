"""scripts/slurm/submit.sh must create logs/ before sbatch, because SLURM opens the log files before the job starts."""

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]


def test_submit_creates_the_logs_directory_before_sbatch(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "scripts" / "slurm").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts/slurm/submit.sh", root / "scripts/slurm/submit.sh")
    (root / "scripts/slurm/ppo.sbatch").write_text("#!/bin/bash\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name, body in {
        "uv": "echo '--partition=fake'",
        "sbatch": '[ -d logs ] || { echo "sbatch: logs/ is missing" >&2; exit 1; }',
    }.items():
        (fake_bin / name).write_text(f"#!/bin/bash\n{body}\n")
        (fake_bin / name).chmod(0o755)

    result = subprocess.run(
        ["bash", str(root / "scripts/slurm/submit.sh"), "ppo"],
        env={"PATH": f"{fake_bin}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (root / "logs").is_dir()

"""scripts/common.sh must stop the calling script when the W&B settings cannot be loaded."""

import subprocess
from pathlib import Path

COMMON_SH = Path(__file__).parents[1] / "scripts" / "common.sh"


def _source_common_sh(tmp_path: Path, uv_script: str) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(f"#!/bin/bash\n{uv_script}\n")
    fake_uv.chmod(0o755)
    return subprocess.run(
        ["bash", "-c", f'DHB_EXTRA=flashsac; source "{COMMON_SH}"; echo reached-the-caller'],
        env={"PATH": f"{fake_bin}:/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
    )


def test_failing_config_load_stops_the_caller(tmp_path: Path):
    result = _source_common_sh(tmp_path, "echo 'export WANDB_PROJECT=half-loaded'; exit 3")
    assert result.returncode != 0
    assert "reached-the-caller" not in result.stdout
    assert "dexhand-bench env" in result.stderr


def test_successful_config_load_exports_the_settings(tmp_path: Path):
    result = _source_common_sh(tmp_path, "echo 'export WANDB_PROJECT=fake-project'")
    assert result.returncode == 0, result.stderr
    assert "W&B: <default entity>/fake-project" in result.stdout
    assert "reached-the-caller" in result.stdout

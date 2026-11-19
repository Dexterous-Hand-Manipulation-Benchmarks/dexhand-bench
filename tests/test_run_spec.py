"""A run's env_spec.json: round trip, the recorded dexmachina commit, and loud errors for bad files."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("dexmachina.envs.constructors", reason="needs genesis and dexmachina")
import dexmachina.envs.run_spec as run_spec  # noqa: E402  (after the skip: needs dexmachina)
from dexmachina.envs.spec import EnvSpec  # noqa: E402

SPEC = replace(EnvSpec(), hand="allegro_hand", clip="box-30-230-s01-01", hybrid_scales=(0.1, 0.2), use_rl_games=True)
REWARD_OVERRIDES = {"force_penalty": 0.1, "task_rew_weight": 1.0}


def _git(repo: Path, *args: str) -> str:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    done = subprocess.run(["git", *identity, "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


@pytest.fixture(autouse=True)
def fake_dexmachina(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "dexmachina"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "file.py").write_text("x = 1\n")
    _git(repo, "add", "file.py")
    _git(repo, "commit", "-q", "-m", "first")
    monkeypatch.setattr(run_spec, "DEXMACHINA_ROOT", repo)
    return repo


def test_git_helper_outside_a_git_repo_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_spec, "DEXMACHINA_ROOT", tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    with pytest.raises(RuntimeError, match="editable install"):
        run_spec.save_run_spec(tmp_path / "run", SPEC, {})


def test_round_trip(tmp_path: Path) -> None:
    run_spec.save_run_spec(tmp_path, SPEC, REWARD_OVERRIDES)
    loaded = run_spec.load_run_spec(tmp_path)
    assert loaded.spec == SPEC
    assert dict(loaded.reward_overrides) == REWARD_OVERRIDES


def test_saves_into_a_directory_that_does_not_exist_yet(tmp_path: Path) -> None:
    path = run_spec.save_run_spec(tmp_path / "new" / "run", SPEC, {})
    assert path.is_file()


def test_records_the_commit_and_whether_the_checkout_was_dirty(tmp_path: Path, fake_dexmachina: Path) -> None:
    clean, dirty = tmp_path / "clean", tmp_path / "dirty"
    run_spec.save_run_spec(clean, SPEC, {})
    (fake_dexmachina / "file.py").write_text("x = 2\n")
    run_spec.save_run_spec(dirty, SPEC, {})
    first, second = run_spec.load_run_spec(clean), run_spec.load_run_spec(dirty)
    assert first.dexmachina_sha == second.dexmachina_sha == _git(fake_dexmachina, "rev-parse", "HEAD")
    assert (first.dexmachina_dirty, second.dexmachina_dirty) == (False, True)


def test_untracked_files_make_the_checkout_dirty(tmp_path: Path, fake_dexmachina: Path) -> None:
    (fake_dexmachina / "new.py").write_text("y = 1\n")
    run_spec.save_run_spec(tmp_path, SPEC, {})
    assert run_spec.load_run_spec(tmp_path).dexmachina_dirty


def test_a_directory_without_the_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="predates it"):
        run_spec.load_run_spec(tmp_path)


def test_an_existing_file_is_never_replaced(tmp_path: Path) -> None:
    run_spec.save_run_spec(tmp_path, SPEC, {})
    with pytest.raises(FileExistsError, match="exists already"):
        run_spec.save_run_spec(tmp_path, replace(SPEC, hand="inspire_hand"), {})
    assert run_spec.load_run_spec(tmp_path).spec == SPEC


@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_a_file_with_other_keys_than_a_run_spec_is_refused(tmp_path: Path, change: str) -> None:
    path = run_spec.save_run_spec(tmp_path, SPEC, {})
    content: dict[str, Any] = json.loads(path.read_text())
    if change == "missing":
        del content[run_spec.SHA_KEY]
    else:
        content["extra"] = 1
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="does not have the keys of a run spec"):
        run_spec.load_run_spec(tmp_path)


@pytest.mark.parametrize("key, value", [(run_spec.SHA_KEY, 123), (run_spec.DIRTY_KEY, "no")])
def test_a_file_with_a_wrongly_typed_commit_record_is_refused(tmp_path: Path, key: str, value: object) -> None:
    path = run_spec.save_run_spec(tmp_path, SPEC, {})
    content: dict[str, Any] = json.loads(path.read_text())
    content[key] = value
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="must be a string and a boolean"):
        run_spec.load_run_spec(tmp_path)

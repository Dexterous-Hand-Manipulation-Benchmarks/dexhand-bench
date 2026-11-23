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


def _diff_sha256() -> str | None:
    return run_spec.current_dexmachina_state().diff_sha256


def test_the_hash_is_none_for_a_clean_checkout_and_stable_for_the_same_changes(fake_dexmachina: Path) -> None:
    assert _diff_sha256() is None
    (fake_dexmachina / "file.py").write_text("x = 2\n")
    (fake_dexmachina / "new.py").write_text("y = 1\n")
    first = _diff_sha256()
    assert first is not None and _diff_sha256() == first


@pytest.mark.parametrize(
    "change",
    ["deleted file", "binary file", "untracked content", "untracked name", "symlink target"],
)
def test_the_hash_tells_apart_changes_that_differ_only_in(fake_dexmachina: Path, change: str) -> None:
    def make(variant: int) -> str | None:
        for path in fake_dexmachina.iterdir():
            if path.name != ".git":
                path.unlink()
        (fake_dexmachina / "file.py").write_text("x = 1\n")
        if change == "deleted file":
            (fake_dexmachina / "file.py").write_text("x = 2\n")
            if variant:
                (fake_dexmachina / "file.py").unlink()
        elif change == "binary file":
            (fake_dexmachina / "blob.bin").write_bytes(bytes([0, 255, variant]))
        elif change == "untracked content":
            (fake_dexmachina / "new.py").write_text(f"y = {variant}\n")
        elif change == "untracked name":
            (fake_dexmachina / f"new{variant}.py").write_text("y = 1\n")
        else:
            (fake_dexmachina / "link").symlink_to(f"target{variant}")
        return _diff_sha256()

    first, second = make(0), make(1)
    assert first is not None and second is not None and first != second


def test_an_untracked_nested_repository_is_refused_with_the_fix(fake_dexmachina: Path) -> None:
    nested = fake_dexmachina / "nested"
    nested.mkdir()
    _git(nested, "init", "-q")
    (nested / "a.py").write_text("z = 1\n")
    with pytest.raises(RuntimeError, match="neither a file nor a symlink"):
        run_spec.current_dexmachina_state()


def test_saving_from_a_dirty_checkout_warns_with_the_fix(
    tmp_path: Path, fake_dexmachina: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_spec.save_run_spec(tmp_path / "clean", SPEC, {})
    assert "WARNING" not in capsys.readouterr().out
    (fake_dexmachina / "file.py").write_text("x = 2\n")
    run_spec.save_run_spec(tmp_path / "dirty", SPEC, {})
    warning = capsys.readouterr().out
    assert "uncommitted changes" in warning and "Commit or stash" in warning
    diff_sha256 = run_spec.load_run_spec(tmp_path / "dirty").dexmachina_diff_sha256
    assert diff_sha256 is not None and diff_sha256 in warning


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


@pytest.mark.parametrize(
    "key, value", [(run_spec.SHA_KEY, 123), (run_spec.DIRTY_KEY, "no"), (run_spec.DIFF_KEY, 1)]
)
def test_a_file_with_a_wrongly_typed_commit_record_is_refused(tmp_path: Path, key: str, value: object) -> None:
    path = run_spec.save_run_spec(tmp_path, SPEC, {})
    content: dict[str, Any] = json.loads(path.read_text())
    content[key] = value
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="must be a string, a boolean and a string or null"):
        run_spec.load_run_spec(tmp_path)


def test_a_hash_on_a_clean_checkout_is_refused(tmp_path: Path) -> None:
    path = run_spec.save_run_spec(tmp_path, SPEC, {})
    content: dict[str, Any] = json.loads(path.read_text())
    content[run_spec.DIFF_KEY] = "0" * 64
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="only a checkout with uncommitted changes"):
        run_spec.load_run_spec(tmp_path)


def test_current_state_is_the_commit_and_dirty_flag_that_save_run_spec_records(
    tmp_path: Path, fake_dexmachina: Path
) -> None:
    clean = run_spec.current_dexmachina_state()
    assert clean == run_spec.DexmachinaState(_git(fake_dexmachina, "rev-parse", "HEAD"), False, None)
    (fake_dexmachina / "file.py").write_text("x = 2\n")
    assert run_spec.current_dexmachina_state().dirty
    run_spec.save_run_spec(tmp_path, SPEC, {})
    saved = run_spec.load_run_spec(tmp_path)
    assert run_spec.current_dexmachina_state() == saved.dexmachina_state

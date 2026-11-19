"""Reconstructing env_spec.json and config.yaml of the lab FlashSAC runs from their W&B config.

The expected values do not come from the script: hand, clip, seed and the run name are the lab report's table (hands
allegro, inspire, xhand, dex3; seeds 0, 1000, 2000; box-30-230 over 200 steps, 5e7 env steps) and the lab's run naming;
the two mixer clips only appear in the run names. The action mode is residual (the lab runs had no action_mode key).
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
from dexmachina.envs.constructors import parse_clip_string  # noqa: E402  (after the skip: needs dexmachina)
from dexmachina.envs.run_spec import load_run_spec  # noqa: E402
from dexmachina.envs.spec import EnvSpec  # noqa: E402

SCRIPT = Path(__file__).parents[1] / "scripts/analysis/write_lab_run_files.py"
module_spec = importlib.util.spec_from_file_location("write_lab_run_files", SCRIPT)
assert module_spec is not None and module_spec.loader is not None
lab = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = lab
module_spec.loader.exec_module(lab)

EXPORT = Path(__file__).parents[1] / "notes/wandb_runs.json"
CHECKPOINTS = Path(__file__).parents[1] / "notes/measurements/bridge_eval/ckpt"
SEEDS = (0, 1000, 2000)
# (hand, clip, first frame, last frame): the clip's frames are the 200 steps of the report's table (150 for mixer).
LAB_CLIPS = (
    ("allegro_hand", "box-30-230", 30, 230),
    ("inspire_hand", "box-30-230", 30, 230),
    ("xhand", "box-30-230", 30, 230),
    ("dex3_hand", "box-30-230", 30, 230),
    ("allegro_hand", "mixer-0-150", 0, 150),
    ("inspire_hand", "mixer-0-150", 0, 150),
)
LAB_RUNS = [(hand, clip, start, end, seed) for hand, clip, start, end in LAB_CLIPS for seed in SEEDS]
TEAM_NAME = "not-a-real-team"
# The report's shared-config table, row "Environment steps": 5.0e7.
REPORT_ENV_STEPS = 50_000_000
CHECKPOINT_STEP = 48820
CHECKPOINT_FILES = ("actor.pt", "obs_normalizer.pt")


def run_name(hand: str, clip: str, seed: int) -> str:
    return f"{hand}_{clip}_seed{seed}_10M_official"


def wandb_config(hand: str, clip: str, seed: int) -> dict[str, Any]:
    """The W&B config of a lab run: the keys and values that the export of a lab run has."""
    return {
        "agent": {
            "agent_type": "flashSAC", "actor_hidden_dim": 256, "normalize_observation": True, "seed": seed,
            "asymmetric_observation": False,
        },
        "agent_load_path": None, "buffer_load_path": None, "evaluation_per_interaction_step": 4882,
        "exp_name": run_name(hand, clip, seed), "gamma": 0.99, "group_name": "FlashSAC", "logger_type": "wandb",
        "logging_per_interaction_step": 488, "metrics_per_interaction_step": 4882, "n_step": 3,
        "num_env_steps": 50_000_896, "num_eval_envs": 1024, "num_eval_episodes": 1024, "num_interaction_steps": 48829,
        "num_record_envs": None, "num_record_episodes": 0, "num_train_envs": 1024, "num_update_steps": 195316,
        "project_name": "lab_project", "recording_per_interaction_step": 48829,
        "save_buffer_per_interaction_step": None,
        "save_checkpoint_per_interaction_step": 4882, "save_path": "models/FlashSAC/x/seed0-TIMESTAMP", "seed": seed,
        "updates_per_interaction_step": 4, "entity_name": TEAM_NAME,
        "actor_blocks": 2, "actor_hidden_dim": 256, "algo": "FlashSAC", "batch_size": 4096, "clip": clip,
        "critic_blocks": 2, "critic_hidden_dim": 512, "curriculum": True, "hand": hand, "learning_rate": 0.0003,
        "normalize_reward": True, "num_envs": 1024, "use_amp": True,
        "benchmark/hand": hand, "benchmark/clip": clip, "benchmark/seed": seed,
        "env": {
            "clip": clip, "curriculum": True, "env_name": f"dexmachina/{hand}", "env_type": "dexmachina", "hand": hand,
            "max_episode_steps": None, "num_env_steps": 50_000_896, "num_eval_envs": None, "num_record_envs": None,
            "num_train_envs": 1024, "rescale_action": None, "retarget_name": "para", "rl_device": "cuda:0",
            "seed": seed,
            "reward_cfg": {
                "action_penalty": 0.01, "bc_beta": 500, "bc_rew_weight": 0.3, "contact_beta": 10,
                "contact_rew_weight": 3, "force_penalty": 0.1, "imi_rew_weight": 0.3, "imi_wrist_weight": 0.5,
                "obj_arti_beta": 5, "obj_pos_beta": 10, "obj_rot_beta": 1, "task_rew_weight": 1,
                "use_retarget_contact": True,
            },
        },
    }  # fmt: skip


def write(
    tmp_path: Path,
    hand: str = "allegro_hand",
    clip: str = "box-30-230",
    seed: int = 0,
    changes: dict[str, Any] | None = None,
    saved_dirname: str = f"step{CHECKPOINT_STEP}",
    flashsac_commit: str = lab.UPDATE_WITH_NAN_CHECK,
) -> Path:
    """Runs the script's per-run step on a fake checkpoint root and returns the run's output directory.

    Like the lab layout, the evalseed0 files are links into the step directory the run saved.
    """
    name = run_name(hand, clip, seed)
    saved = tmp_path / "saved" / name / saved_dirname
    saved.mkdir(parents=True)
    source = tmp_path / "ckpt" / name / lab.LAB_CHECKPOINT_SUBDIR
    source.mkdir(parents=True)
    for filename in CHECKPOINT_FILES:
        (saved / filename).write_bytes(b"weights")
        (source / filename).symlink_to(saved / filename)
    config = {**wandb_config(hand, clip, seed), **(changes or {})}
    lab.write_run_files(name, lab.LabRun(config, flashsac_commit), tmp_path / "ckpt", tmp_path / "out")
    return tmp_path / "out" / name


@pytest.mark.parametrize(("hand", "clip", "start", "end", "seed"), LAB_RUNS)
def test_spec_matches_the_lab_table_and_the_run_name(
    tmp_path: Path, hand: str, clip: str, start: int, end: int, seed: int
) -> None:
    run = load_run_spec(write(tmp_path, hand, clip, seed))
    spec = run.spec
    assert (spec.hand, spec.seed) == (hand, seed)
    assert spec.action_mode == "residual"
    assert parse_clip_string(spec.clip)[:3] == (clip.split("-")[0], start, end)
    assert spec.clip == f"{clip}-s01-01"
    assert spec.num_envs * CHECKPOINT_STEP == pytest.approx(REPORT_ENV_STEPS, rel=1e-3)
    assert run.dexmachina_sha == lab.LAB_DEXMACHINA_SHA
    assert run.dexmachina_dirty  # unknown working tree: recorded as dirty so that the eval warns


def test_rewards_of_the_run_are_kept(tmp_path: Path) -> None:
    run = load_run_spec(write(tmp_path))
    assert run.reward_overrides["force_penalty"] == 0.1
    assert run.spec.task_rew_betas == (10, 1, 5)
    assert run.spec.imi_rew_weight == 0.3
    assert run.spec.actuate_object


def test_marker_lists_every_field_filled_from_a_default(tmp_path: Path) -> None:
    run_dir = write(tmp_path)
    marker = json.loads((run_dir / lab.RECONSTRUCTED_FILENAME).read_text())
    defaulted = marker["filled_from_envspec_defaults"]
    spec_json = json.loads(load_run_spec(run_dir).spec.to_json())
    defaults = json.loads(EnvSpec().to_json())
    assert {"kp_init", "first_stop_iter"} <= defaulted.keys()
    assert defaulted["kp_init"] == 100.0  # EnvSpec's default, written here independently of the script
    assert "hand" not in defaulted
    assert not lab.CLIP_DERIVED_SPEC_FIELDS & defaulted.keys()
    set_fields = {"hand", "clip", "seed", "num_envs", "action_mode", "imi_rew_weight", "task_rew_betas"}
    assert not set_fields & defaulted.keys()
    assert all(defaulted[field] == defaults[field] == spec_json[field] for field in defaulted)
    changed = {field for field in spec_json if spec_json[field] != defaults[field]}
    assert changed.isdisjoint(defaulted.keys())
    assert marker["set_by_lab_code"]["action_mode"] == "residual"
    assert marker["dexmachina_sha_of_lab_code"] == lab.LAB_DEXMACHINA_SHA


def test_step_directory_is_the_layout_evaluate_benchmark_reads(tmp_path: Path) -> None:
    run_dir = write(tmp_path)
    step_dir = run_dir / f"step{CHECKPOINT_STEP}"
    assert sorted(path.name for path in step_dir.iterdir()) == sorted(CHECKPOINT_FILES)
    assert (step_dir / "actor.pt").read_bytes() == b"weights"
    assert {path.name for path in run_dir.iterdir() if path.is_file()} == {
        "env_spec.json", "config.yaml", lab.RECONSTRUCTED_FILENAME,
    }  # fmt: skip
    assert OmegaConf.load(run_dir / "config.yaml").agent.normalize_observation is True


def test_checkpoint_step_is_the_last_saved_one() -> None:
    assert lab.checkpoint_step(wandb_config("allegro_hand", "box-30-230", 0)) == CHECKPOINT_STEP


def test_the_team_name_is_not_written(tmp_path: Path) -> None:
    run_dir = write(tmp_path)
    assert all(TEAM_NAME not in path.read_text() for path in run_dir.iterdir() if path.is_file())
    assert OmegaConf.load(run_dir / "config.yaml").entity_name is None


def test_unmapped_top_level_key_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="new_option"):
        write(tmp_path, changes={"new_option": 1})


def test_unmapped_env_key_raises(tmp_path: Path) -> None:
    config = wandb_config("allegro_hand", "box-30-230", 0)
    env = {**config["env"], "action_mode": "hybrid"}
    with pytest.raises(ValueError, match="action_mode"):
        write(tmp_path, changes={"env": env})


def test_unmapped_reward_key_raises(tmp_path: Path) -> None:
    config = wandb_config("allegro_hand", "box-30-230", 0)
    env = {**config["env"], "reward_cfg": {**config["env"]["reward_cfg"], "new_reward": 1.0}}
    with pytest.raises(ValueError, match="new_reward"):
        write(tmp_path, changes={"env": env})


def test_set_action_rescaling_raises(tmp_path: Path) -> None:
    config = wandb_config("allegro_hand", "box-30-230", 0)
    with pytest.raises(ValueError, match="rescale_action"):
        write(tmp_path, changes={"env": {**config["env"], "rescale_action": 0.5}})


def test_logger_copy_that_disagrees_with_the_env_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hand"):
        write(tmp_path, changes={"hand": "xhand"})


def test_missing_checkpoint_files_raise(tmp_path: Path) -> None:
    name = run_name("allegro_hand", "box-30-230", 0)
    (tmp_path / "ckpt" / name / lab.LAB_CHECKPOINT_SUBDIR).mkdir(parents=True)
    run = lab.LabRun(wandb_config("allegro_hand", "box-30-230", 0), lab.UPDATE_WITH_NAN_CHECK)
    with pytest.raises(FileNotFoundError, match="actor.pt"):
        lab.write_run_files(name, run, tmp_path / "ckpt", tmp_path / "out")


def test_checkpoint_that_is_not_the_last_saved_step_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="step48820"):
        write(tmp_path, saved_dirname="step4882")


def test_missing_hydra_key_raises(tmp_path: Path) -> None:
    config = wandb_config("allegro_hand", "box-30-230", 0)
    del config["n_step"]
    run = lab.LabRun(config, lab.UPDATE_WITH_NAN_CHECK)
    with pytest.raises(ValueError, match="n_step"):
        lab.hydra_config("run", run.config)


def test_curriculum_false_leaves_the_object_unactuated(tmp_path: Path) -> None:
    config = wandb_config("allegro_hand", "box-30-230", 0)
    changes = {"curriculum": False, "env": {**config["env"], "curriculum": False}}
    assert not load_run_spec(write(tmp_path, changes=changes)).spec.actuate_object


def test_curriculum_copy_that_disagrees_with_the_env_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="curriculum"):
        write(tmp_path, changes={"curriculum": False})


def test_marker_records_what_differs_from_training_and_the_unknown_dirty_state(tmp_path: Path) -> None:
    marker = json.loads((write(tmp_path) / lab.RECONSTRUCTED_FILENAME).read_text())
    assert marker["dexmachina_dirty"].startswith("unknown")
    differences = " ".join(marker["known_differences_from_training"])
    assert "scaled twice" in differences
    assert "with the NaN check" in differences
    assert marker["derived_from_clip"]["frame_start"] == 30
    assert marker["derived_from_clip"]["frame_end"] == 230
    assert marker["derived_from_clip"]["spec_file_holds"]["frame_start"] == 40  # EnvSpec's default


def test_older_update_commit_is_recorded_as_without_the_nan_check(tmp_path: Path) -> None:
    marker = json.loads(
        (write(tmp_path, flashsac_commit=lab.UPDATE_WITHOUT_NAN_CHECK) / lab.RECONSTRUCTED_FILENAME).read_text()
    )
    assert "without the NaN check" in " ".join(marker["known_differences_from_training"])


def test_unknown_flashsac_commit_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="UPDATE_VARIANTS"):
        write(tmp_path, flashsac_commit="0" * 40)


def test_existing_out_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError, match="--out-dir"):
        lab.main(["--export", "e.json", "--checkpoint-root", str(tmp_path), "--out-dir", str(tmp_path)])


def test_two_runs_with_one_name_raise(tmp_path: Path) -> None:
    run = {
        "group": "FlashSAC",
        "config": wandb_config("allegro_hand", "box-30-230", 0),
        "metadata": {"git": {"commit": lab.UPDATE_WITH_NAN_CHECK}},
    }
    export = tmp_path / "export.json"
    export.write_text(json.dumps([run, run]))
    with pytest.raises(ValueError, match="two FlashSAC runs"):
        lab.lab_runs(export)


@pytest.mark.skipif(not EXPORT.is_file(), reason="needs the local W&B export")
def test_the_export_has_the_18_lab_runs_with_their_names() -> None:
    """Runs only where the git-ignored W&B export exists (a local check against the real runs)."""
    runs = lab.lab_runs(EXPORT)
    assert sorted(runs) == sorted(run_name(hand, clip, seed) for hand, clip, _, _, seed in LAB_RUNS)
    for name, run in runs.items():
        lab.hydra_config(name, run.config)
        spec, _, _ = lab.reconstruct_spec(name, run.config)
        assert name == run_name(spec.hand, spec.clip.removesuffix("-s01-01"), spec.seed)
        assert lab.checkpoint_step(run.config) == CHECKPOINT_STEP
        assert run.flashsac_commit in lab.UPDATE_VARIANTS
        # build_eval_agent refuses asymmetric observations.
        assert run.config["agent"]["asymmetric_observation"] is False


def _flashsac_eval_module() -> Any:
    path = Path(__file__).parents[1] / "algorithms/FlashSAC/eval_dexmachina.py"
    module_spec = importlib.util.spec_from_file_location("flashsac_eval_dexmachina", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not (EXPORT.is_file() and CHECKPOINTS.is_dir()), reason="needs the local export and checkpoints")
def test_reconstructed_files_load_and_act_like_the_lab_evaluation_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Local check on allegro box seed 0: the agent built from the written config acts as the bridging eval's loader.

    The reference loader strips the `_orig_mod.` prefix (the lab checkpoints were saved from torch.compile'd networks)
    and runs eager, as notes/measurements/bridge_eval/eval_wrapper.py does.
    """
    import torch

    monkeypatch.setattr("torch._dynamo.config.disable", True)
    evaluation = _flashsac_eval_module()
    name = run_name("allegro_hand", "box-30-230", 0)
    step_dir = lab.write_run_files(name, lab.lab_runs(EXPORT)[name], CHECKPOINTS, tmp_path)
    agent_config = OmegaConf.load(step_dir.parent / "config.yaml").agent
    source = CHECKPOINTS / name / lab.LAB_CHECKPOINT_SUBDIR
    normalizer = torch.load(source / "obs_normalizer.pt", map_location="cpu")
    actor_state = torch.load(source / "actor.pt", map_location="cpu")["network_state_dict"]
    obs_dim = int(normalizer["_mean"].shape[1])
    action_dim = int(next(v for k, v in actor_state.items() if k.endswith("predictor.std_w.w.weight")).shape[0])
    generator = torch.Generator().manual_seed(0)
    observations = normalizer["_mean"] + normalizer["_std"] * torch.randn(5, obs_dim, generator=generator)

    saved_agent = evaluation.build_eval_agent(agent_config, obs_dim, action_dim, 1.0, "cpu")
    saved_agent.load(str(step_dir))
    actions = evaluation.deterministic_actions(saved_agent, observations, "cpu")

    plain = OmegaConf.merge(agent_config, {"use_compile": False})
    torch_load = torch.load

    def load_without_compile_prefix(*args: Any, **kwargs: Any) -> Any:
        checkpoint = torch_load(*args, **kwargs)
        if isinstance(checkpoint, dict) and "network_state_dict" in checkpoint:
            state = checkpoint["network_state_dict"]
            checkpoint["network_state_dict"] = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        return checkpoint

    monkeypatch.setattr(torch, "load", load_without_compile_prefix)
    reference_agent = evaluation.build_eval_agent(plain, obs_dim, action_dim, 1.0, "cpu")
    reference_agent.load(str(source))
    reference = evaluation.deterministic_actions(reference_agent, observations, "cpu")

    difference = float((actions - reference).abs().max())
    with capsys.disabled():
        print(f"\nmax |action difference| over 5 observations: {difference:.3e}, share of actions below 0.99 in "
              f"magnitude: {float((actions.abs() < 0.99).float().mean()):.2f}")
    assert difference < 1e-6
    assert float((actions.abs() < 0.99).float().mean()) > 0.1  # tanh not saturated everywhere, so the check bites

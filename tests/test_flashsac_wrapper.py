"""Unit tests for FlashSAC's DexMachina wrapper against a fake env (no simulator)."""
import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from conftest import flashsac_env_kwargs
from fakes import FakeDexEnv, FakeDexVecEnv

flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")
from dexmachina.envs.benchmark_task import BENCHMARK_TASK  # noqa: E402

BUDGET = BENCHMARK_TASK.budget_env_steps
assert BUDGET is not None


def make(**kwargs):
    raw = FakeDexEnv(**kwargs)
    return raw, flash_dex.GenesisFlashSACWrapper(FakeDexVecEnv(raw), rl_device="cpu")


def test_final_obs_is_the_pre_reset_observation():
    raw, env = make(episode_length=3)
    env.reset()
    for _ in range(2):
        obs, _, term, trunc, infos = env.step(torch.zeros(2, 2))
        assert not (term.any() or trunc.any())
        torch.testing.assert_close(infos["final_obs"], obs)

    obs, _, term, trunc, infos = env.step(torch.zeros(2, 2))
    assert term.all() and not trunc.any()
    torch.testing.assert_close(obs[:, 0], torch.tensor([0.0, 100.0]))  # already reset
    torch.testing.assert_close(infos["final_obs"][:, 0], torch.tensor([3.0, 103.0]))  # last obs of the episode


def test_actions_are_clamped_and_broadcast():
    raw, env = make()
    env.reset()
    env.step(torch.full((1, 2), 5.0))
    torch.testing.assert_close(raw.received_actions, torch.ones(2, 2))


def test_success_metrics_are_the_env_success_rates():
    raw, env = make()
    raw.success = {"success_2cm": torch.tensor([True, False]), "success_5cm": torch.tensor([True, True])}
    env.reset()
    *_, infos = env.step(torch.zeros(2, 2))
    assert infos["train/success_2cm"] == 0.5
    assert infos["train/success_5cm"] == 1.0
    assert math.isclose(infos["env/pos_dist"], raw.dist["pos_dist"], rel_tol=1e-6)


def test_tracking_errors_report_every_metric_dexmachina_provides():
    distances = ("pos_dist", "rot_dist", "arti_dist", "keypoint_dist")
    rew_dict = {key: torch.tensor([0.0, 1.0]) for key in distances}
    rew_dict.update({key: torch.tensor([True, False]) for key in ("success_2cm", "success_5cm")})
    means = flash_dex.TrackingErrors.from_rew_dict(rew_dict).log_means()
    expected_names = {"train/success_2cm", "train/success_5cm"} | {f"env/{key}" for key in distances}
    assert means == {name: 0.5 for name in expected_names}


def test_tracking_errors_leave_out_metrics_dexmachina_does_not_provide():
    means = flash_dex.TrackingErrors.from_rew_dict({"pos_dist": torch.tensor([0.25, 0.75])}).log_means()
    assert means == {"env/pos_dist": 0.5}


def test_curriculum_clock_visits_every_epoch_once_in_order():
    max_epochs = flashsac_env_kwargs({})["max_epochs"]
    num_interaction_steps = flash_dex.REPORTED_INTERACTION_STEPS
    epochs = [
        flash_dex.curriculum_epoch(step, num_interaction_steps, max_epochs)
        for step in range(1, num_interaction_steps + 1)
    ]
    # The reported run size divides evenly, so the last step lands on max_epochs itself.
    assert list(dict.fromkeys(epochs)) == list(range(max_epochs + 1))


def test_env_config_keeps_the_factory_defaults_for_absent_keys():
    config = flash_dex.DexmachinaEnvConfig.from_hydra({"max_epochs": 10, "curriculum": False}, "dexmachina/box")
    assert config.clip == "dexmachina/box"
    assert config.hand is None
    assert config.reward_overrides is None
    assert config.actuate_object is False
    assert config.retarget_name == flash_dex.DEFAULT_RETARGET_NAME


def test_env_config_field_cannot_be_reassigned():
    config = flash_dex.DexmachinaEnvConfig.from_hydra({"max_epochs": 10}, "")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_epochs = 20


def test_env_config_reward_overrides_cannot_be_written():
    config = flash_dex.DexmachinaEnvConfig.from_hydra({"max_epochs": 10, "reward_cfg": {"bc_beta": 1.0}}, "")
    with pytest.raises(TypeError):
        config.reward_overrides["bc_beta"] = 2.0


def test_env_config_rejects_curriculum_without_an_actuated_object():
    with pytest.raises(ValueError) as error:
        flash_dex.DexmachinaEnvConfig.from_hydra(
            {"max_epochs": 10, "curriculum": True, "actuate_object": False}, "dexmachina/box"
        )
    message = str(error.value)
    assert "curriculum" in message
    assert "actuate_object" in message


class FakeCamera:
    def __init__(self, frame: torch.Tensor | None = None, error: Exception | None = None) -> None:
        self.frame = frame
        self.error = error

    def render(self) -> tuple[torch.Tensor | None, None, None, None]:
        if self.error is not None:
            raise self.error
        return self.frame, None, None, None


def test_render_returns_the_camera_frame_as_a_batch() -> None:
    raw, env = make()
    raw._floating_camera = FakeCamera(frame=torch.full((4, 5, 3), 1.0))
    image = env.render()
    assert image.shape == (1, 4, 5, 3)
    assert image.dtype == np.uint8
    assert (image == 255).all()


def test_render_failure_raises_instead_of_returning_a_black_image() -> None:
    raw, env = make()
    raw._floating_camera = FakeCamera(error=RuntimeError("no display"))
    with pytest.raises(RuntimeError, match="num_record_episodes=0") as excinfo:
        env.render()
    assert "no display" in str(excinfo.value.__cause__)


def test_render_without_camera_raises() -> None:
    _, env = make()
    with pytest.raises(RuntimeError, match="record_video"):
        env.render()


def _composed_eval_config() -> DictConfig:
    """The read-only Hydra config of a dexmachina run, composed the way evaluation composes it."""
    config_dir = str(Path(__file__).parents[1] / "algorithms/FlashSAC/configs")
    cfg, _ = flash_dex.compose_eval_config(config_dir, "flashSAC_base", ["env=dexmachina", "env.rl_device=cpu"])
    return cfg


NUM_ENVS = 1024


def benchmark_spec_for(hand, clip, seed, num_envs):
    from dexmachina.envs.benchmark_task import RunChoices, benchmark_spec

    return benchmark_spec(RunChoices(hand=hand, clip=clip, seed=seed, num_envs=num_envs))


def test_dexmachina_spec_is_the_benchmark_task_of_the_run():
    kwargs = flashsac_env_kwargs({"hand": "allegro_hand", "clip": "box-30-230", "reward_cfg": {"force_penalty": 0.1}})
    spec, reward_overrides = flash_dex.dexmachina_spec(8, kwargs, 7, "dexmachina/allegro_hand")
    assert spec == benchmark_spec_for("allegro_hand", "box-30-230", 7, 8)
    assert spec.budget_env_steps == BUDGET
    assert spec.curr_schedule == "fixed"
    assert reward_overrides == {"force_penalty": 0.1}


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"max_epochs": 5000}, "max_epochs"),
        ({"retarget_name": "genesis"}, "retarget_name"),
        ({"curriculum": False}, "actuate_object"),
        ({"reward_cfg": {"contact_rew_weight": 1.0}}, "contact_rew_weight"),
        ({"reward_cfg": {"obj_pos_beta": 1.0}}, "obj_pos_beta"),
        ({"reward_cfg": {"objdex_baseline": True}}, "objdex_baseline"),
        ({"reward_cfg": {"force_penalty": 0.2}}, "force_penalty"),
        ({"reward_cfg": {"task_rew_weight": 0.5}}, "task_rew_weight"),
        ({"reward_cfg": {"bc_wrist_weight": 0.25}}, "bc_wrist_weight"),
    ],
)
def test_dexmachina_spec_refuses_a_run_that_overrides_a_task_field(overrides, field):
    kwargs = {**flashsac_env_kwargs({"hand": "allegro_hand", "clip": "box-30-230"}), **overrides}
    with pytest.raises(ValueError, match="benchmark task fixes") as error:
        flash_dex.dexmachina_spec(8, kwargs, 7, "dexmachina/allegro_hand")
    assert field in str(error.value)


def test_dexmachina_spec_accepts_overrides_that_equal_the_task():
    rewards = {"contact_rew_weight": 3.0, "obj_pos_beta": 10.0, "imi_rew_weight": 0.3, "force_penalty": 0.1}
    kwargs = flashsac_env_kwargs({"hand": "allegro_hand", "clip": "box-30-230", "reward_cfg": rewards})
    spec, _ = flash_dex.dexmachina_spec(8, kwargs, 7, "dexmachina/allegro_hand")
    assert spec == benchmark_spec_for("allegro_hand", "box-30-230", 7, 8)


def test_dexmachina_spec_needs_a_hand():
    with pytest.raises(ValueError, match="env.hand"):
        flash_dex.dexmachina_spec(8, flashsac_env_kwargs({"clip": "box-30-230"}), 7, "dexmachina/allegro_hand")


def env_with_spec(num_envs):
    raw = FakeDexEnv()
    vec: Any = FakeDexVecEnv(raw)
    vec.spec, vec.reward_overrides = benchmark_spec_for("allegro_hand", "box-30-230", 0, num_envs), {}
    return flash_dex.GenesisFlashSACWrapper(vec, rl_device="cpu")


# The smallest multiple of the env count that reaches the budget: what the loop needs to train it in full.
BUDGET_RUN_ENV_STEPS = BUDGET + NUM_ENVS - BUDGET % NUM_ENVS


@pytest.mark.parametrize("extra", [0, NUM_ENVS - 1])
def test_budget_check_accepts_a_run_that_trains_the_budget_to_within_one_interaction_step(extra):
    flash_dex.check_env_step_budget(env_with_spec(NUM_ENVS), BUDGET_RUN_ENV_STEPS + extra)


def test_budget_check_refuses_the_budget_itself_when_it_is_not_a_multiple_of_the_env_count():
    # The loop runs int(budget / num_envs) interaction steps, which trains less than the budget.
    assert BUDGET % NUM_ENVS != 0
    for num_env_steps in (BUDGET, BUDGET + 1):
        with pytest.raises(ValueError, match="trains") as error:
            flash_dex.check_env_step_budget(env_with_spec(NUM_ENVS), num_env_steps)
        assert str(num_env_steps) in str(error.value)


@pytest.mark.parametrize("num_env_steps", [1_000_000, 2 * BUDGET])
def test_budget_check_refuses_a_run_of_another_length(num_env_steps):
    with pytest.raises(ValueError, match="budget"):
        flash_dex.check_env_step_budget(env_with_spec(NUM_ENVS), num_env_steps)


def curriculum_env(env_steps):
    env = env_with_spec(NUM_ENVS)
    env.env.curriculum = SimpleNamespace(env_steps=env_steps)
    return env


def test_curriculum_check_accepts_the_trained_env_steps():
    flash_dex.check_curriculum_saw_the_trained_steps(curriculum_env(10 * NUM_ENVS), 10)


def test_curriculum_check_refuses_a_curriculum_that_counted_other_steps():
    with pytest.raises(ValueError, match="counted") as error:
        flash_dex.check_curriculum_saw_the_trained_steps(curriculum_env(9 * NUM_ENVS), 10)
    assert str(9 * NUM_ENVS) in str(error.value)
    assert str(10 * NUM_ENVS) in str(error.value)


@pytest.mark.parametrize("missing", ["spec", "reward_overrides"])
def test_wrapper_without_a_spec_refuses_to_report_one(missing):
    from dexmachina.envs.spec import EnvSpec

    _, env = make()
    env.vec_env.spec = None if missing == "spec" else EnvSpec()
    env.vec_env.reward_overrides = None if missing == "reward_overrides" else {}
    with pytest.raises(ValueError, match="from_spec"):
        _ = env.env_spec


def test_run_files_round_trip(tmp_path):
    from dexmachina.envs.run_spec import RUN_SPEC_FILENAME, load_run_spec
    from dexmachina.envs.spec import EnvSpec

    raw = FakeDexEnv()
    vec: Any = FakeDexVecEnv(raw)
    vec.spec = dataclasses.replace(EnvSpec(), hand="allegro_hand", use_rl_games=True)
    vec.reward_overrides = {"force_penalty": 0.1}
    env = flash_dex.GenesisFlashSACWrapper(vec, rl_device="cpu")
    cfg = _composed_eval_config()

    flash_dex.save_run_files(cfg, env, tmp_path)

    assert (tmp_path / RUN_SPEC_FILENAME).is_file() and (tmp_path / flash_dex.RUN_CONFIG_FILENAME).is_file()
    saved = load_run_spec(tmp_path)
    assert (saved.spec, dict(saved.reward_overrides)) == (vec.spec, vec.reward_overrides)
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert OmegaConf.load(tmp_path / flash_dex.RUN_CONFIG_FILENAME) == resolved

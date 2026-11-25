"""train.py's loop on a fake env: what it logs, stores and saves."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeDexEnv, FakeDexVecEnv

create_agent = pytest.importorskip("flash_rl.agents").create_agent
flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")
GlobalHydra = pytest.importorskip("hydra.core.global_hydra").GlobalHydra

FLASHSAC_DIR = Path(__file__).parents[1] / "algorithms/FlashSAC"
NUM_ENVS = 2
NUM_INTERACTION_STEPS = 12
EPISODE_INFO = {"episode_return": 1.5}
TINY_RUN_OVERRIDES = [
    "env=dexmachina",
    "env.rl_device=cpu",
    "logger_type=tensorboard",
    f"num_env_steps={NUM_ENVS * NUM_INTERACTION_STEPS}",
    f"num_train_envs={NUM_ENVS}",
    "num_eval_episodes=1",
    "agent.device_type=cpu",
    "agent.buffer_device_type=cpu",
    "agent.use_compile=false",
    "agent.use_amp=false",
    "agent.buffer_max_length=1000",
    "agent.buffer_min_length=4",
    "agent.sample_batch_size=4",
    "agent.actor_hidden_dim=8",
    "agent.critic_hidden_dim=8",
    "evaluation_per_interaction_step=null",
    "metrics_per_interaction_step=null",
    "recording_per_interaction_step=null",
    "logging_per_interaction_step=4",
    "save_checkpoint_per_interaction_step=5",
]


class RecordingLogger:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.logged_steps: list[int] = []
        self.transitions: list[dict[str, Any]] = []

    def update_metric(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def log_metric(self, step: int) -> None:
        self.logged_steps.append(step)

    def reset(self) -> None:
        pass


class EpisodeInfoVecEnv(FakeDexVecEnv):
    def step(self, actions):
        observations, reward, terminated, truncated, info = super().step(actions)
        info["log"] = dict(EPISODE_INFO)
        return observations, reward, terminated, truncated, info


class TrainRun:
    def __init__(self, module: Any, logger: RecordingLogger, directory: Path) -> None:
        self.module = module
        self.logger = logger
        self.directory = directory

    def run(self, extra_overrides: list[str]) -> None:
        GlobalHydra.instance().clear()
        overrides = [*TINY_RUN_OVERRIDES, *extra_overrides]
        args = self.module.argparse.Namespace(config_path="./configs", config_name="flashSAC_base", overrides=overrides)
        self.module.run(args)


@pytest.fixture
def train_run(monkeypatch, tmp_path, request):
    request.addfinalizer(GlobalHydra.instance().clear)  # train.py leaves Hydra initialised
    monkeypatch.setattr("torch._dynamo.config.disable", True)
    spec = importlib.util.spec_from_file_location("flashsac_train", FLASHSAC_DIR / "train.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    logger = RecordingLogger()

    def create_fake_envs(**_):
        env = flash_dex.GenesisFlashSACWrapper(
            EpisodeInfoVecEnv(FakeDexEnv(num_envs=NUM_ENVS, episode_length=3)), rl_device="cpu"
        )
        return env, env, env

    def create_recording_agent(**kwargs):
        agent = create_agent(**kwargs)
        process_transition = agent.process_transition

        def record(transition):
            logger.transitions.append({key: value.clone() for key, value in transition.items()})
            process_transition(transition)

        agent.process_transition = record
        return agent

    monkeypatch.setattr(module, "create_envs", create_fake_envs)
    monkeypatch.setattr(module, "create_agent", create_recording_agent)
    monkeypatch.setattr(module, "create_logger", lambda cfg: logger)
    monkeypatch.setattr(module, "UniversalLogger", None)
    monkeypatch.setattr(module, "evaluate", lambda *args: {})
    monkeypatch.setattr(module, "record_video", lambda *args: {})
    for check in ("check_env_step_budget", "refuse_resume_with_fixed_schedule", "save_run_files"):
        monkeypatch.setattr(module, check, lambda *args: None)
    monkeypatch.setattr(module, "check_curriculum_saw_the_trained_steps", lambda *args: None)
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "train.py")])
    return TrainRun(module, logger, tmp_path)


def test_episode_info_enters_the_averages_once_per_step(train_run):
    train_run.run([])
    logged = [update for update in train_run.logger.updates if EPISODE_INFO.keys() <= update.keys()]
    assert len(logged) == NUM_INTERACTION_STEPS


def test_the_training_envs_success_rate_is_not_logged_under_an_eval_key(train_run):
    train_run.run([])
    logged_keys = {key for update in train_run.logger.updates for key in update}
    assert not {key for key in logged_keys if key.startswith("eval/")}  # evaluate() is faked to log nothing
    assert {"train/success_2cm", "train/success_5cm"} <= logged_keys


def saved_steps(train_run) -> list[str]:
    run_dirs = list((train_run.directory / "models").glob("*/*/*/*/seed*"))
    assert len(run_dirs) == 1
    return sorted(path.name for path in run_dirs[0].glob("step*"))


def test_the_final_step_is_saved_although_it_is_not_a_multiple_of_the_interval(train_run):
    train_run.run([])
    assert saved_steps(train_run) == sorted(["step5", "step10", f"step{NUM_INTERACTION_STEPS}"])


def test_a_final_step_on_the_interval_is_saved_once(train_run):
    train_run.run([f"save_checkpoint_per_interaction_step={NUM_INTERACTION_STEPS}"])
    assert saved_steps(train_run) == [f"step{NUM_INTERACTION_STEPS}"]


def test_the_final_step_is_saved_without_a_checkpoint_interval(train_run):
    train_run.run(["save_checkpoint_per_interaction_step=null"])
    assert saved_steps(train_run) == [f"step{NUM_INTERACTION_STEPS}"]


def test_a_finished_episodes_next_observation_is_the_one_before_the_reset(train_run):
    train_run.run([])
    transitions = train_run.logger.transitions
    assert len(transitions) == NUM_INTERACTION_STEPS
    for transition in transitions:
        # FakeDexEnv: every dimension of env i's observation is t + 100 * i, and t restarts at 0 after a reset.
        expected = transition["observation"] + 1
        assert (transition["next_observation"] == expected).all()
    assert any(transition["terminated"].any() for transition in transitions)

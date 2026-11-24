"""dexhand-ppo's DexMachina training adapter against fake envs (no simulator)."""
from dataclasses import replace

import pytest
import torch

from fakes import FakeDexEnv, FakeDexVecEnv

dex = pytest.importorskip("dexhand_ppo.envs.dexmachina", reason="needs the dexhand-ppo extra")
from dexhand_ppo.config import Config  # noqa: E402
from dexmachina.envs.benchmark_task import BENCHMARK_TASK  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

NUM_ENVS, EPISODE_LENGTH, NUM_STEPS, EMAX = 2, 3, 4, 100
ITERATIONS = 10
ZERO_EPOCH = 90
SPEC = replace(BENCHMARK_TASK, num_envs=NUM_ENVS, max_epochs=EMAX)


def _cfg():
    return OmegaConf.merge(OmegaConf.structured(Config), {
        "device": "cpu", "total_steps": ITERATIONS * NUM_ENVS * NUM_STEPS,
        "env": {"num_envs": NUM_ENVS}, "ppo": {"num_steps": NUM_STEPS},
    })


class StubCurriculum:
    decay_terms = ("kp", "kv", "fr")
    zero_epoch = ZERO_EPOCH

    def __init__(self):
        self.curr_gains = {"kp": 80.0, "kv": 5.0, "fr": 50.0}
        self.curr_gains_lower = dict(self.curr_gains)
        self.gain_history = []
        self.last_decay_epoch = 0
        self.applied = None

    def reset_object_gains(self):
        self.applied = dict(self.curr_gains)


class FakeTrainEnv(FakeDexEnv):
    """FakeDexEnv plus the BaseEnv attributes the adapter reads."""

    def __init__(self):
        super().__init__(num_envs=NUM_ENVS, episode_length=EPISODE_LENGTH)
        self.num_obs = self.obs_dim
        self.use_curriculum = True
        self.curriculum = StubCurriculum()
        self.curriculum_calls = []
        self.next_reward = torch.ones(NUM_ENVS)

    def step(self, actions):
        obs, _, terminated, time_outs, extras = super().step(actions)
        self.rew_dict["task_rew"] = torch.full((NUM_ENVS,), 0.5)
        return obs, self.next_reward.clone(), terminated, time_outs, extras

    def set_curriculum(self, epoch):
        """Scripted schedule: reductions at 10 and 20, a dialback at 30, hard stop after ZERO_EPOCH."""
        self.curriculum_calls.append(epoch)
        gains = self.curriculum.curr_gains
        if epoch in (10, 20):
            gains["kp"] *= 0.9
            gains["kv"] *= 0.9
        elif epoch == 30:
            gains["kp"] /= 0.9
        elif epoch > ZERO_EPOCH and gains["kp"] > 0:
            for key in gains:
                gains[key] = 0.0
        return False


class TruncatingVecEnv(FakeDexVecEnv):
    """Reports every episode end as truncated instead of terminated (a task with time limits, unlike the benchmark)."""

    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        return obs, reward, truncated, terminated, info


def _train_env(vec_env_class=FakeDexVecEnv):
    vec_env = vec_env_class(FakeTrainEnv())
    vec_env.spec = SPEC
    return dex.DexMachinaTrainEnv(_cfg(), vec_env=vec_env)


@pytest.fixture
def adapter():
    return _train_env()


def test_finished_episodes_are_reported(adapter):
    adapter.reset()
    for _ in range(EPISODE_LENGTH):
        _, _, terminated, truncated, info = adapter.step(torch.zeros(NUM_ENVS, 2))
    assert terminated.all() and not truncated.any()
    episode = info["episode"]
    assert episode["return"].tolist() == [3.0, 3.0] and episode["length"].tolist() == [3.0, 3.0]
    assert episode["task_rew"].tolist() == [0.5, 0.5]


def test_a_truncation_of_the_vec_env_reaches_the_trainer_with_its_final_obs():
    adapter = _train_env(TruncatingVecEnv)
    adapter.reset()
    for _ in range(EPISODE_LENGTH):
        obs, _, terminated, truncated, info = adapter.step(torch.zeros(NUM_ENVS, 2))
    assert truncated.all() and not terminated.any()
    assert info["final_obs_mask"].all()
    # obs of env i after t steps is t + 100 i (FakeDexEnv): the final obs is the pre-reset one, obs the reset one
    assert info["final_obs"][:, 0].tolist() == [3.0, 103.0] and obs[:, 0].tolist() == [0.0, 100.0]


def test_nan_rewards_are_replaced_and_counted(adapter):
    adapter.reset()
    adapter.env.next_reward = torch.tensor([float("nan"), 1.0])
    _, rew, _, _, _ = adapter.step(torch.zeros(NUM_ENVS, 2))
    assert rew.tolist() == [0.0, 1.0]
    assert adapter.on_iteration(0)["train/nan_rewards"] == 1


def test_curriculum_epochs_reductions_dialbacks_and_hard_stop(adapter):
    steps_per_iteration = NUM_ENVS * NUM_STEPS
    metrics = {}
    for iteration in range(ITERATIONS):
        metrics = adapter.on_iteration(iteration * steps_per_iteration)
        adapter.on_iteration(iteration * steps_per_iteration)  # same epoch again: no extra call
    assert adapter.env.curriculum_calls == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    adapter.on_iteration(ITERATIONS * steps_per_iteration)     # epoch 100 > zero_epoch: hard stop
    metrics = adapter.on_iteration(ITERATIONS * steps_per_iteration)
    assert metrics["curriculum/num_reductions"] == 2
    assert metrics["curriculum/num_dialbacks"] == 1
    assert metrics["curriculum/hard_stop_applied"] == 1.0
    assert metrics["env/kp"] == 0.0 and metrics["curriculum/epoch"] == EMAX


def test_curriculum_state_round_trip(adapter):
    for epoch_step in (0, 10 * NUM_ENVS * NUM_STEPS // 10, 2 * NUM_ENVS * NUM_STEPS):
        adapter.on_iteration(epoch_step)
    state = adapter.state_dict()
    restored = _train_env()
    restored.load_state_dict(state)
    assert restored.num_reductions == adapter.num_reductions and restored.current_epoch == adapter.current_epoch
    assert restored.env.curriculum.curr_gains == adapter.env.curriculum.curr_gains
    assert restored.env.curriculum.applied == adapter.env.curriculum.curr_gains   # re-applied to the object


def test_score_metrics_logs_every_score_under_its_field_name():
    from dexmachina.envs.evaluation import EvalScore

    scored = EvalScore(auc_add=0.5, auc_add_of_mean=0.4, auc_add_part={"top": 0.6, "bottom": 0.4}, mean_add=0.03,
                       success_2cm=0.2, success_5cm=0.3, nan_rate=0.0, drop_rate=1.0)
    assert dex.score_metrics("eval/floor/", scored) == {
        "eval/floor/auc_add": 0.5, "eval/floor/auc_add_of_mean": 0.4, "eval/floor/mean_add": 0.03,
        "eval/floor/success_2cm": 0.2, "eval/floor/success_5cm": 0.3, "eval/floor/nan_rate": 0.0,
        "eval/floor/drop_rate": 1.0, "eval/floor/auc_add_part/top": 0.6, "eval/floor/auc_add_part/bottom": 0.4,
    }
    assert dex.DexMachinaEvaluator.selection_metric in dex.score_metrics("eval/", scored)

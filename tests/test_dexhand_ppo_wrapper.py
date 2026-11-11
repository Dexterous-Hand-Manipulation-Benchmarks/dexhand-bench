"""dexhand-ppo's DexMachina adapter and evaluator against fake envs (no simulator)."""
import pytest
import torch

from fakes import FakeDexEnv

dex = pytest.importorskip("dexhand_ppo.envs.dexmachina", reason="needs the dexhand-ppo extra")
from omegaconf import OmegaConf  # noqa: E402
from dexhand_ppo.config import Config  # noqa: E402

NUM_ENVS, EPISODE_LENGTH, NUM_STEPS, EMAX = 2, 3, 4, 100
ITERATIONS = 10
ZERO_EPOCH = 90


def _cfg():
    return OmegaConf.merge(OmegaConf.structured(Config), {
        "device": "cpu", "total_steps": ITERATIONS * NUM_ENVS * NUM_STEPS,
        "env": {"num_envs": NUM_ENVS, "dexmachina": {"curriculum_max_epochs": EMAX}},
        "ppo": {"num_steps": NUM_STEPS},
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
    """FakeDexEnv plus the BaseEnv attributes the adapter reads, and persistent in-place buffers like BaseEnv."""

    def __init__(self):
        super().__init__(num_envs=NUM_ENVS, episode_length=EPISODE_LENGTH)
        self.num_obs = self.obs_dim
        self.use_curriculum = True
        self.curriculum = StubCurriculum()
        self.curriculum_calls = []
        self.rew_buf = torch.zeros(NUM_ENVS)
        self.reset_buf = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.next_reward = torch.ones(NUM_ENVS)

    def step(self, actions):
        obs, _, terminated, _, extras = super().step(actions)
        self.rew_buf.copy_(self.next_reward)      # in place, as BaseEnv does
        self.reset_buf.copy_(terminated)
        self.rew_dict["task_rew"] = torch.full((NUM_ENVS,), 0.5)
        return obs, self.rew_buf, self.reset_buf, self.reset_buf, extras

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


@pytest.fixture
def adapter():
    return dex.DexMachinaVecEnv(_cfg(), env=FakeTrainEnv())


def test_rewards_and_flags_are_cloned(adapter):
    adapter.reset()
    _, rew, done, _, _ = adapter.step(torch.zeros(NUM_ENVS, 2))
    adapter.env.next_reward = torch.full((NUM_ENVS,), 7.0)
    adapter.step(torch.zeros(NUM_ENVS, 2))
    assert rew.tolist() == [1.0, 1.0]           # not overwritten by the next step's in-place update
    assert done.tolist() == [False, False]


def test_every_episode_end_is_terminal_and_reported(adapter):
    adapter.reset()
    for t in range(EPISODE_LENGTH):
        _, _, terminated, truncated, info = adapter.step(torch.zeros(NUM_ENVS, 2))
        assert not truncated.any()
    assert terminated.all()
    episode = info["episode"]
    assert episode["return"].tolist() == [3.0, 3.0] and episode["length"].tolist() == [3.0, 3.0]
    assert episode["task_rew"].tolist() == [0.5, 0.5]


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
    restored = dex.DexMachinaVecEnv(_cfg(), env=FakeTrainEnv())
    restored.load_state_dict(state)
    assert restored.num_reductions == adapter.num_reductions and restored.current_epoch == adapter.current_epoch
    assert restored.env.curriculum.curr_gains == adapter.env.curriculum.curr_gains
    assert restored.env.curriculum.applied == adapter.env.curriculum.curr_gains   # re-applied to the object


class FakeEvalEnv:
    """Eval env with 2 real envs and dexmachina's dummy last env (terrible ADD, must be excluded)."""

    num_envs, max_episode_length, use_curriculum = 3, 4, False

    def __init__(self):
        self.rew_dict = {}

    def reset(self):
        return {"policy": torch.zeros(self.num_envs, 3)}, {}

    def step(self, actions):
        add_top = torch.tensor([0.005, 0.03, 9.0])
        add_bottom = torch.tensor([0.005, 0.01, 9.0])
        log = {"add_top": add_top, "add_bottom": add_bottom, "add": (add_top + add_bottom) / 2}
        self.rew_dict = {"success_2cm": torch.tensor([True, False, False]),
                         "success_5cm": torch.tensor([True, True, False]), "pos_dist": torch.zeros(3)}
        return {"policy": torch.zeros(self.num_envs, 3)}, torch.zeros(3), None, None, {"log": log}


def test_evaluator_excludes_dummy_env_and_averages_auc_per_part():
    from dexmachina.envs.add_metric import compute_auc

    evaluator = dex.DexMachinaEvaluator(_cfg(), env=FakeEvalEnv())
    metrics = evaluator.evaluate(lambda obs: torch.zeros(obs.shape[0], 2))
    assert metrics["eval/num_envs"] == 2
    expected_auc = (compute_auc([0.005, 0.03] * 4) + compute_auc([0.005, 0.01] * 4)) / 2
    assert metrics["eval/auc_add"] == pytest.approx(expected_auc)
    assert metrics["eval/add_success_2cm"] == pytest.approx((0.5 + 1.0) / 2)   # top: 1 of 2, bottom: 2 of 2
    assert metrics["eval/success_2cm"] == pytest.approx(0.5)
    assert metrics["eval/add"] < 0.1                                           # the dummy's 9 m is not included

"""FlashSAC evaluates on its own env, so that evaluation never moves the training env's curriculum clock.

The fixed schedule's clock lives in the env, so a run that loads a checkpoint would restart it at 0: refused.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fakes import FakeDexEnv, FakeDexVecEnv

flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")
flash_envs = pytest.importorskip("flash_rl.envs")
evaluation = pytest.importorskip("flash_rl.evaluation")

TRAIN_ENVS = 4
EPISODES = 2
ACTION_DIM = 2


class ZeroActionAgent:
    def sample_actions(self, interaction_step: int, prev_transition: dict[str, Any], training: bool) -> np.ndarray:
        return np.zeros((len(prev_transition["next_observation"]), ACTION_DIM))


class SpecVecEnv(FakeDexVecEnv):
    """A fake vec env that knows its spec, as DexMachinaVecEnv.from_spec's does."""

    def __init__(self, env: FakeDexEnv, spec: Any) -> None:
        super().__init__(env)
        self.spec = spec
        self.reward_overrides: dict[str, Any] = {}


def _train_vec_env(spec: Any = object()) -> SpecVecEnv:
    return SpecVecEnv(FakeDexEnv(num_envs=TRAIN_ENVS), spec)


def test_evaluating_does_not_step_the_training_env(monkeypatch):
    monkeypatch.setattr("genesis.init", lambda **_: None)
    monkeypatch.setattr(flash_dex, "make_dexmachina_env", lambda *args: _train_vec_env())
    monkeypatch.setattr("dexmachina.envs.factory.make_eval_env", lambda spec, device: FakeDexEnv(num_envs=1))
    monkeypatch.setattr("dexmachina.envs.vec_env.DexMachinaVecEnv", FakeDexVecEnv)

    train_env, eval_env, record_env = flash_envs.create_envs(
        env_type="dexmachina", seed=0, env_name="", num_train_envs=TRAIN_ENVS, num_eval_envs=None,
        num_record_envs=None, rescale_action=None, max_episode_steps=None, rl_device="cpu",
    )
    train_env.reset()
    evaluation.evaluate(ZeroActionAgent(), eval_env, EPISODES, "dexmachina")

    assert train_env.env.received_actions is None
    assert eval_env.env.received_actions is not None
    assert record_env is not train_env


@pytest.mark.parametrize(("agent_load_path", "buffer_load_path"), [("agent", None), (None, "buffer")])
def test_loading_into_a_run_with_the_fixed_schedule_is_refused(agent_load_path, buffer_load_path):
    vec_env = _train_vec_env(SimpleNamespace(curr_schedule="fixed"))
    env = flash_dex.GenesisFlashSACWrapper(vec_env, rl_device="cpu")
    with pytest.raises(ValueError, match="which a new run starts at 0"):
        flash_dex.refuse_resume_with_fixed_schedule(env, agent_load_path, buffer_load_path)
    flash_dex.refuse_resume_with_fixed_schedule(env, None, None)
    vec_env.spec = SimpleNamespace(curr_schedule="uniform")
    flash_dex.refuse_resume_with_fixed_schedule(env, agent_load_path, buffer_load_path)

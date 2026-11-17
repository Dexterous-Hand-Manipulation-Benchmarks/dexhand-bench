"""FlashSAC's training seed must decide the agent's initial weights and warm-up actions.

DexMachina's curriculum reseeds the global RNGs with a fixed 42 while the env is built, so train.py has to seed
again after building the envs; the fake env here does the same reseed.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from fakes import FakeDexEnv, FakeDexVecEnv

flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")
curriculum = pytest.importorskip("dexmachina.envs.curriculum")
GlobalHydra = pytest.importorskip("hydra.core.global_hydra").GlobalHydra

FLASHSAC_DIR = Path(__file__).parents[1] / "algorithms/FlashSAC"
CURRICULUM_SEED = curriculum.get_curriculum_cfg()["seed"]
CPU_AGENT_OVERRIDES = [
    "env=dexmachina",
    "env.rl_device=cpu",
    "agent.device_type=cpu",
    "agent.buffer_device_type=cpu",
    "agent.use_compile=false",
    "agent.use_amp=false",
    "agent.buffer_max_length=1000",
]


class CurriculumReseedingEnv(FakeDexEnv):
    def __init__(self) -> None:
        super().__init__()
        np.random.seed(CURRICULUM_SEED)
        torch.manual_seed(CURRICULUM_SEED)


@pytest.fixture
def train(monkeypatch):
    # The agent compiles helper functions even with use_compile=false; the test only needs the initial weights.
    monkeypatch.setattr("torch._dynamo.config.disable", True)
    spec = importlib.util.spec_from_file_location("flashsac_train", FLASHSAC_DIR / "train.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def create_fake_envs(**_):
        env = flash_dex.GenesisFlashSACWrapper(FakeDexVecEnv(CurriculumReseedingEnv()), rl_device="cpu")
        return env, env, env

    monkeypatch.setattr(module, "create_envs", create_fake_envs)
    return module


def initial_weights_and_warmup_action(train, seed: int) -> tuple[torch.Tensor, np.ndarray]:
    cfg = train.compose_config("./configs", "flashSAC_base", [*CPU_AGENT_OVERRIDES, f"seed={seed}"])
    GlobalHydra.instance().clear()
    train_env, _, _, agent = train.create_envs_and_agent(cfg)
    weights = torch.cat([p.detach().flatten() for p in agent._actor.network.parameters()])
    return weights, train_env.action_space.sample()


def test_seeds_give_different_initial_weights_and_warmup_actions(train):
    weights_0, action_0 = initial_weights_and_warmup_action(train, 0)
    weights_1, action_1 = initial_weights_and_warmup_action(train, 1)
    assert not torch.equal(weights_0, weights_1)
    assert not np.array_equal(action_0, action_1)


def test_same_seed_gives_the_same_initial_weights_and_warmup_actions(train):
    weights_a, action_a = initial_weights_and_warmup_action(train, 3)
    weights_b, action_b = initial_weights_and_warmup_action(train, 3)
    assert torch.equal(weights_a, weights_b)
    np.testing.assert_array_equal(action_a, action_b)

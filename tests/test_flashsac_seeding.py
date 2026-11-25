"""FlashSAC's training seed must decide the agent's initial weights and warm-up actions, and the parallel envs must
not share their random actions or exploration noise.

Resets while the env is built still consume the global RNGs, so train.py seeds again after building the envs; the
fake env here overwrites the global RNGs with a fixed seed, the worst case of that.
"""

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from fakes import FakeDexEnv, FakeDexVecEnv

flash_envs = pytest.importorskip("flash_rl.envs", reason="needs the flashsac extra")
flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")
curriculum = pytest.importorskip("dexmachina.envs.curriculum")
GlobalHydra = pytest.importorskip("hydra.core.global_hydra").GlobalHydra

FLASHSAC_DIR = Path(__file__).parents[1] / "algorithms/FlashSAC"
BUILD_RESEED = curriculum.get_curriculum_cfg()["seed"]
CPU_AGENT_OVERRIDES = [
    "env=dexmachina",
    "env.rl_device=cpu",
    "agent.device_type=cpu",
    "agent.buffer_device_type=cpu",
    "agent.use_compile=false",
    "agent.use_amp=false",
    "agent.buffer_max_length=1000",
]


class ReseedingEnv(FakeDexEnv):
    def __init__(self) -> None:
        super().__init__()
        np.random.seed(BUILD_RESEED)
        torch.manual_seed(BUILD_RESEED)


@pytest.fixture
def train(monkeypatch):
    # The agent compiles helper functions even with use_compile=false; the test only needs the initial weights.
    monkeypatch.setattr("torch._dynamo.config.disable", True)
    spec = importlib.util.spec_from_file_location("flashsac_train", FLASHSAC_DIR / "train.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def create_fake_envs(**_):
        env = flash_dex.GenesisFlashSACWrapper(FakeDexVecEnv(ReseedingEnv()), rl_device="cpu")
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


# Enough steps that two envs with independent Zeta repeat lengths resample on the same steps only by vanishing chance.
NOISE_STEPS = 50


def env_and_agent(train) -> tuple[Any, Any]:
    cfg = train.compose_config("./configs", "flashSAC_base", [*CPU_AGENT_OVERRIDES, "seed=0"])
    GlobalHydra.instance().clear()
    train_env, _, _, agent = train.create_envs_and_agent(cfg)
    return train_env, agent


def same_observation_for_all_envs(train_env) -> dict[str, np.ndarray]:
    return {"next_observation": np.zeros((train_env.num_envs, *train_env.observation_space.shape))}


def test_warmup_gives_each_env_its_own_action(train):
    train_env, _ = env_and_agent(train)
    actions = train.warmup_actions(train_env.action_space, train_env.num_envs)
    assert actions.shape == (train_env.num_envs, *train_env.action_space.shape)
    assert not np.array_equal(actions[0], actions[1])


def test_envs_with_the_same_observation_get_different_actions(train):
    train_env, agent = env_and_agent(train)
    actions = agent.sample_actions(0, prev_transition=same_observation_for_all_envs(train_env), training=True)
    assert not np.array_equal(actions[0], actions[1])


def test_envs_resample_their_exploration_noise_on_different_steps(train):
    train_env, agent = env_and_agent(train)
    transition = same_observation_for_all_envs(train_env)
    agent.sample_actions(0, prev_transition=transition, training=True)
    resampled = []
    for step in range(1, NOISE_STEPS):
        # The actions mix noise with the actor's output; only the cached noise shows when an env resampled.
        previous_noise = agent._cached_noise.clone()
        agent.sample_actions(step, prev_transition=transition, training=True)
        resampled.append((agent._cached_noise != previous_noise).any(dim=-1).numpy())
    resampled_per_env = np.stack(resampled, axis=1)
    assert not np.array_equal(resampled_per_env[0], resampled_per_env[1])


def test_the_dexmachina_env_is_built_with_the_runs_seed(train: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    seeds: list[int] = []

    def fake_make_dexmachina_env(
        num_envs: int, device: str, hydra_kwargs: Any, seed: int, env_name: str
    ) -> Any:
        seeds.append(seed)
        return FakeDexVecEnv(FakeDexEnv())

    monkeypatch.setattr("genesis.init", lambda **_: None)
    monkeypatch.setattr(flash_dex, "make_dexmachina_env", fake_make_dexmachina_env)
    monkeypatch.setattr(flash_dex, "make_dexmachina_eval_env", lambda train_env, device: train_env)
    cfg = train.compose_config("./configs", "flashSAC_base", [*CPU_AGENT_OVERRIDES, "seed=1234"])
    GlobalHydra.instance().clear()
    flash_envs.create_envs(**cfg.env)
    assert seeds == [1234]

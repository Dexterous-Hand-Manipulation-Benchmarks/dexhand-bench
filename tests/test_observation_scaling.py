"""Reading the observations must not change them: calling get_observations twice gives the same result."""

from dataclasses import replace
from typing import Any

import pytest
import torch

pytest.importorskip("dexmachina.envs.constructors", reason="needs genesis and dexmachina")
import dexmachina.envs.factory as factory  # noqa: E402  (after the skip: needs dexmachina)
from dexmachina.envs.spec import EnvSpec  # noqa: E402

CPU = "cpu"
CLIP = "box-30-230-s01-01"


@pytest.mark.sim
def test_get_observations_twice_gives_the_same_observation(genesis_cpu, reseed, isolated_hand_cfgs) -> None:
    spec = replace(
        EnvSpec(), num_envs=2, seed=0, hand="allegro_hand", clip=CLIP, retarget_name="para",
        actuate_object=True, use_rl_games=True,
    )  # fmt: skip
    reseed(spec.seed)
    env: Any = factory.make_env(spec, CPU, {})
    env.reset()
    env.step(torch.zeros(env.num_envs, env.action_dim))
    robot_velocity = env.robots["left"].dof_vel.clone()
    object_velocity = env.object.root_lin_vel.clone()

    first = env.get_observations()["policy"].clone()
    second = env.get_observations()["policy"]

    assert torch.equal(first, second)
    assert torch.equal(env.robots["left"].dof_vel, robot_velocity)
    assert torch.equal(env.object.root_lin_vel, object_velocity)

"""The benchmark task is one EnvSpec; every EnvSpec field is either a run choice or fixed by the task."""

import json
import re
from dataclasses import fields, replace
from typing import Any

import pytest

from conftest import GOLDEN_DIR

pytest.importorskip("dexmachina.envs.constructors", reason="needs genesis and dexmachina")
import dexmachina.envs.factory as factory  # noqa: E402
from dexmachina.envs.benchmark_task import (  # noqa: E402  (after the skip: needs dexmachina)
    BENCHMARK_CLIPS,
    BENCHMARK_TASK,
    RUN_FIELDS,
    RunChoices,
    benchmark_spec,
)
from dexmachina.envs.spec import EnvSpec  # noqa: E402

# Values the FastTD3, REPPO and dexhand-ppo configs of the lab runs share, where they differ from EnvSpec's defaults.
DIFFERS_FROM_DEFAULT: dict[str, Any] = {
    "action_mode": "hybrid",
    "hybrid_scales": (0.1, 1.0),
    "group_collisions": True,
    "observe_tip_dist": True,
    "observe_contact_force": True,
    "early_reset_threshold": 0.6,
    "actuate_object": True,
    "retarget_name": "para",
    "use_rl_games": True,
    "imi_rew_weight": 0.3,
    "contact_rew_weight": 3.0,
    "bc_rew_weight": 0.3,
    "action_penalty": 0.01,
    "use_retarget_contact": True,
    "kp_init": 80.0,
    "kv_init": 5.0,
    "upper_ratios": (0.9, 0.9, 1.0),
    "lower_ratios": (0.8, 0.8, 1.0),
    "wait_epochs": 100,
    "skip_grad": True,
    "dialback_ep_len": 80,
    "curr_rew_thres": (0.6, 0.01, 0.01, 0.01),
    "max_epochs": 10000,
    "rand_init_ratio": 0.02,
}
CHOICES = RunChoices(
    hand="xhand", clip=BENCHMARK_CLIPS[1], seed=7, num_envs=16, vis=True, record_video=True, record_interval=100
)
CPU = "cpu"
ENV_SPEC_FIELDS = {f.name for f in fields(EnvSpec)}


def test_run_fields_name_only_env_spec_fields() -> None:
    assert set(RUN_FIELDS) <= ENV_SPEC_FIELDS


def test_every_env_spec_field_is_a_run_field_or_a_known_task_field() -> None:
    known_fields = set(json.loads((GOLDEN_DIR / "env_argparser.json").read_text())["order"])
    unreviewed = ENV_SPEC_FIELDS - known_fields
    assert not unreviewed, (
        f"EnvSpec fields {sorted(unreviewed)} are new: a new EnvSpec field becomes a task field at its default "
        "unless added to RunChoices. Classify it, then re-record the argparser fixture."
    )
    assert known_fields - ENV_SPEC_FIELDS == set(), "EnvSpec fields were removed; re-record the argparser fixture."


def test_task_fields_that_differ_from_the_defaults_are_the_benchmark_values() -> None:
    task_fields = ENV_SPEC_FIELDS - set(RUN_FIELDS)
    differs = {
        name: getattr(BENCHMARK_TASK, name)
        for name in task_fields
        if getattr(BENCHMARK_TASK, name) != getattr(EnvSpec(), name)
    }
    assert differs == DIFFERS_FROM_DEFAULT


def test_benchmark_spec_sets_the_run_fields() -> None:
    spec = benchmark_spec(CHOICES)
    for name in RUN_FIELDS:
        assert getattr(spec, name) == getattr(CHOICES, name), name


def test_benchmark_spec_leaves_the_task_fields_alone() -> None:
    spec = benchmark_spec(CHOICES)
    for name in ENV_SPEC_FIELDS - set(RUN_FIELDS):
        assert getattr(spec, name) == getattr(BENCHMARK_TASK, name), name


def test_short_clip_form_resolves_to_the_benchmark_clip() -> None:
    assert BENCHMARK_CLIPS[0] == "box-30-230-s01-01"
    spec = benchmark_spec(replace(CHOICES, clip="box-30-230"))
    assert spec.clip == BENCHMARK_CLIPS[0]


def test_unknown_clip_is_refused_with_the_allowed_clips() -> None:
    with pytest.raises(ValueError, match=f"not a benchmark clip.*{re.escape(BENCHMARK_CLIPS[0])}"):
        benchmark_spec(replace(CHOICES, clip="box-40-200-s01-u01"))


@pytest.mark.sim
def test_make_env_builds_the_benchmark_task(genesis_cpu: None, isolated_hand_cfgs: None) -> None:
    spec = benchmark_spec(RunChoices(hand="allegro_hand", clip=BENCHMARK_CLIPS[0], seed=0, num_envs=2))
    env = factory.make_env(spec, CPU, {})
    assert env.num_envs == spec.num_envs
    assert env.use_curriculum

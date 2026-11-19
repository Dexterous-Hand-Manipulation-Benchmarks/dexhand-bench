"""DexMachinaVecEnv reports episode ends by cause, keeps BaseEnv's reset set, and returns copies.

The unit tests drive a fake env that runs dexmachina's real `BaseEnv._get_dones` and mirrors the flow of
`BaseEnv.step` (dones, rewards, resets, observation after the reset). The simulator test compares the wrapper with
the FlashSAC golden rollout.
"""

import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest
import torch

pytest.importorskip("dexmachina.envs.base_env", reason="needs genesis and dexmachina")
import dexmachina.envs.vec_env as vec_env_module  # noqa: E402
from dexmachina.envs.base_env import BaseEnv  # noqa: E402  (after the skip: needs dexmachina)
from dexmachina.envs.maniptrans_curr import ManipTransCurriculum  # noqa: E402
from dexmachina.envs.spec import EnvSpec  # noqa: E402
from dexmachina.envs.vec_env import AUTORESET_SAME_STEP, DexMachinaVecEnv  # noqa: E402
from dexmachina.envs.vec_env_check import VecEnvContractError, check_vec_env  # noqa: E402

# check_vec_env reads the spaces, which need the optional gymnasium (not in the dexhand-ppo extra).
needs_gymnasium = pytest.mark.skipif(importlib.util.find_spec("gymnasium") is None, reason="gymnasium is optional")

NUM_ENVS = 2
OBS_DIM = 3
ACTION_DIM = 2
OBS_CLIP = 5.0
ENV_OFFSET = 1.0
TABLE_HEIGHT = 0.6
ABOVE_TABLE = 1.0
BELOW_TABLE = 0.0
EARLY_RESET_THRESHOLD = 0.5
EARLY_RESET_INTERVAL = 2
DEMO_LENGTH = 3
NO_CHUNK = -1
AUX_CHECK_STEPS = 20


class _Object:
    def __init__(self, num_envs: int) -> None:
        self.root_pos = torch.full((num_envs, 3), ABOVE_TABLE)


class FakeBaseEnv:
    """A BaseEnv stand-in with the real `_get_dones` and the step flow of `BaseEnv.step`.

    The observation of env i is its episode step plus i in every dimension, so the observation before a reset
    differs from the one after it. Like BaseEnv, it returns its own buffers and updates them in place.
    """

    _get_dones = BaseEnv._get_dones
    _record_early_reset = BaseEnv._record_early_reset

    def __init__(
        self,
        max_episode_length: int = DEMO_LENGTH,
        chunk_ep_length: int = NO_CHUNK,
        early_reset_threshold: float = 0.0,
        rewards: Sequence[float] = (0.0,) * NUM_ENVS,
    ) -> None:
        self.num_envs = NUM_ENVS
        self.device = "cpu"
        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM
        self.action_clip = 1.0
        self.obs_clip = OBS_CLIP
        self.use_rl_games = True
        self.is_eval = False
        self.demo_length = max_episode_length
        self.max_episode_length = chunk_ep_length if chunk_ep_length > 0 else max_episode_length
        self.chunk_ep_length = chunk_ep_length
        self.early_reset_threshold = early_reset_threshold
        self.early_reset_interval = EARLY_RESET_INTERVAL
        self.early_reset_aux_thres: dict[str, float] = {}
        self.table_height = TABLE_HEIGHT
        self.n_objects = 1
        self.object_names = ["box"]
        self.objects: dict[str, _Object] = {"box": _Object(NUM_ENVS)}
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.int32)
        self.episode_start_buf = torch.zeros(NUM_ENVS, dtype=torch.int32)
        self.nan_envs = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.fell_off_mask = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.early_reset_overrides = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.early_reset_mask = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.cumulative_task_rew = torch.zeros(NUM_ENVS)
        self.cumulative_con_rew = torch.zeros(NUM_ENVS)
        self.cumulative_imi_rew = torch.zeros(NUM_ENVS)
        self.cumulative_bc_rew = torch.zeros(NUM_ENVS)
        self.reset_buf = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.reset_terminated = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.reset_time_outs = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.rew_buf = torch.zeros(NUM_ENVS)
        self.obs_buf = torch.zeros(NUM_ENVS, OBS_DIM)
        self.extras: dict[str, Any] = {}
        self.rewards = torch.tensor(rewards)
        self.curriculum: object | None = None
        self.fail_next_step = False
        self.maniptrans_resets = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.silent_resets = torch.zeros(NUM_ENVS, dtype=torch.bool)

    def get_observations(self) -> dict[str, torch.Tensor]:
        offsets = ENV_OFFSET * torch.arange(NUM_ENVS, dtype=torch.float32)
        return {"policy": (self.episode_length_buf.float() + offsets)[:, None].repeat(1, OBS_DIM)}

    def reset_idx(self, env_idxs: torch.Tensor, update_curriculum: bool = True) -> None:
        self.episode_length_buf[env_idxs] = 0
        self.cumulative_task_rew[env_idxs] = 0.0

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        self.reset_idx(torch.arange(NUM_ENVS))
        return self.get_observations(), {}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        if self.fail_next_step:
            raise RuntimeError("simulator failed")
        self.episode_length_buf += 1
        need_reset, timeout = self._get_dones()
        self.reset_terminated[:], self.reset_time_outs[:] = need_reset, timeout
        self.reset_buf[:] = need_reset | timeout | self.maniptrans_resets
        self.rew_buf[:] = self.rewards
        self.cumulative_task_rew += self.rew_buf
        self.extras["log"] = {"pos_dist": self.rew_buf}
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).squeeze(-1))
        self.reset_idx(self.silent_resets.nonzero(as_tuple=False).squeeze(-1))
        self.obs_buf[:] = self.get_observations()["policy"]
        return {"policy": self.obs_buf}, self.rew_buf, self.reset_terminated, self.reset_time_outs, self.extras


def _vec(env: FakeBaseEnv) -> DexMachinaVecEnv:
    return DexMachinaVecEnv(env)


def _reset_and_step(vec: DexMachinaVecEnv, times: int = 1) -> Any:
    vec.reset()
    for _ in range(times - 1):
        vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))
    return vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))


def _flags(tensor: torch.Tensor) -> list[bool]:
    return tensor.tolist()


def test_demo_end_is_terminated_and_final_obs_is_the_ended_episode() -> None:
    vec = _vec(FakeBaseEnv(max_episode_length=DEMO_LENGTH))
    obs, _, terminated, truncated, info = _reset_and_step(vec, DEMO_LENGTH)
    assert _flags(terminated) == [True, True]
    assert _flags(truncated) == [False, False]
    assert _flags(info["end_cause"].demo_end) == [True, True]
    assert _flags(info["_final_obs"]) == [True, True]
    assert info["final_obs"][:, 0].tolist() == [float(DEMO_LENGTH), DEMO_LENGTH + ENV_OFFSET]
    assert obs[:, 0].tolist() == [0.0, ENV_OFFSET]


def test_final_obs_differs_from_obs_only_on_ended_envs() -> None:
    env = FakeBaseEnv(max_episode_length=10, rewards=(0.0, 1.0), early_reset_threshold=EARLY_RESET_THRESHOLD)
    obs, _, terminated, _, info = _reset_and_step(_vec(env), EARLY_RESET_INTERVAL + 1)
    assert _flags(terminated) == [True, False]
    assert _flags((info["final_obs"] != obs).any(dim=-1)) == [True, False]


def test_steps_without_resets_report_no_flags() -> None:
    obs, _, terminated, truncated, info = _reset_and_step(_vec(FakeBaseEnv()))
    assert not terminated.any() and not truncated.any() and not info["_final_obs"].any()
    assert torch.equal(info["final_obs"], obs)


def test_chunk_timeout_is_truncated_not_terminated() -> None:
    _, _, terminated, truncated, info = _reset_and_step(_vec(FakeBaseEnv(chunk_ep_length=2)), 2)
    assert _flags(truncated) == [True, True]
    assert _flags(terminated) == [False, False]
    assert _flags(info["end_cause"].chunk_timeout) == [True, True]
    assert not info["end_cause"].demo_end.any()


def test_dropped_object_is_terminated() -> None:
    vec = _vec(FakeBaseEnv())
    vec.reset()
    vec.env.objects["box"].root_pos[0, 2] = BELOW_TABLE  # pyright: ignore[reportAttributeAccessIssue]
    _, _, terminated, truncated, info = vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))
    assert _flags(terminated) == [True, False]
    assert not truncated.any()
    assert _flags(info["end_cause"].dropped) == [True, False]


def test_drop_that_early_reset_overwrites_does_not_reset_and_is_not_flagged() -> None:
    env = FakeBaseEnv(early_reset_threshold=EARLY_RESET_THRESHOLD, rewards=(1.0, 1.0))
    vec = _vec(env)
    vec.reset()
    env.objects["box"].root_pos[0, 2] = BELOW_TABLE
    _, _, terminated, truncated, info = vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))
    assert not terminated.any() and not truncated.any() and not info["_final_obs"].any()
    assert env.episode_length_buf.tolist() == [1, 1]


def test_drop_in_the_same_step_as_an_early_reset_is_reported_as_early_reset_only() -> None:
    env = FakeBaseEnv(max_episode_length=10, early_reset_threshold=EARLY_RESET_THRESHOLD, rewards=(0.0, 1.0))
    vec = _vec(env)
    vec.reset()
    env.objects["box"].root_pos[0, 2] = BELOW_TABLE
    for _ in range(EARLY_RESET_INTERVAL):
        vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))
    _, _, terminated, _, info = vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))
    assert _flags(terminated) == [True, False]
    assert _flags(info["end_cause"].early_reset) == [True, False]
    assert _flags(info["end_cause"].dropped) == [False, False]


def test_auxiliary_reward_threshold_resets_after_its_check_interval() -> None:
    env = FakeBaseEnv(max_episode_length=AUX_CHECK_STEPS + 10)
    env.early_reset_aux_thres = {"con": EARLY_RESET_THRESHOLD}
    vec = _vec(env)
    vec.reset()
    for _ in range(AUX_CHECK_STEPS):
        _, _, terminated, _, _ = vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))
        assert not terminated.any()
    _, _, terminated, truncated, info = vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))
    assert _flags(terminated) == [True, True]
    assert not truncated.any()
    assert _flags(info["end_cause"].early_reset) == [True, True]


def test_eval_env_ends_only_at_the_demo_end() -> None:
    env = FakeBaseEnv(early_reset_threshold=EARLY_RESET_THRESHOLD, rewards=(0.0, 0.0))
    env.is_eval = True
    vec = _vec(env)
    vec.reset()
    env.objects["box"].root_pos[0, 2] = BELOW_TABLE
    for _ in range(DEMO_LENGTH - 1):
        _, _, terminated, truncated, info = vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))
        assert not terminated.any() and not truncated.any() and not info["_final_obs"].any()
    _, _, terminated, _, info = vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))
    assert _flags(info["end_cause"].demo_end) == [True, True]
    assert not info["end_cause"].dropped.any() and not info["end_cause"].early_reset.any()


def test_chunk_longer_than_half_the_demo_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"chunk_ep_length=3 is too long for a demo of 3 steps.*<= 2"):
        _vec(FakeBaseEnv(max_episode_length=DEMO_LENGTH, chunk_ep_length=DEMO_LENGTH))


def test_a_failing_env_step_leaves_no_capture_behind() -> None:
    env = FakeBaseEnv()
    vec = _vec(env)
    vec.reset()
    env.fail_next_step = True
    with pytest.raises(RuntimeError, match="simulator failed"):
        vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))
    assert vec._capture is None  # pyright: ignore[reportPrivateUsage]


def test_early_reset_is_terminated() -> None:
    env = FakeBaseEnv(max_episode_length=10, early_reset_threshold=EARLY_RESET_THRESHOLD, rewards=(0.0, 1.0))
    _, _, terminated, truncated, info = _reset_and_step(_vec(env), EARLY_RESET_INTERVAL + 1)
    assert _flags(terminated) == [True, False]
    assert not truncated.any()
    assert _flags(info["end_cause"].early_reset) == [True, False]


def test_nan_is_terminated_even_when_a_chunk_ends_in_the_same_reset_and_step() -> None:
    env = FakeBaseEnv(chunk_ep_length=1)
    vec = _vec(env)
    vec.reset()
    env.nan_envs[0] = True
    _, _, terminated, truncated, info = vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))
    assert _flags(terminated) == [True, False]
    assert _flags(truncated) == [False, True]
    assert _flags(info["end_cause"].nan) == [True, False]


def test_maniptrans_early_termination_is_terminated() -> None:
    env = FakeBaseEnv()
    env.curriculum = ManipTransCurriculum.__new__(ManipTransCurriculum)
    vec = _vec(env)
    vec.reset()
    env.maniptrans_resets[0] = True
    _, _, terminated, truncated, info = vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))
    assert _flags(terminated) == [True, False]
    assert not truncated.any()
    assert _flags(info["end_cause"].maniptrans) == [True, False]


def test_terminated_or_truncated_is_exactly_the_set_of_reset_envs() -> None:
    env = FakeBaseEnv(chunk_ep_length=2, rewards=(0.0, 1.0))
    vec = _vec(env)
    vec.reset()
    env.nan_envs[1] = True
    for _ in range(3):
        _, _, terminated, truncated, info = vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))
        assert torch.equal(terminated | truncated, env.reset_buf)
        assert torch.equal(info["_final_obs"], env.reset_buf)


def test_a_reset_without_cause_raises() -> None:
    env = FakeBaseEnv()
    vec = _vec(env)
    vec.reset()
    env.silent_resets[1] = True
    with pytest.raises(RuntimeError, match=r"Envs \[1\] were reset by BaseEnv without a known end cause"):
        vec.step(torch.zeros(NUM_ENVS, ACTION_DIM))


def test_total_env_steps_counts_num_envs_per_reset_and_step() -> None:
    vec = _vec(FakeBaseEnv())
    _reset_and_step(vec, 4)
    assert vec.total_env_steps == 4 * NUM_ENVS


def test_returned_tensors_are_not_the_envs_buffers() -> None:
    env = FakeBaseEnv(rewards=(1.0, 1.0))
    vec = _vec(env)
    obs, reward, terminated, _, info = _reset_and_step(vec)
    assert obs.data_ptr() != env.obs_buf.data_ptr()
    assert reward.data_ptr() != env.rew_buf.data_ptr()
    assert terminated.data_ptr() != env.reset_terminated.data_ptr()
    assert info["log"]["pos_dist"].data_ptr() != env.rew_buf.data_ptr()


@needs_gymnasium
def test_conformance_check_passes_on_the_wrapper() -> None:
    check_vec_env(_vec(FakeBaseEnv(rewards=(0.0, 1.0), early_reset_threshold=EARLY_RESET_THRESHOLD)), steps=8)


class _AliasingVecEnv(DexMachinaVecEnv):
    """Hands out one buffer that every step overwrites, as BaseEnv does."""

    def __init__(self, env: FakeBaseEnv) -> None:
        super().__init__(env)
        self.returned_obs = torch.zeros(NUM_ENVS, OBS_DIM)

    def step(self, actions: torch.Tensor) -> Any:
        obs, reward, terminated, truncated, info = super().step(actions)
        self.returned_obs.copy_(obs)
        return self.returned_obs, reward, terminated, truncated, info


@needs_gymnasium
def test_conformance_check_catches_returned_buffers_that_the_next_step_overwrites() -> None:
    with pytest.raises(VecEnvContractError, match="obs returned at step 0 changed when stepping again"):
        check_vec_env(_AliasingVecEnv(FakeBaseEnv()), steps=3)


class _WrongShapeVecEnv(DexMachinaVecEnv):
    def step(self, actions: torch.Tensor) -> Any:
        obs, reward, terminated, truncated, info = super().step(actions)
        return obs, reward[:1], terminated, truncated, info


@needs_gymnasium
def test_conformance_check_catches_wrong_shapes() -> None:
    with pytest.raises(VecEnvContractError, match=r"reward has shape \(1,\), expected \(2,\)"):
        check_vec_env(_WrongShapeVecEnv(FakeBaseEnv()), steps=1)


def test_env_without_rl_games_return_is_rejected() -> None:
    env = FakeBaseEnv()
    env.use_rl_games = False
    with pytest.raises(ValueError, match="use_rl_games=True"):
        _vec(env)


def test_reset_rejects_a_seed() -> None:
    with pytest.raises(ValueError, match="seeded when it is built"):
        _vec(FakeBaseEnv()).reset(seed=1)


def test_spaces_come_from_the_env_clips() -> None:
    pytest.importorskip("gymnasium", reason="gymnasium is optional")
    vec = _vec(FakeBaseEnv())
    assert vec.single_observation_space.shape == (OBS_DIM,)
    assert vec.single_observation_space.high.tolist() == [OBS_CLIP] * OBS_DIM
    assert vec.single_action_space.shape == (ACTION_DIM,)
    assert vec.single_action_space.low.tolist() == [-1.0] * ACTION_DIM


def test_spaces_without_gymnasium_fail_with_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gymnasium.spaces", None)
    with pytest.raises(ImportError, match="need gymnasium"):
        _ = _vec(FakeBaseEnv()).single_observation_space


def test_autoreset_mode_matches_gymnasium_when_it_has_the_enum() -> None:
    vector = pytest.importorskip("gymnasium.vector", reason="gymnasium is optional")
    if not hasattr(vector, "AutoresetMode"):
        pytest.skip("gymnasium < 1.0 has no AutoresetMode")
    assert vector.AutoresetMode(AUTORESET_SAME_STEP) is vector.AutoresetMode.SAME_STEP


def test_from_spec_keeps_the_spec_and_reward_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vec_env_module, "make_env", lambda spec, device, reward_overrides: FakeBaseEnv())
    spec, overrides = replace(EnvSpec(), use_rl_games=True, hand="allegro_hand"), {"force_penalty": 0.1}
    vec = DexMachinaVecEnv.from_spec(spec, "cpu", overrides)
    assert (vec.spec, vec.reward_overrides) == (spec, overrides)
    overrides["force_penalty"] = 9.9
    assert vec.reward_overrides == {"force_penalty": 0.1}


GOLDEN_REWARD_OVERRIDES = {"force_penalty": 0.1, "task_rew_weight": 1.0, "bc_wrist_weight": 0.25}
GOLDEN_CLIP = "box-30-230-s01-01"


def _golden_vec_env(meta: dict[str, Any], reseed: Any) -> DexMachinaVecEnv:
    from conftest import flashsac_env_kwargs

    max_epochs = flashsac_env_kwargs(meta["env_kwargs"])["max_epochs"]
    spec = replace(
        EnvSpec(), num_envs=meta["num_envs"], seed=meta["seed"], max_epochs=max_epochs, hand="allegro_hand",
        clip=GOLDEN_CLIP, retarget_name="para", actuate_object=True, use_rl_games=True,
        task_rew_betas=(10.0, 1.0, 5.0), action_penalty=0.01, imi_rew_weight=0.3, imi_wrist_weight=0.5,
        contact_rew_weight=3.0, contact_beta=10.0, bc_rew_weight=0.3, bc_beta=500.0, use_retarget_contact=True,
    )  # fmt: skip
    reseed(meta["seed"])
    return DexMachinaVecEnv.from_spec(spec, "cpu", GOLDEN_REWARD_OVERRIDES)


@pytest.mark.sim
def test_real_env_matches_the_golden_rollout(genesis_cpu: Any, reseed: Any, isolated_hand_cfgs: Any) -> None:
    from conftest import assert_close, compared_steps, load_golden

    data, meta = load_golden("flashsac_allegro_box")
    vec = _golden_vec_env(meta, reseed)

    obs0, _ = vec.reset()
    assert_close(obs0, data["obs0"], "reset obs")
    compared = compared_steps(meta)
    for t in range(meta["steps"]):
        obs, rew, terminated, truncated, info = vec.step(torch.from_numpy(data["actions"][t]))
        assert torch.equal(terminated | truncated, info["_final_obs"]), f"flags and final_obs mask differ at step {t}"
        assert not truncated.any(), f"truncated at step {t} without chunking"
        if t >= compared:
            continue
        assert_close(terminated, data["terminated"][t], f"terminated at step {t}")
        assert_close(obs, data["obs"][t], f"obs at step {t}")
        assert_close(rew, data["rew"][t], f"reward at step {t}")
        assert_close(info["final_obs"], data["final_obs"][t], f"final_obs at step {t}")


@pytest.mark.sim
@needs_gymnasium
def test_real_env_conforms(genesis_cpu: Any, reseed: Any, isolated_hand_cfgs: Any) -> None:
    from conftest import load_golden

    _, meta = load_golden("flashsac_allegro_box")
    check_vec_env(_golden_vec_env(meta, reseed), steps=3)

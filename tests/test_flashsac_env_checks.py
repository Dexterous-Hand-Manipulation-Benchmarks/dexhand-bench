"""FlashSAC's DexMachina factory against real envs: what it configures must be what the built env uses."""
import pytest

from conftest import flashsac_env_kwargs, load_golden

pytestmark = pytest.mark.sim
flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")

DEVICE = "cpu"


def _golden_kwargs(**overrides):
    _, meta = load_golden("flashsac_allegro_box")
    kwargs = flashsac_env_kwargs(meta["env_kwargs"])
    return {**kwargs, **overrides}, meta


def _build(kwargs, meta):
    return flash_dex.make_dexmachina_env(meta["num_envs"], DEVICE, kwargs, meta["seed"], kwargs["env_name"])


def test_objdex_baseline_with_the_golden_rewards_is_rejected(genesis_cpu, isolated_hand_cfgs):
    kwargs, meta = _golden_kwargs()
    kwargs["reward_cfg"] = {**kwargs["reward_cfg"], "objdex_baseline": True}
    with pytest.raises(ValueError, match="obj_pos_beta"):
        _build(kwargs, meta)


def test_bc_wrist_weight_override_reaches_the_reward_module(genesis_cpu, isolated_hand_cfgs):
    bc_wrist_weight = 0.25
    kwargs, meta = _golden_kwargs()
    kwargs["reward_cfg"] = {**kwargs["reward_cfg"], "bc_wrist_weight": bc_wrist_weight}
    env = _build(kwargs, meta)
    assert env.reward_module.bc_wrist_weight == bc_wrist_weight


def test_env_without_curriculum_builds(genesis_cpu, isolated_hand_cfgs):
    kwargs, meta = _golden_kwargs(curriculum=False)
    env = _build(kwargs, meta)
    assert not env.use_curriculum

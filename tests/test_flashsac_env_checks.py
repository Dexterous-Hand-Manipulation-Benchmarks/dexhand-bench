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
    return flash_dex.make_dexmachina_env(meta["num_envs"], DEVICE, kwargs, meta["seed"], kwargs["env_name"]).env


def test_env_without_curriculum_is_rejected(genesis_cpu, isolated_hand_cfgs):
    kwargs, meta = _golden_kwargs(curriculum=False)
    with pytest.raises(ValueError, match="actuate_object"):
        _build(kwargs, meta)

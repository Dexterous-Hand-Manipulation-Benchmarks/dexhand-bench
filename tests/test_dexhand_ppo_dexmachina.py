"""dexhand-ppo end to end on the real DexMachina env (CPU): env construction, training, curriculum, evaluation,
checkpoints and provenance."""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.sim
dex = pytest.importorskip("dexhand_ppo.envs.dexmachina", reason="needs the dexhand-ppo extra")
from omegaconf import OmegaConf  # noqa: E402
from dexhand_ppo.config import Config  # noqa: E402

PROTOCOL = Path(dex.__file__).parents[1] / "configs" / "env" / "dexmachina.yaml"


def _cfg(tmp_path, preset_overrides=None):
    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(PROTOCOL), {
        "device": "cpu", "total_steps": 32, "run_dir": str(tmp_path / "run"),
        "env": {"num_envs": 2, "dexmachina": {"num_eval_envs": 1}},
        "ppo": {"num_steps": 8, "num_minibatches": 2},
        "eval": {"interval_steps": 16}, "checkpoint": {"interval_steps": 16},
    }, preset_overrides or {})
    return cfg


def test_protocol_arguments_are_all_known_to_dexmachina(genesis_cpu):
    from dexmachina.envs.constructors import get_common_argparser

    argv = dex.build_argv(OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(PROTOCOL)), 2, False)
    _, unknown = get_common_argparser().parse_known_args(argv)
    assert unknown == []


def test_short_training_run(genesis_cpu, isolated_hand_cfgs, tmp_path):
    from dexhand_ppo.train import train

    cfg = _cfg(tmp_path)
    result = train(cfg)
    assert result["global_step"] == 32
    run_dir = Path(cfg.run_dir)
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert (provenance["env"]["obs_dim"], provenance["env"]["action_dim"]) == (510, 44)
    assert provenance["env"]["max_episode_length"] == 200
    assert provenance["env"]["env_kwargs"]["curriculum_cfg"]["zero_epoch"] == 9000
    assert provenance["env"]["env_kwargs"]["env_cfg"]["rand_init_ratio"] == 0.02
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    last = metrics[-1]
    assert last["curriculum/epoch"] == 5000 and last["train/nan_rewards"] == 0
    assert 0.0 <= last["eval/auc_add"] <= 1.0 and last["eval/num_envs"] == 1
    for name in ("final.pt", "best.pt", "step_16.pt"):
        assert (run_dir / "checkpoints" / name).exists()

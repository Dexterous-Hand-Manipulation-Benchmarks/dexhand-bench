"""dexhand-ppo end to end on the real DexMachina env (CPU): the benchmark task, training, curriculum, the shared
evaluation, checkpoints, run spec and provenance."""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.sim
dex = pytest.importorskip("dexhand_ppo.envs.dexmachina", reason="needs the dexhand-ppo extra")
from dexhand_ppo.config import Config  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

BENCHMARK_ENV = Path(dex.__file__).parents[1] / "configs" / "env" / "dexmachina.yaml"


def _cfg(tmp_path, preset_overrides=None):
    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(BENCHMARK_ENV), {
        "device": "cpu", "total_steps": 32, "run_dir": str(tmp_path / "run"),
        "env": {"num_envs": 2},
        "ppo": {"num_steps": 8, "num_minibatches": 2},
        "eval": {"interval_steps": 16}, "checkpoint": {"interval_steps": 16},
    }, preset_overrides or {})
    return cfg


def test_short_training_run(genesis_cpu, isolated_hand_cfgs, tmp_path):
    from dexhand_ppo.train import train
    from dexmachina.envs.benchmark_task import RunChoices, benchmark_spec
    from dexmachina.envs.evaluation import load_trajectory, score
    from dexmachina.envs.factory import task_hash
    from dexmachina.envs.run_spec import load_run_spec
    from dexmachina.envs.spec import EnvSpec

    cfg = _cfg(tmp_path)
    result = train(cfg)
    assert result["global_step"] == 32
    run_dir = Path(cfg.run_dir)
    expected_spec = benchmark_spec(RunChoices(hand="allegro_hand", clip="box-30-230", seed=0, num_envs=2))
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert EnvSpec.from_json(json.dumps(provenance["env"]["spec"])) == expected_spec
    assert provenance["env"]["task_hash"] == task_hash(expected_spec)
    assert (provenance["env"]["obs_dim"], provenance["env"]["action_dim"]) == (510, 44)
    assert provenance["env"]["max_episode_length"] == 200
    assert provenance["env"]["curriculum_zero_epoch"] == 9000
    run_spec = load_run_spec(run_dir / "checkpoints")
    # spec and reward overrides are all that build_env_kwargs reads, so the env got the benchmark task's kwargs
    assert run_spec.spec == expected_spec and run_spec.reward_overrides == {}
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    last = metrics[-1]
    assert last["curriculum/epoch"] == 5000 and last["train/nan_rewards"] == 0
    for key in ("eval/auc_add", "eval/auc_add_of_mean", "eval/floor/auc_add", "eval/drop_rate"):
        assert 0.0 <= last[key] <= 1.0, key
    # The floor is deterministic on CPU: an evaluation that leaked state into the next would change it.
    evaluations = [m for m in metrics if "eval/floor/auc_add" in m]
    assert len(evaluations) == 2
    floors = [{k: v for k, v in m.items() if k.startswith("eval/floor/")} for m in evaluations]
    assert floors[0] == floors[1]
    final_trajectory = run_dir / "eval_trajectories" / "trajectory_step_32.npz"
    saved = sorted(p.name for p in final_trajectory.parent.iterdir())
    assert saved == ["trajectory_step_16.npz", final_trajectory.name]
    assert score(load_trajectory(final_trajectory)).auc_add == last["eval/auc_add"]
    for name in ("final.pt", "best.pt", "step_16.pt"):
        assert (run_dir / "checkpoints" / name).exists()


def test_an_evaluation_leaves_the_global_random_streams_alone(genesis_cpu, isolated_hand_cfgs, tmp_path):
    # The online evaluation sets no seed because nothing random reaches it; reseeding would change training.
    import random

    import numpy as np
    import torch

    cfg = _cfg(tmp_path)
    evaluator = dex.DexMachinaEvaluator(dex.run_spec(cfg), torch.device("cpu"), tmp_path / "trajectories")
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    evaluator.evaluate(lambda obs: torch.zeros(obs.shape[0], evaluator.env.action_dim), global_step=0)
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert before[0] == after[0]
    assert all(np.array_equal(a, b) for a, b in zip(before[1][1:3], after[1][1:3]))
    assert torch.equal(before[2], after[2])

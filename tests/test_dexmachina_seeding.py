"""The run seed must reach DexMachina's own random draws, and building an env must leave the global RNGs alone.

The curriculum used to reseed numpy and torch with a fixed 42 at every build, and the ADD vertex sampling reseeded numpy.
"""

import json

import numpy as np
import pytest
import torch

from conftest import GOLDEN_DIR, flashsac_env_kwargs

curriculum = pytest.importorskip("dexmachina.envs.curriculum", reason="needs dexmachina")
constructors = pytest.importorskip("dexmachina.envs.constructors", reason="needs genesis and dexmachina")
dexmachina_object = pytest.importorskip("dexmachina.envs.object", reason="needs genesis and dexmachina")

FLASHSAC_GOLDEN = "flashsac_allegro_box"
GLOBAL_SEEDS = (0, 123)
RUN_SEED = 7
NUM_ENVS = 2
OBJECT_DOF = 1
EPISODE_LENGTH = 200
NUM_VERTICES = 500
NUM_SAMPLES = 300
DEFAULT_VERTEX_SEED = 42


class _StubSolver:
    def set_geom_sol_params(self, params):
        pass


class _StubObject:
    device = "cpu"
    entity = type("Entity", (), {"_solver": _StubSolver()})()

    def __init__(self):
        self.gains = []

    def set_joint_gains(self, **kwargs):
        self.gains.append(kwargs)


def _make_curriculum(seed, obj=None):
    cfg = curriculum.get_curriculum_cfg(dict(schedule="uniform", decay_solimp=True, seed=seed))
    curr = curriculum.Curriculum(cfg, obj or _StubObject(), ["task"], num_envs=NUM_ENVS,
                                 achieved_length=EPISODE_LENGTH, max_episode_length=EPISODE_LENGTH)
    curr.obj_ndof = OBJECT_DOF
    curr.curr_gains = {key: 2.0 for key in curr.curr_gains}
    curr.curr_gains_lower = {key: 1.0 for key in curr.curr_gains}
    return curr


def _draw_global():
    return np.random.rand(3), torch.rand(3)


@pytest.mark.parametrize("global_seed", GLOBAL_SEEDS)
def test_curriculum_leaves_global_rngs_alone(global_seed):
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    expected = _draw_global()

    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    curr = _make_curriculum(RUN_SEED)
    curr.reset_object_gains()
    curr.reset_solimp()
    actual = _draw_global()

    np.testing.assert_array_equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_curriculum_draws_follow_its_seed():
    def sampled_gains(seed, global_seed):
        obj = _StubObject()
        curr = _make_curriculum(seed, obj)
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        curr.reset_object_gains()
        return obj.gains[0]["kp"]

    first, other_global_seed = (sampled_gains(RUN_SEED, global_seed) for global_seed in GLOBAL_SEEDS)
    assert torch.equal(first, other_global_seed)
    assert not torch.equal(first, sampled_gains(RUN_SEED + 1, GLOBAL_SEEDS[0]))


class _ConfigCaptured(Exception):
    def __init__(self, env_cfg):
        self.env_cfg = env_cfg


def test_env_config_passes_the_run_seed_to_the_curriculum(genesis_cpu, monkeypatch):
    flash_dex = pytest.importorskip("flash_rl.envs.dexmachina", reason="needs the flashsac extra")
    import dexmachina.envs.factory as factory

    def capture_env_cfg(args, device):
        raise _ConfigCaptured(constructors.get_all_env_cfg(args, device=device))

    monkeypatch.setattr(factory, "get_all_env_cfg", capture_env_cfg)
    golden = json.loads((GOLDEN_DIR / f"{FLASHSAC_GOLDEN}.json").read_text())["env_kwargs"]
    with pytest.raises(_ConfigCaptured) as captured:
        flash_dex.make_dexmachina_env(NUM_ENVS, "cpu", flashsac_env_kwargs(golden), RUN_SEED, golden["env_name"])
    assert captured.value.env_cfg["curriculum_cfg"]["seed"] == RUN_SEED


def _object_with_mesh(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    mesh_path = tmp_path / "mesh.obj"
    trimesh.creation.icosphere(subdivisions=4).export(mesh_path)
    obj = type("Object", (), {"cfg": {"top_mesh_fname": str(mesh_path)}})()
    return obj, trimesh.load(mesh_path).vertices


def test_mesh_vertices_are_unchanged_and_global_numpy_is_untouched(tmp_path):
    obj, vertices = _object_with_mesh(tmp_path)
    np.random.seed(DEFAULT_VERTEX_SEED)
    idxs = np.random.choice(vertices.shape[0], NUM_SAMPLES, replace=NUM_SAMPLES > vertices.shape[0])
    expected = torch.tensor(vertices[idxs], dtype=torch.float32)

    np.random.seed(RUN_SEED)
    expected_next_draw = np.random.rand()
    np.random.seed(RUN_SEED)
    sampled = dexmachina_object.ArticulatedObject.sample_mesh_vertices(obj, NUM_SAMPLES)

    assert torch.equal(sampled, expected)
    assert np.random.rand() == expected_next_draw

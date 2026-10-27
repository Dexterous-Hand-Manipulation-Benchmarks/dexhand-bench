from omegaconf import OmegaConf

from dexhand_bench import config
from dexhand_bench.cli import main


def test_local_yaml_overrides_defaults(tmp_path, monkeypatch):
    local = tmp_path / "local.yaml"
    local.write_text("wandb:\n  entity: my-team\nslurm:\n  partition: gpu_a100\n")
    monkeypatch.setenv("DHB_CONFIG", str(local))
    cfg = config.load_config()
    assert cfg.wandb.entity == "my-team"
    assert cfg.wandb.project == "dexmachina_clean"  # from default.yaml
    assert cfg.slurm.partition == "gpu_a100"


def test_environment_variables_win(monkeypatch):
    cfg = OmegaConf.create({"wandb": {"entity": "from-config", "project": "p"}})
    monkeypatch.setenv("WANDB_ENTITY", "from-env")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    assert config.wandb_settings(cfg) == {"WANDB_ENTITY": "from-env", "WANDB_PROJECT": "p"}


def test_sbatch_args_merge_job_overrides():
    cfg = OmegaConf.create({"slurm": {
        "partition": "gpu", "account": None, "gres": "gpu:1", "cpus_per_task": 8, "extra_args": ["--mail-type=END"],
        "jobs": {"ppo": {"time": "00:30:00", "partition": "short"}},
    }})
    assert config.sbatch_args(cfg, "ppo") == [
        "--partition=short", "--time=00:30:00", "--gres=gpu:1", "--cpus-per-task=8", "--mail-type=END"]
    assert config.sbatch_args(cfg, "other")[0] == "--partition=gpu"


def test_cli_env_prints_exports(capsys, monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "team x")
    main(["env"])
    assert "export WANDB_ENTITY='team x'" in capsys.readouterr().out

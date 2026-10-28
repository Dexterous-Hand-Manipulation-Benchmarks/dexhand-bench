"""`dexhand-bench` command line: read the benchmark config from shell scripts."""
import argparse
import shlex

from omegaconf import OmegaConf

from dexhand_bench.config import load_config, sbatch_args, wandb_settings


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="dexhand-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("config", help="print the merged config, or one value (empty if null)")
    show.add_argument("key", nargs="?")
    sub.add_parser("env", help="print `export WANDB_ENTITY=... WANDB_PROJECT=...` for eval in shell scripts")
    sb = sub.add_parser("sbatch-args", help="print sbatch flags for a job from the slurm config")
    sb.add_argument("job")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.command == "config":
        if args.key is None:
            print(OmegaConf.to_yaml(cfg), end="")
        else:
            value = OmegaConf.select(cfg, args.key)
            print("" if value is None else value)
    elif args.command == "env":
        for name, value in wandb_settings(cfg).items():
            print(f"export {name}={shlex.quote(value)}" if value else f"unset {name}")
    elif args.command == "sbatch-args":
        print(" ".join(shlex.quote(a) for a in sbatch_args(cfg, args.job)))


if __name__ == "__main__":
    main()

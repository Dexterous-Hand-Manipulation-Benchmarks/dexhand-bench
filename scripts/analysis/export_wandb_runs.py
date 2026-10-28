"""Export every run's config, summary and code version from W&B to one JSON file for offline analysis."""
import argparse
import json
from pathlib import Path

import wandb

from dexhand_bench.config import load_config, wandb_settings

SUMMARY_TYPES = (bool, int, float, str)
METADATA_KEYS = ("program", "args", "git", "startedAt")  # no host names: they identify private machines


def run_record(run) -> dict:
    metadata = run.metadata or {}
    return {
        "id": run.id,
        "name": run.name,
        "project": run.project,
        "group": run.group,
        "state": run.state,
        "created_at": run.created_at,
        "tags": run.tags,
        "config": run.config,
        "summary": {k: v for k, v in run.summary.items() if isinstance(v, SUMMARY_TYPES)},
        "metadata": {k: metadata[k] for k in METADATA_KEYS if k in metadata},
    }


def main() -> None:
    settings = wandb_settings(load_config())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=settings["WANDB_ENTITY"], help="default: your W&B default entity")
    parser.add_argument("--project", action="append", help="repeatable; default: the configured project")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    projects = args.project or [settings["WANDB_PROJECT"]]
    if not all(projects):
        parser.error("no W&B project: pass --project or set wandb.project in configs/")

    api = wandb.Api()
    entity = args.entity or api.default_entity
    records = [run_record(run) for project in projects for run in api.runs(f"{entity}/{project}")]
    args.output.write_text(json.dumps(records, indent=1, default=str))
    print(f"{len(records)} runs from {', '.join(projects)} -> {args.output}")


if __name__ == "__main__":
    main()

# pyright: strict
"""ADD-AUC of trivial "floor" policies on the benchmark task, on the CPU.

The floor is what a policy scores without having learned anything: "zero" sends the all-zero action, "uniform"
samples every action dimension uniformly within the env's action bounds. Reading a trained policy's eval/auc_add
against this floor shows how much of it is learned.

Every cell (hand, clip, policy, seed) is one rollout of the benchmark env built by `make_eval_env` (1 env, dexmachina's
`is_eval` mode, unassisted object, no early resets), the same env and protocol the evaluation uses. Each step is
recorded as a `StepRecord` before the env's reset and scored by `evaluation.score()`, so the floor and a policy's
eval/auc_add go through one scorer: AUC per object part over all frames, overall = mean over the parts.

What "zero" means in the benchmark's hybrid action mode: the wrist joints follow the retargeted demo wrist (residual
0), the finger joints go to the middle of their limits (the absolute mapping of 0), so a zero action is not "replay
the demo". It is deterministic, so its seeds differ only through Genesis' own randomness.

Hands with a missing retargeted clip (dex3_hand and xhand have only the box clip) fail at env construction; the cell
is recorded as failed and the run continues.

Defaults are the full run. One quick cell:
    uv run --exact --extra flashsac python scripts/analysis/floor_add_auc.py --hands allegro_hand \
        --clips box-30-230-s01-01 --policies zero --seeds 1 --out-dir <dir>

Outputs in --out-dir: floor_runs.csv (one row per cell, appended as cells finish) and floor_summary.md.
"""

import argparse
import copy
import csv
import gc
import importlib
import pkgutil
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any

import genesis as gs  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import torch
from dexmachina.envs.benchmark_task import (  # pyright: ignore[reportMissingTypeStubs]
    BENCHMARK_CLIPS,
    RunChoices,
    benchmark_spec,
)
from dexmachina.envs.evaluation import (  # pyright: ignore[reportMissingTypeStubs]
    EvalScore,
    rollout_episode,
    score,
)
from dexmachina.envs.factory import make_eval_env  # pyright: ignore[reportMissingTypeStubs]
from genesis.utils import set_random_seed  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

DEFAULT_HANDS = ("allegro_hand", "inspire_hand", "dex3_hand", "xhand")
DEFAULT_SEEDS = 3
CPU = "cpu"
# dexmachina finds a hand's config dict by this attribute suffix (get_hand_specific_cfg in envs/robot.py).
HAND_CFG_SUFFIX = "_CFGS"
RUNS_CSV = "floor_runs.csv"
SUMMARY_MD = "floor_summary.md"
# The floor is scored in memory and never stored, so its trajectories carry no spec hash or SHA.
NOT_STORED = ""


class Policy(str, Enum):
    ZERO = "zero"
    UNIFORM = "uniform"


@dataclass(frozen=True)
class CellRow:
    hand: str
    clip: str
    policy: str
    seed: int
    auc_overall: float
    auc_top: float
    auc_bottom: float
    mean_add: float
    drop_rate: float
    nan_rate: float
    wall_seconds: float
    error: str


CSV_COLUMNS = tuple(f.name for f in fields(CellRow))


def snapshot_hand_cfgs() -> dict[tuple[Any, str], Any]:
    """Pristine copies of dexmachina's hand configs: every env build extends a hand's kpt_link_names in place."""
    hand_cfgs = importlib.import_module("dexmachina.envs.hand_cfgs")
    snapshot: dict[tuple[Any, str], Any] = {}
    for module_info in pkgutil.iter_modules(hand_cfgs.__path__):
        module = importlib.import_module(f"{hand_cfgs.__name__}.{module_info.name}")
        for name in dir(module):
            if name.upper().endswith(HAND_CFG_SUFFIX):
                snapshot[(module, name)] = copy.deepcopy(getattr(module, name))
    return snapshot


def restore_hand_cfgs(snapshot: dict[tuple[Any, str], Any]) -> None:
    for (module, name), pristine in snapshot.items():
        setattr(module, name, copy.deepcopy(pristine))


def build_env(hand: str, clip: str, seed: int) -> Any:
    spec = benchmark_spec(RunChoices(hand=hand, clip=clip, seed=seed, num_envs=1))
    return make_eval_env(spec, CPU)


def action_source(policy: Policy, env: Any, seed: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """The policy's action for every next step; the uniform policy draws from a generator seeded with `seed`."""
    shape = (env.num_envs, env.action_dim)
    if policy is Policy.ZERO:
        return lambda _observation: torch.zeros(shape, device=env.device)
    generator = torch.Generator(device=env.device).manual_seed(seed)

    def uniform(_observation: torch.Tensor) -> torch.Tensor:
        return (torch.rand(shape, generator=generator, device=env.device) * 2.0 - 1.0) * env.action_clip

    return uniform


def run_episode(env: Any, policy: Policy, seed: int, hand: str, clip: str) -> EvalScore:
    trajectory = rollout_episode(
        env, action_source(policy, env, seed), hand, clip, spec_hash=NOT_STORED, dexmachina_sha=NOT_STORED
    )
    return score(trajectory)


def run_cell(hand: str, clip: str, policy: Policy, seed: int, cfg_snapshot: dict[tuple[Any, str], Any]) -> CellRow:
    start = time.monotonic()
    nan = float("nan")
    key: dict[str, Any] = dict(hand=hand, clip=clip, policy=policy.value, seed=seed)
    env = None
    try:
        restore_hand_cfgs(cfg_snapshot)
        set_random_seed(seed)
        env = build_env(hand, clip, seed)
        result = run_episode(env, policy, seed, hand, clip)
        return CellRow(
            **key, auc_overall=result.auc_add, auc_top=result.auc_add_part["top"],
            auc_bottom=result.auc_add_part["bottom"], mean_add=result.mean_add,
            drop_rate=result.drop_rate, nan_rate=result.nan_rate, wall_seconds=time.monotonic() - start, error="",
        )  # fmt: skip
    except Exception as error:  # noqa: BLE001  one failed cell must not stop the unattended run; it is recorded and listed
        return CellRow(
            **key, auc_overall=nan, auc_top=nan, auc_bottom=nan, mean_add=nan, drop_rate=nan,
            nan_rate=nan, wall_seconds=time.monotonic() - start,
            error=f"{type(error).__name__}: {error}".replace("\n", " "),
        )  # fmt: skip
    finally:
        del env
        gc.collect()


def append_row(csv_path: Path, row: CellRow) -> None:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open(newline="") as existing:
            header = next(csv.reader(existing))
        if tuple(header) != CSV_COLUMNS:
            raise ValueError(
                f"{csv_path} has the columns {header}, this script writes {list(CSV_COLUMNS)}. Appending would "
                "misalign the rows: use a new --out-dir."
            )
    with csv_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(asdict(row))


def summarize(rows: Sequence[CellRow]) -> str:
    """Markdown: failed cells first, then mean and min-max of the overall AUC over the seeds of each cell."""
    failed = [row for row in rows if row.error]
    lines = ["# Floor ADD-AUC", ""]
    if failed:
        lines += [f"## FAILED CELLS ({len(failed)} of {len(rows)} runs)", ""]
        lines += [f"- {r.hand} / {r.clip} / {r.policy} / seed {r.seed}: {r.error}" for r in failed]
        lines.append("")
    groups: dict[tuple[str, str, str], list[CellRow]] = defaultdict(list)
    for row in rows:
        if not row.error:
            groups[(row.hand, row.clip, row.policy)].append(row)
    lines += [
        "| hand | clip | policy | seeds | AUC mean | AUC min-max | mean ADD [m] | dropped | NaN |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (hand, clip, policy), cell in sorted(groups.items()):
        auc = [row.auc_overall for row in cell]
        lines.append(
            f"| {hand} | {clip} | {policy} | {len(cell)} | {np.mean(auc):.4f} | {min(auc):.4f}-{max(auc):.4f} "
            f"| {np.mean([row.mean_add for row in cell]):.4f} | {np.mean([row.drop_rate for row in cell]):.2f} "
            f"| {np.mean([row.nan_rate for row in cell]):.2f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hands", nargs="+", default=list(DEFAULT_HANDS))
    parser.add_argument("--clips", nargs="+", default=list(BENCHMARK_CLIPS))
    parser.add_argument("--policies", nargs="+", type=Policy, choices=list(Policy), default=list(Policy))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="number of seeds, 0..N-1")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    csv_path = out_dir / RUNS_CSV
    if csv_path.exists():
        raise FileExistsError(
            f"{csv_path} exists; rows are appended, so reusing it would mix runs. Use a new --out-dir."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    gs.init(backend=gs.cpu, logging_level="warning", seed=0)  # pyright: ignore
    cfg_snapshot = snapshot_hand_cfgs()
    cells = [
        (hand, clip, policy, seed)
        for hand in args.hands
        for clip in args.clips
        for policy in args.policies
        for seed in range(args.seeds)
    ]
    rows: list[CellRow] = []
    run_start = time.monotonic()
    for index, (hand, clip, policy, seed) in enumerate(cells, start=1):
        row = run_cell(hand, clip, policy, seed, cfg_snapshot)
        rows.append(row)
        append_row(csv_path, row)
        (out_dir / SUMMARY_MD).write_text(summarize(rows))
        outcome = f"FAILED {row.error}" if row.error else f"auc {row.auc_overall:.4f}"
        print(
            f"[{index}/{len(cells)}] {hand} {clip} {policy.value} seed {seed}: {outcome} "
            f"(cell {row.wall_seconds:.0f} s, total {time.monotonic() - run_start:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()

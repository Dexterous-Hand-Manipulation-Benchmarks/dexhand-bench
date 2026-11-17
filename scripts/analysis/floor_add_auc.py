# pyright: strict
"""ADD-AUC of trivial "floor" policies on the benchmark task, on the CPU.

The floor is what a policy scores without having learned anything: "zero" sends the all-zero action, "uniform"
samples every action dimension uniformly within the env's action bounds. Reading a trained policy's eval/auc_add
against this floor shows how much of it is learned.

Every cell (hand, clip, policy, seed) builds the benchmark env (`benchmark_spec`) with early resets and the
object-fell-off reset off, zeroes the object's assistive gains as FlashSAC's unassisted eval pass does, and runs
`max_episode_length` steps, so every env tracks the whole clip once, as in the benchmark's evaluation protocol
(unassisted, no early resets). The env is not in dexmachina's `is_eval` mode (that mode freezes its last env). The
early_resets column should stay 0; anything else is a NaN env. Per-frame ADD comes from dexmachina's online
`ADDMetric` (same vertex sampling as the offline `compute_add.py`) and is aggregated by dexmachina's
`compute_add_stats`, which is what FlashSAC's eval/auc_add reads: AUC per object part over all (env, frame) values,
overall = mean over the parts.

What "zero" means in the benchmark's hybrid action mode: the wrist joints follow the retargeted demo wrist (residual
0), the finger joints go to the middle of their limits (the absolute mapping of 0), so a zero action is not "replay
the demo". It is deterministic, so its seeds differ only through Genesis' own randomness.

Hands with a missing retargeted clip (dex3_hand and xhand have only the box clip) fail at env construction; the cell
is recorded as failed and the run continues.

Defaults are the full run. One quick cell:
    uv run --exact --extra flashsac python scripts/analysis/floor_add_auc.py --hands allegro_hand \
        --clips box-30-230-s01-01 --policies zero --seeds 1 --num-envs 2 --out-dir <dir>

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
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import genesis as gs  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
import torch
from dexmachina.envs.add_metric import ADD_AUC_THRESHOLDS  # pyright: ignore[reportMissingTypeStubs]
from dexmachina.envs.benchmark_task import (  # pyright: ignore[reportMissingTypeStubs]
    BENCHMARK_CLIPS,
    RunChoices,
    benchmark_spec,
)
from dexmachina.envs.factory import make_env  # pyright: ignore[reportMissingTypeStubs]
from dexmachina.eval.compute_add import (  # pyright: ignore[reportMissingTypeStubs]
    compute_add_stats,  # pyright: ignore[reportUnknownVariableType]
)
from genesis.utils import set_random_seed  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

DEFAULT_HANDS = ("allegro_hand", "inspire_hand", "dex3_hand", "xhand")
DEFAULT_SEEDS = 3
DEFAULT_NUM_ENVS = 8
CPU = "cpu"
PARTS = ("top", "bottom")
# dexmachina finds a hand's config dict by this attribute suffix (get_hand_specific_cfg in envs/robot.py).
HAND_CFG_SUFFIX = "_CFGS"
RUNS_CSV = "floor_runs.csv"
SUMMARY_MD = "floor_summary.md"
ADD_LOG_KEY = "add_{part}"
# 0 switches the early-reset check off (base_env.py checks it only for a positive threshold).
NO_EARLY_RESET = 0.0
# An object below this height never counts as fallen off, so a dropped object stays dropped and is scored as a miss
# instead of restarting the episode (table_height is read only by that check in base_env._get_dones).
NEVER_FALLEN_OFF = float("-inf")
NO_REWARD_OVERRIDES: Mapping[str, Any] = MappingProxyType({})


class Policy(str, Enum):
    ZERO = "zero"
    UNIFORM = "uniform"


@dataclass(frozen=True)
class AddStats:
    auc_overall: float
    auc_top: float
    auc_bottom: float
    mean_add: float


@dataclass(frozen=True)
class CellRow:
    hand: str
    clip: str
    policy: str
    seed: int
    num_envs: int
    auc_overall: float
    auc_top: float
    auc_bottom: float
    mean_add: float
    ep_length: int
    early_resets: int
    nan_envs: int
    wall_seconds: float
    error: str


CSV_COLUMNS = tuple(f.name for f in fields(CellRow))


@dataclass(frozen=True)
class EpisodeResult:
    stats: AddStats
    ep_length: int
    early_resets: int
    nan_envs: int


def add_stats(add_by_part: dict[str, npt.NDArray[np.float64]]) -> AddStats:
    """AUC and mean ADD as FlashSAC's eval computes them; each array holds one part's ADD per (env, frame)."""
    stats: dict[str, dict[str, float]] = compute_add_stats(add_by_part, list(ADD_AUC_THRESHOLDS))  # pyright: ignore[reportUnknownVariableType]
    auc = stats["auc"]
    return AddStats(auc["overall"], auc["top"], auc["bottom"], stats["mean_add"]["overall"])


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


def build_env(hand: str, clip: str, seed: int, num_envs: int) -> Any:
    """The benchmark env without early resets, object's assistive gains zeroed, online ADD metric attached."""
    spec = benchmark_spec(RunChoices(hand=hand, clip=clip, seed=seed, num_envs=num_envs))
    env: Any = make_env(replace(spec, early_reset_threshold=NO_EARLY_RESET), CPU, NO_REWARD_OVERRIDES)
    if env.object is None or env.object.demo_states is None:
        raise ValueError(
            f"the env for {hand}/{clip} has no object with a demo trajectory, so there is no ADD to measure"
        )
    if env.object.actuated:
        env.object.set_joint_gains(0.0, 0.0, force_range=0.0)
    # BaseEnv attaches the metric itself only when env_cfg["log_add_metric"] is set, which no EnvSpec field reaches.
    env.add_metric = env.object.setup_add_metric(num_samples=env.add_num_verts)
    env.table_height = NEVER_FALLEN_OFF
    env.record_curriculum = False
    return env


def sample_actions(policy: Policy, env: Any, generator: torch.Generator) -> torch.Tensor:
    shape = (env.num_envs, env.action_dim)
    if policy is Policy.ZERO:
        return torch.zeros(shape, device=env.device)
    unit = torch.rand(shape, generator=generator, device=env.device) * 2.0 - 1.0
    return unit * env.action_clip


def run_episode(env: Any, policy: Policy, seed: int) -> EpisodeResult:
    generator = torch.Generator(device=env.device).manual_seed(seed)
    ep_length: int = env.max_episode_length
    add_per_part: dict[str, list[npt.NDArray[np.float64]]] = {part: [] for part in PARTS}
    nan_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    early_resets = 0
    env.reset()
    for step in range(ep_length):
        _, _, terminated, _, extras = env.step(sample_actions(policy, env, generator))
        for part in PARTS:
            frame = extras["log"][ADD_LOG_KEY.format(part=part)]
            add_per_part[part].append(frame.detach().cpu().numpy().astype(np.float64))
        nan_seen |= env.nan_envs
        if step < ep_length - 1:
            early_resets += int(terminated.sum().item())
    add_by_part = {part: np.stack(frames, axis=1) for part, frames in add_per_part.items()}
    return EpisodeResult(add_stats(add_by_part), ep_length, early_resets, int(nan_seen.sum().item()))


def run_cell(
    hand: str, clip: str, policy: Policy, seed: int, num_envs: int, cfg_snapshot: dict[tuple[Any, str], Any]
) -> CellRow:
    start = time.monotonic()
    nan = float("nan")
    key: dict[str, Any] = dict(hand=hand, clip=clip, policy=policy.value, seed=seed, num_envs=num_envs)
    env = None
    try:
        restore_hand_cfgs(cfg_snapshot)
        set_random_seed(seed)
        env = build_env(hand, clip, seed, num_envs)
        result = run_episode(env, policy, seed)
        stats = result.stats
        return CellRow(
            **key, auc_overall=stats.auc_overall, auc_top=stats.auc_top, auc_bottom=stats.auc_bottom,
            mean_add=stats.mean_add, ep_length=result.ep_length, early_resets=result.early_resets,
            nan_envs=result.nan_envs, wall_seconds=time.monotonic() - start, error="",
        )  # fmt: skip
    except Exception as error:  # noqa: BLE001  one failed cell must not stop the unattended run; it is recorded and listed
        return CellRow(
            **key, auc_overall=nan, auc_top=nan, auc_bottom=nan, mean_add=nan, ep_length=0, early_resets=0,
            nan_envs=0, wall_seconds=time.monotonic() - start,
            error=f"{type(error).__name__}: {error}".replace("\n", " "),
        )  # fmt: skip
    finally:
        del env
        gc.collect()


def append_row(csv_path: Path, row: CellRow) -> None:
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
        "| hand | clip | policy | seeds | AUC mean | AUC min-max | mean ADD [m] | early resets | NaN envs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (hand, clip, policy), cell in sorted(groups.items()):
        auc = [row.auc_overall for row in cell]
        lines.append(
            f"| {hand} | {clip} | {policy} | {len(cell)} | {np.mean(auc):.4f} | {min(auc):.4f}-{max(auc):.4f} "
            f"| {np.mean([row.mean_add for row in cell]):.4f} | {sum(row.early_resets for row in cell)} "
            f"| {sum(row.nan_envs for row in cell)} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hands", nargs="+", default=list(DEFAULT_HANDS))
    parser.add_argument("--clips", nargs="+", default=list(BENCHMARK_CLIPS))
    parser.add_argument("--policies", nargs="+", type=Policy, choices=list(Policy), default=list(Policy))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="number of seeds, 0..N-1")
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
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
        row = run_cell(hand, clip, policy, seed, args.num_envs, cfg_snapshot)
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

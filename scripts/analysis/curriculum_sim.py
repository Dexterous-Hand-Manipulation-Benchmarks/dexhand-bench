"""Simulates dexmachina's gain curriculum without a physics engine.

The real `Curriculum` class is driven by a synthetic reward process that is identical per environment step, so any
dependence of the decay timing on the number of parallel environments comes from how the curriculum is fed.

What is not modelled: objects falling off, NaN envs, the early-reset thresholds of the auxiliary rewards and `is_eval`.
`set_curriculum` is called before the env step, and only when the epoch has increased since the last call, as in the
training loops of FlashSAC, FastTD3 and REPPO.

The per-step reward is a function of the env's current kp and of the total env steps consumed, so the decay timing feeds
back into the rewards. The SETTINGS are invented knobs, not fitted to any run.

Start modes: "synced" = all envs start at t=0, which is what dexmachina does today (`reset()` sets every env to 0,
base_env.py). "staggered" = a hypothetical start with random episode phases; dexmachina has no such mode.

Reward models: "homogeneous" = every env draws the same reward process; "heterogeneous" = in addition each env carries a
task-reward offset drawn anew at every reset.

Variants (what the curriculum is fed with, see `CURRICULUM_VARIANTS`): "gated" = today's behaviour (control);
"per_episode_K" = one deque entry per finished episode; "scaled_deque" = deque length scaled with 1/num_envs
(control); "fixed" = fixed decay schedule.

Primary metrics: the assistance integral (mean of kp/kp_init over epochs 0..zero_epoch, kp = mean over envs) and the
epoch where kp falls below one. A cell passes if max/min - 1 of the seed means across num_envs is within
PASS_TOLERANCE for both. The first decay epoch is a diagnostic only.

Usage: uv run --exact --extra flashsac python scripts/analysis/curriculum_sim.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import itertools
import sys
import time
import types
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.signal import lfilter

REPO_ROOT = Path(__file__).resolve().parents[2]
CURRICULUM_FILE = REPO_ROOT / "environments/dexmachina/dexmachina/envs/curriculum.py"


class StartMode(str, Enum):
    SYNCED = "synced"
    STAGGERED = "staggered"


class RewardModel(str, Enum):
    HOMOGENEOUS = "homogeneous"
    HETEROGENEOUS = "heterogeneous"


class Variant(str, Enum):
    GATED = "gated"
    PER_EPISODE_K = "per_episode_K"
    SCALED_DEQUE = "scaled_deque"
    FIXED = "fixed"


INTEGRAL = "assistance_integral"
KP_BELOW_ONE = "kp_below_one_epoch"
FIRST_DECAY = "first_decay_epoch"
PRIMARY_METRICS = (INTEGRAL, KP_BELOW_ONE)

BUDGET_ENV_STEPS = 50_000_000
MAX_EPOCHS = 10_000
NUM_ZERO_EPOCH = 1_000
EPISODE_LENGTH = 200
MAX_STEP_LENGTH = EPISODE_LENGTH - 1
EARLY_RESET_THRESHOLD = 0.6
EARLY_RESET_INTERVAL = 5
MAX_ACHIEVED_EMA = 0.5
KP_INIT = 80.0
KV_INIT = 5.0
FORCE_RANGE_INIT = 50.0
KP_DECAYED = 1.0
REWARD_KEYS = ["task", "con", "imi", "bc"]
ZERO_EPOCH = max(MAX_EPOCHS - NUM_ZERO_EPOCH, 0)

# Curriculum values of the benchmark's shared run settings (FastTD3 fast_td3/hyperparams.py, REPPO
# config/env/dexmachina.yaml and the dexhand-ppo configs agree on them).
BASE_CURRICULUM_KWARGS: dict[str, Any] = {
    "kp_init": KP_INIT,
    "kv_init": KV_INIT,
    "force_range_init": FORCE_RANGE_INIT,
    "rew_thresholds": {"task": 0.6, "con": 0.01, "imi": 0.01, "bc": 0.01},
    "wait_epochs": 100,
    "decay_rew": False,
    "schedule": "uniform",
    "fixed_mode": "uniform",
    "interval": 500,
    "deque_freq": 1,
    "grad_threshold": 0.001,
    "gain_mode": "all",
    "uniform_mode": "slow",
    "upper_ratios": {"kp": 0.9, "kv": 0.9, "fr": 1.0},
    "lower_ratios": {"kp": 0.8, "kv": 0.8, "fr": 1.0},
    "deque_len": 30,
    "decay_solimp": False,
    "resample_every_epoch": -1,
    "skip_grad": True,
    "zero_epoch": ZERO_EPOCH,
    "dialback_ep_len": 80,
    "dialback_min_epochs": 500,
    "dialback_ratios": {"kp": 0.98, "kv": 1.0, "fr": 1.0},
}

PER_EPISODE_DEQUE_LEN = 512
# First full episode at 8192 envs: EPISODE_LENGTH * 8192 / (BUDGET_ENV_STEPS / MAX_EPOCHS) = 327.7, rounded up.
PER_EPISODE_WAIT_EPOCHS = 330
# One episode moves the length average by 1/K, so it averages over about K episodes.
PER_EPISODE_EMA = 1.0 / PER_EPISODE_DEQUE_LEN
SCALED_DEQUE_BASE_LEN = 30
SCALED_DEQUE_BASE_ENVS = 1024

NUM_ENVS_GRID = (1024, 2048, 4096, 8192)
RAND_INIT_GRID = (0.0, 0.02)
SEEDS = (0, 1, 2)
PASS_TOLERANCE = 0.10
NEVER = float("nan")
HETEROGENEOUS_OFFSET_STD = 0.1


@dataclass(frozen=True)
class RewardSetting:
    name: str
    base: float
    assist_weight: float
    skill_weight: float
    skill_steps: float
    noise_std: float


SETTINGS = (
    RewardSetting("easy", base=0.45, assist_weight=0.25, skill_weight=0.35, skill_steps=2e6, noise_std=0.15),
    RewardSetting("medium", base=0.30, assist_weight=0.30, skill_weight=0.45, skill_steps=8e6, noise_std=0.15),
    RewardSetting("hard", base=0.20, assist_weight=0.30, skill_weight=0.50, skill_steps=3e7, noise_std=0.15),
)
AUX_REWARD_SCALE = 0.5


@dataclass(frozen=True)
class RunSpec:
    reward_model: RewardModel
    variant: Variant
    num_envs: int
    rand_init_ratio: float
    start: StartMode
    setting: RewardSetting
    seed: int


@dataclass(frozen=True)
class RunResult:
    reward_model: str
    variant: str
    num_envs: int
    rand_init_ratio: float
    start: str
    setting: str
    seed: int
    assistance_integral: float
    kp_below_one_epoch: float
    first_decay_epoch: float
    kp_at_zero_epoch: float
    n_decays: int
    n_dialbacks: int
    median_decay_spacing: float
    update_progress_calls: int
    vector_steps: int


def load_curriculum_module() -> types.ModuleType:
    """Loads curriculum.py by path with a stub for genesis, which it imports but never uses."""
    if not CURRICULUM_FILE.exists():
        raise ImportError(
            f"{CURRICULUM_FILE} is missing: the dexmachina submodule is not checked out. "
            "Run `git submodule update --init` in the repo root."
        )
    sys.modules.setdefault("genesis", types.ModuleType("genesis"))
    spec = importlib.util.spec_from_file_location("dexmachina_curriculum", CURRICULUM_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Python cannot load {CURRICULUM_FILE} as a module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_sim_curriculum_class(module: types.ModuleType) -> type[Any]:
    """The real Curriculum plus the hooks the variants need; nothing else is overridden."""
    base: type[Any] = module.Curriculum

    class SimCurriculum(base):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.last_dialed_back = False

        def determine_dialback(self, epoch_num: int) -> bool:
            self.last_dialed_back = super().determine_dialback(epoch_num)
            return self.last_dialed_back

        def update_progress_episodes(
            self, rewards: dict[str, npt.NDArray[np.float64]], episode_lengths: npt.NDArray[np.float64]
        ) -> None:
            """update_progress (curriculum.py:309-317) with one deque entry per finished episode."""
            for key, values in rewards.items():
                self.rew_deques[key].extend(values.tolist())
            self.ep_lens.extend(episode_lengths.tolist())

    return SimCurriculum


def curriculum_kwargs(variant: Variant, num_envs: int) -> dict[str, Any]:
    kwargs = dict(BASE_CURRICULUM_KWARGS)
    if variant is Variant.PER_EPISODE_K:
        kwargs.update(deque_len=PER_EPISODE_DEQUE_LEN, wait_epochs=PER_EPISODE_WAIT_EPOCHS)
    elif variant is Variant.SCALED_DEQUE:
        kwargs.update(deque_len=max(1, round(SCALED_DEQUE_BASE_LEN * SCALED_DEQUE_BASE_ENVS / num_envs)))
    elif variant is Variant.FIXED:
        kwargs.update(schedule="fixed", budget_env_steps=BUDGET_ENV_STEPS)
    return kwargs


class FakeTaskObject:
    """Records the joint gains the curriculum sets; `kp` is the per-env mean over joints."""

    device = "cpu"

    def __init__(self, num_envs: int) -> None:
        self.kp: npt.NDArray[np.float64] = np.full(num_envs, KP_INIT)
        self.kp_mean = KP_INIT

    def set_joint_gains(
        self, kp: float | torch.Tensor, kv: float | torch.Tensor, force_range: float | torch.Tensor, env_idxs: None
    ) -> None:
        """Called by curriculum.py:349 with (num_envs, joints) tensors when the gains are sampled, else floats."""
        if isinstance(kp, torch.Tensor):
            self.kp = kp.numpy().astype(np.float64).mean(axis=1)
        else:
            self.kp = np.full_like(self.kp, kp)
        self.kp_mean = float(self.kp.mean())


@dataclass
class EnvState:
    length: npt.NDArray[np.int64]
    start: npt.NDArray[np.int64]
    cumulative: dict[str, npt.NDArray[np.float64]]
    task_offset: npt.NDArray[np.float64]
    max_achieved_length: float = 0.0


def draw_offsets(spec: RunSpec, count: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    if spec.reward_model is RewardModel.HETEROGENEOUS:
        return rng.normal(0.0, HETEROGENEOUS_OFFSET_STD, count)
    return np.zeros(count)


def initial_state(spec: RunSpec, rng: np.random.Generator) -> EnvState:
    length = np.zeros(spec.num_envs, dtype=np.int64)
    if spec.start is StartMode.STAGGERED:
        length = rng.integers(0, EPISODE_LENGTH, spec.num_envs)
    return EnvState(
        length=length,
        start=length.copy(),
        cumulative={key: np.zeros(spec.num_envs) for key in REWARD_KEYS},
        task_offset=draw_offsets(spec, spec.num_envs, rng),
    )


def reset_mask(state: EnvState) -> npt.NDArray[np.bool_]:
    """base_env._get_dones: the last early-reset interval below the stepped length decides, timeouts still reset."""
    stepped = state.length - state.start
    timeout = state.length >= EPISODE_LENGTH
    interval = (stepped - 1) // EARLY_RESET_INTERVAL * EARLY_RESET_INTERVAL
    early = state.cumulative["task"] < EARLY_RESET_THRESHOLD * interval
    need_reset = np.where(stepped > 0, early, timeout)
    return need_reset | timeout


def step_rewards(
    spec: RunSpec, state: EnvState, kp: npt.NDArray[np.float64], env_steps_consumed: int, rng: np.random.Generator
) -> dict[str, npt.NDArray[np.float64]]:
    setting = spec.setting
    assist = kp / KP_INIT
    skill = min(1.0, env_steps_consumed / setting.skill_steps)
    mean = setting.base + setting.assist_weight * assist + setting.skill_weight * skill
    rewards: dict[str, npt.NDArray[np.float64]] = {}
    for key in REWARD_KEYS:
        scale = 1.0 if key == "task" else AUX_REWARD_SCALE
        noise = rng.normal(0.0, setting.noise_std, spec.num_envs)
        rewards[key] = np.clip(scale * mean + noise, 0.0, 1.0)
    rewards["task"] = np.clip(rewards["task"] + state.task_offset, 0.0, 1.0)
    return rewards


def reset_envs(state: EnvState, reset_ids: npt.NDArray[np.intp], spec: RunSpec, rng: np.random.Generator) -> None:
    """base_env.reset_idx; the curriculum update is done by the caller before this."""
    count = len(reset_ids)
    starts = np.zeros(count, dtype=np.int64)
    if spec.rand_init_ratio > 0.0:
        randomised = rng.random(count) <= spec.rand_init_ratio
        end_t = min(int(state.max_achieved_length) + 1, MAX_STEP_LENGTH)
        starts[randomised] = rng.integers(0, end_t, count)[randomised]
    state.length[reset_ids] = starts
    state.start[reset_ids] = starts
    state.task_offset[reset_ids] = draw_offsets(spec, count, rng)
    for cumulative in state.cumulative.values():
        cumulative[reset_ids] = 0.0


def episode_rewards(state: EnvState, reset_ids: npt.NDArray[np.intp]) -> dict[str, npt.NDArray[np.float64]]:
    return {key: state.cumulative[key][reset_ids] / EPISODE_LENGTH for key in REWARD_KEYS}


def feed_curriculum(curriculum: Any, state: EnvState, reset_ids: npt.NDArray[np.intp], spec: RunSpec) -> None:
    rewards = episode_rewards(state, reset_ids)
    progressed = (state.length[reset_ids] - state.start[reset_ids]).astype(np.float64)
    if spec.variant is Variant.PER_EPISODE_K:
        # Exponential average updated once per episode: y[i] = (1-a) y[i-1] + a x[i].
        decay = 1.0 - PER_EPISODE_EMA
        averages, _ = lfilter([PER_EPISODE_EMA], [1.0, -decay], progressed, zi=[decay * state.max_achieved_length])
        state.max_achieved_length = float(averages[-1])
        curriculum.update_progress_episodes(rewards, averages)
    else:
        # int() truncation as in base_env.
        state.max_achieved_length = int(
            state.max_achieved_length * MAX_ACHIEVED_EMA + float(progressed.mean()) * MAX_ACHIEVED_EMA
        )
        mean_rewards = {key: float(values.mean()) for key, values in rewards.items()}
        curriculum.update_progress(mean_rewards, state.max_achieved_length)


def median_spacing(decay_epochs: list[int]) -> float:
    return float(np.median(np.diff(decay_epochs))) if len(decay_epochs) > 1 else NEVER


def run_one(spec: RunSpec, curriculum_class: type[Any], module: types.ModuleType) -> RunResult:
    rng = np.random.default_rng(spec.seed)
    task_object = FakeTaskObject(spec.num_envs)
    curriculum = curriculum_class(
        module.get_curriculum_cfg(curriculum_kwargs(spec.variant, spec.num_envs)),
        task_object,
        REWARD_KEYS,
        spec.num_envs,
        0,
        EPISODE_LENGTH,
    )
    state = initial_state(spec, rng)
    total_steps = BUDGET_ENV_STEPS // spec.num_envs
    update_progress_calls = 0
    decay_epochs: list[int] = []
    dialbacks = 0
    kp_below_one_epoch = NEVER
    kp_at_zero_epoch = NEVER
    assistance_sum = 0.0
    assistance_steps = 0
    last_epoch = -1

    for vector_step in range(1, total_steps + 1):
        # Same mapping as FlashSAC's curriculum_epoch (flash_rl/envs/dexmachina.py), in integer arithmetic.
        epoch = vector_step * spec.num_envs * MAX_EPOCHS // BUDGET_ENV_STEPS
        gains_changed = False
        if spec.variant is Variant.FIXED:
            # The env counts its own training steps (base_env.py) and re-draws the object gains when a decay is due.
            gains_changed = curriculum.advance_env_steps(spec.num_envs)
            if gains_changed:
                curriculum.reset_object_gains()
        elif epoch > last_epoch:
            with contextlib.redirect_stdout(io.StringIO()):
                _, gains_changed, _ = curriculum.set_curriculum(epoch)
            last_epoch = epoch

        state.length += 1
        resets = reset_mask(state)
        rewards = step_rewards(spec, state, task_object.kp, vector_step * spec.num_envs, rng)
        for key in REWARD_KEYS:
            state.cumulative[key] += rewards[key]
        reset_ids = np.flatnonzero(resets)
        if len(reset_ids) > 0:
            feed_curriculum(curriculum, state, reset_ids, spec)
            update_progress_calls += 1
            reset_envs(state, reset_ids, spec, rng)

        if epoch > ZERO_EPOCH:
            continue
        if gains_changed:
            if curriculum.last_dialed_back:
                dialbacks += 1
            else:
                decay_epochs.append(epoch)
        kp_at_zero_epoch = task_object.kp_mean
        assistance_sum += task_object.kp_mean / KP_INIT
        assistance_steps += 1
        if task_object.kp_mean < KP_DECAYED and np.isnan(kp_below_one_epoch):
            kp_below_one_epoch = epoch

    return RunResult(
        reward_model=spec.reward_model.value,
        variant=spec.variant.value,
        num_envs=spec.num_envs,
        rand_init_ratio=spec.rand_init_ratio,
        start=spec.start.value,
        setting=spec.setting.name,
        seed=spec.seed,
        assistance_integral=assistance_sum / assistance_steps,
        kp_below_one_epoch=kp_below_one_epoch,
        first_decay_epoch=decay_epochs[0] if decay_epochs else NEVER,
        kp_at_zero_epoch=kp_at_zero_epoch,
        n_decays=len(decay_epochs),
        n_dialbacks=dialbacks,
        median_decay_spacing=median_spacing(decay_epochs),
        update_progress_calls=update_progress_calls,
        vector_steps=total_steps,
    )


def grid() -> Iterator[RunSpec]:
    for reward_model, variant, num_envs, ratio, start, setting, seed in itertools.product(
        RewardModel, Variant, NUM_ENVS_GRID, RAND_INIT_GRID, StartMode, SETTINGS, SEEDS
    ):
        yield RunSpec(reward_model, variant, num_envs, ratio, start, setting, seed)


def spread(means: list[float]) -> float:
    """max/min - 1 over the num_envs means; NaN if any is NaN (the metric never happened)."""
    return NEVER if any(np.isnan(means)) else max(means) / min(means) - 1.0


def seed_mean(values: list[float]) -> float:
    """NaN if any seed never reached the event: a mean over the others would hide it (counted in never_seeds)."""
    return NEVER if any(np.isnan(values)) else float(np.mean(values))


def format_cell(values: list[float], decimals: int) -> str:
    if any(np.isnan(values)):
        return "never"
    return f"{np.mean(values):.{decimals}f} [{min(values):.{decimals}f}-{max(values):.{decimals}f}]"


def cell_verdict(spreads: dict[str, float]) -> str:
    if np.isnan(spreads[KP_BELOW_ONE]):
        return "N/A (kp never below one for some num_envs)"
    passed = all(spreads[metric] <= PASS_TOLERANCE for metric in PRIMARY_METRICS)
    return "PASS" if passed else "FAIL"


def verdict_table(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for reward_model, variant, ratio, start, setting in itertools.product(
        RewardModel, Variant, RAND_INIT_GRID, StartMode, (s.name for s in SETTINGS)
    ):
        cell = results[
            (results["reward_model"] == reward_model.value)
            & (results["variant"] == variant.value)
            & (results["rand_init_ratio"] == ratio)
            & (results["start"] == start.value)
            & (results["setting"] == setting)
        ]
        if cell.empty:
            continue
        row = {
            "reward_model": reward_model.value,
            "variant": variant.value,
            "rand_init_ratio": str(ratio),
            "start": start.value,
            "setting": setting,
        }
        spreads: dict[str, float] = {}
        never_seeds = 0
        for metric, decimals in ((INTEGRAL, 3), (KP_BELOW_ONE, 0), (FIRST_DECAY, 0)):
            per_envs = {n: [float(v) for v in cell[cell["num_envs"] == n][metric]] for n in NUM_ENVS_GRID}
            means = [seed_mean(per_envs[n]) for n in NUM_ENVS_GRID]
            spreads[metric] = spread(means)
            if metric != FIRST_DECAY:
                row[f"{metric}_spread"] = "never" if np.isnan(spreads[metric]) else f"{spreads[metric]:+.1%}"
            for n in NUM_ENVS_GRID:
                row[f"{metric}_n{n}"] = format_cell(per_envs[n], decimals)
            if metric == KP_BELOW_ONE:
                never_seeds = sum(int(np.isnan(v)) for values in per_envs.values() for v in values)
        row["kp_below_one_never_seeds"] = f"{never_seeds} of {len(NUM_ENVS_GRID) * len(SEEDS)}"
        row["verdict"] = cell_verdict(spreads)
        rows.append(row)
    return pd.DataFrame(rows)


VERDICT_LEGEND = (
    f"Cells show the seed mean and [min-max] over {len(SEEDS)} seeds per num_envs. Spread = max/min - 1 of the seed "
    f"means across num_envs; PASS needs both primary spreads within {PASS_TOLERANCE:.0%}. A metric is 'never' if any "
    "seed never reached it. start 'staggered' is hypothetical (dexmachina starts all envs at t=0 = 'synced').\n\n"
)


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def run_grid(out_dir: Path) -> pd.DataFrame:
    module = load_curriculum_module()
    curriculum_class = make_sim_curriculum_class(module)
    specs = list(grid())
    started = time.perf_counter()
    results: list[RunResult] = []
    with (out_dir / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(RunResult)])
        writer.writeheader()
        for index, spec in enumerate(specs, start=1):
            result = run_one(spec, curriculum_class, module)
            results.append(result)
            writer.writerow(asdict(result))
            handle.flush()
            elapsed = time.perf_counter() - started
            eta = elapsed / index * (len(specs) - index)
            print(
                f"[{index}/{len(specs)}] {elapsed:.0f}s eta {eta:.0f}s | {result.reward_model} {result.variant} "
                f"n={result.num_envs} rand={result.rand_init_ratio} {result.start} {result.setting} seed={result.seed}"
                f" | integral {result.assistance_integral:.3f} kp<1 {result.kp_below_one_epoch}",
                flush=True,
            )
    return pd.DataFrame([asdict(result) for result in results])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    results = run_grid(args.out_dir)
    runtime = time.perf_counter() - started

    verdicts = verdict_table(results)
    verdicts.to_csv(args.out_dir / "verdicts.csv", index=False)
    (args.out_dir / "verdicts.md").write_text(VERDICT_LEGEND + markdown_table(verdicts) + "\n")
    print(markdown_table(verdicts))
    print(f"runtime {runtime:.0f} s")


if __name__ == "__main__":
    main()

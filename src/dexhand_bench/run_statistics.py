"""Summaries of one metric of one method over seeds and clips, and the comparison of two methods.

The headline (per-clip and pooled mean with a Student-t interval) is the only part used to rank methods. Insights
(median, IQR, std, custom functions) are computed separately and recorded under their names, for tracking only.

Why an interval and not mean +- std: the std describes how much single seeds vary and does not shrink with more
seeds. The interval mean +- t * s / sqrt(n) says where the true mean (over infinitely many seeds) probably lies, and
narrows as seeds are added, which is what a comparison of methods needs. It uses Student's t instead of the normal
1.96 because s is itself estimated from few seeds; t widens the interval to cover that (4.30 for 3 seeds).

The pooled mean is the mean of the clip means, so its variance is the sum of the clips' variances of the mean divided
by K^2. Its degrees of freedom (how much data the spread estimate rests on, which sets t) come from the
Welch-Satterthwaite formula: about the sum of the clips' n - 1 when the clips contribute equally, down to one clip's
n - 1 when that clip dominates the variance.

Reading the intervals:
- The per-clip intervals are descriptive. With few seeds and values near 0 or 1 they cover the true mean far less
  often than 95 % of the time. Only the pooled interval is meant for ranking.
- The pooled interval reflects the variance between seeds on exactly this set of clips. Adding or removing clips
  changes what is estimated, not only how precisely.
- Each seed must contribute one value from the same evaluation protocol for every method: final checkpoint, same
  evaluation episodes, same environment config and same step budget.

Comparing two methods: whether their intervals overlap says little (two overlapping 95 % intervals can still differ
significantly). `compare_methods` gives the difference of the means per clip, with a Welch interval (the seeds of the
two methods are independent and their spreads may differ), and pooled the mean of those differences over the clips
both methods have, with Welch-Satterthwaite degrees of freedom over all their variances. Only the pooled difference is
meant for ranking; an interval that excludes 0 favours one method on this set of clips.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy import stats  # pyright: ignore[reportMissingTypeStubs]

SeedAggregation = Callable[[Sequence[float]], float]

CONFIDENCE_LEVEL = 0.95
QUARTILES = (25, 75)
# A spread, and with it an interval, needs at least two seeds.
MIN_SEEDS = 2
# Largest spread of the zero-action floors in one cell (all seeds of both methods) that still counts as one floor. The
# floor is bit-identical across seeds on CPU (scripts/analysis/floor_add_auc.py, 3 seeds), so a larger spread means
# different code or device, and the methods' scores would not be measured against the same task.
FLOOR_TOLERANCE = 1e-6


def _iqr(values: Sequence[float]) -> float:
    quartiles = np.percentile(np.asarray(values, dtype=np.float64), QUARTILES)
    return float(quartiles[1] - quartiles[0])


INSIGHT_PRESETS: Mapping[str, SeedAggregation] = {
    "median": lambda values: float(np.median(values)),
    "iqr": _iqr,
    "std": lambda values: float(np.std(values, ddof=1)),
}


@dataclass(frozen=True)
class MeanInterval:
    mean: float
    low: float
    high: float


@dataclass(frozen=True)
class ClipHeadline:
    clip: str
    num_seeds: int
    interval: MeanInterval
    # All seeds gave the same value, so the interval is the mean itself and says nothing about the spread.
    no_spread: bool


@dataclass(frozen=True)
class PooledHeadline:
    clips: tuple[str, ...]
    seeds_per_clip: tuple[int, ...]
    interval: MeanInterval
    clips_without_spread: tuple[str, ...]
    # None when every clip has no spread: the interval has no variance to derive degrees of freedom from.
    degrees_of_freedom: float | None


@dataclass(frozen=True)
class Headline:
    metric: str
    per_clip: tuple[ClipHeadline, ...]
    pooled: PooledHeadline


@dataclass(frozen=True)
class Insights:
    metric: str
    # Insight name -> clip -> value.
    per_clip: Mapping[str, Mapping[str, float]]
    # Insight name -> mean over clips of its per-clip value, clips weighted equally as in the headline. So the pooled
    # "std" is the mean of the clip stds, not a pooled standard deviation.
    pooled: Mapping[str, float]


def _t_quantile(probability: float, degrees_of_freedom: float) -> float:
    """Typed adapter for scipy, which ships no type information."""
    return float(cast(float, stats.t.ppf(probability, degrees_of_freedom)))  # pyright: ignore[reportUnknownMemberType]


def _half_width(variance_of_mean: float, degrees_of_freedom: float) -> float:
    upper_probability = 1 - (1 - CONFIDENCE_LEVEL) / 2
    return _t_quantile(upper_probability, degrees_of_freedom) * math.sqrt(variance_of_mean)


def _welch_satterthwaite(variances: Sequence[float], seed_counts: Sequence[int]) -> float:
    """Degrees of freedom of a sum of independent variance estimates, each from `seed_count` values."""
    return sum(variances) ** 2 / sum(
        variance**2 / (seed_count - 1) for variance, seed_count in zip(variances, seed_counts, strict=True)
    )


def _checked(metric: str, values_by_clip: Mapping[str, Sequence[float]]) -> Mapping[str, npt.NDArray[np.float64]]:
    if not values_by_clip:
        raise ValueError(f"{metric}: no clips given; pass the per-seed values of every clip.")
    checked: dict[str, npt.NDArray[np.float64]] = {}
    for clip, values in values_by_clip.items():
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(
                f"{metric}, clip {clip}: NaN or infinite among the seed values. It would silently turn the mean "
                "and interval into NaN or infinity; find the run that produced it and re-evaluate or drop that seed."
            )
        if len(array) < MIN_SEEDS:
            raise ValueError(
                f"{metric}, clip {clip}: {len(array)} seed(s); an interval needs a spread, so at least {MIN_SEEDS} "
                "seeds. Run another seed for this clip."
            )
        checked[clip] = array
    return checked


def summarize_headline(metric: str, values_by_clip: Mapping[str, Sequence[float]]) -> Headline:
    """Per-clip and pooled mean with 95 % Student-t intervals. `values_by_clip` maps a clip to its per-seed values."""
    arrays = _checked(metric, values_by_clip)
    per_clip: list[ClipHeadline] = []
    variances_of_mean: list[float] = []
    for clip, array in arrays.items():
        n = len(array)
        mean = float(array.mean())
        no_spread = bool(np.ptp(array) == 0)
        variance_of_mean = 0.0 if no_spread else float(array.var(ddof=1)) / n
        half_width = _half_width(variance_of_mean, n - 1)
        per_clip.append(ClipHeadline(clip, n, MeanInterval(mean, mean - half_width, mean + half_width), no_spread))
        variances_of_mean.append(variance_of_mean)

    num_clips = len(per_clip)
    pooled_mean = sum(entry.interval.mean for entry in per_clip) / num_clips
    # Var of the mean of clip means; Welch-Satterthwaite degrees of freedom over the clips' variances of the mean.
    terms = [variance / num_clips**2 for variance in variances_of_mean]
    pooled_variance = sum(terms)
    degrees_of_freedom: float | None = None
    half_width = 0.0
    if not all(entry.no_spread for entry in per_clip):
        degrees_of_freedom = _welch_satterthwaite(terms, [entry.num_seeds for entry in per_clip])
        half_width = _half_width(pooled_variance, degrees_of_freedom)
    pooled = PooledHeadline(
        clips=tuple(entry.clip for entry in per_clip),
        seeds_per_clip=tuple(entry.num_seeds for entry in per_clip),
        interval=MeanInterval(pooled_mean, pooled_mean - half_width, pooled_mean + half_width),
        clips_without_spread=tuple(entry.clip for entry in per_clip if entry.no_spread),
        degrees_of_freedom=degrees_of_freedom,
    )
    return Headline(metric=metric, per_clip=tuple(per_clip), pooled=pooled)


def compute_insights(
    metric: str,
    values_by_clip: Mapping[str, Sequence[float]],
    custom: Mapping[str, SeedAggregation] | None = None,
) -> Insights:
    """The `INSIGHT_PRESETS` and any `custom` functions (by name) per clip and pooled. Never used for ranking."""
    arrays = _checked(metric, values_by_clip)
    custom = custom or {}
    clashes = sorted(INSIGHT_PRESETS.keys() & custom.keys())
    if clashes:
        raise ValueError(
            f"{metric}: custom insight name(s) {clashes} are already presets, so a table could not tell which "
            "function made the number. Give the custom function a different name."
        )
    functions = {**INSIGHT_PRESETS, **custom}
    per_clip = {
        name: {clip: function(array.tolist()) for clip, array in arrays.items()} for name, function in functions.items()
    }
    pooled = {name: sum(by_clip.values()) / len(by_clip) for name, by_clip in per_clip.items()}
    return Insights(metric=metric, per_clip=per_clip, pooled=pooled)


@dataclass(frozen=True)
class ClipResults:
    """One method's per-seed values on one clip, the zero-action floor of each seed's eval, and the eval's task hash."""

    task_hash: str
    values: Sequence[float]
    floors: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.floors):
            raise ValueError(
                f"{len(self.values)} values but {len(self.floors)} floors; each seed's evaluation gives one of each. "
                "Pass the floor of every seed's evaluation."
            )


@dataclass(frozen=True)
class MethodResults:
    name: str
    # Clip -> its results.
    clips: Mapping[str, ClipResults]


@dataclass(frozen=True)
class ClipDifference:
    clip: str
    # Mean of the first method minus mean of the second.
    interval: MeanInterval
    # None when neither method has a spread on this clip.
    degrees_of_freedom: float | None


@dataclass(frozen=True)
class PooledDifference:
    clips: tuple[str, ...]
    interval: MeanInterval
    degrees_of_freedom: float | None


@dataclass(frozen=True)
class Comparison:
    metric: str
    methods: tuple[str, str]
    per_clip: tuple[ClipDifference, ...]
    pooled: PooledDifference
    # Clips that only one method has (a diverged or missing run); they are left out of the comparison.
    clips_only_in: Mapping[str, tuple[str, ...]]


def _variance_of_mean(values: npt.NDArray[np.float64]) -> float:
    return 0.0 if np.ptp(values) == 0 else float(values.var(ddof=1)) / len(values)


def _interval(mean: float, variances: Sequence[float], seed_counts: Sequence[int]) -> tuple[MeanInterval, float | None]:
    """The interval of a mean whose variance is the sum of `variances`; no width when all of them are zero."""
    if not any(variances):
        return MeanInterval(mean, mean, mean), None
    degrees_of_freedom = _welch_satterthwaite(variances, seed_counts)
    half_width = _half_width(sum(variances), degrees_of_freedom)
    return MeanInterval(mean, mean - half_width, mean + half_width), degrees_of_freedom


def _floor_range(first: ClipResults, second: ClipResults) -> tuple[float, float]:
    floors = [*first.floors, *second.floors]
    return min(floors), max(floors)


def normalized_by_floor(method: MethodResults) -> MethodResults:
    """Each seed's value as (value - floor) / (1 - floor) with its own eval's floor: 0 is the zero-action policy, 1 is
    perfect tracking. Not clipped, since a policy can score below the floor."""
    return MethodResults(
        method.name,
        {
            clip: ClipResults(
                results.task_hash,
                [(value - floor) / (1 - floor) for value, floor in zip(results.values, results.floors, strict=True)],
                results.floors,
            )
            for clip, results in method.clips.items()
        },
    )


def compare_methods(metric: str, first: MethodResults, second: MethodResults) -> Comparison:
    """`first` minus `second`, per clip and pooled over the clips both have."""
    name_a, clips_a, name_b, clips_b = first.name, first.clips, second.name, second.clips
    shared = [clip for clip in clips_a if clip in clips_b]
    if not shared:
        raise ValueError(
            f"{metric}: {name_a} and {name_b} have no clip in common ({sorted(clips_a)} vs {sorted(clips_b)}), so "
            "there is nothing to compare. Evaluate both methods on the same clips."
        )
    differing = [clip for clip in shared if clips_a[clip].task_hash != clips_b[clip].task_hash]
    if differing:
        raise ValueError(
            f"{metric}: {name_a} and {name_b} were evaluated on different tasks for clip(s) {differing} (task hashes "
            "differ), so their scores measure different things. Evaluate both on the same env config, or compare "
            "only clips with equal task hashes."
        )
    floor_ranges = {clip: _floor_range(clips_a[clip], clips_b[clip]) for clip in shared}
    unequal_floors = {clip: (low, high) for clip, (low, high) in floor_ranges.items() if high - low > FLOOR_TOLERANCE}
    if unequal_floors:
        raise ValueError(
            f"{metric}: the zero-action floors of {name_a} and {name_b} differ by more than {FLOOR_TOLERANCE} within "
            f"clip(s) {unequal_floors} (min, max). The floor is deterministic for one code version and device, so the "
            "evaluations ran on different dexmachina code or devices and their scores are not comparable. Evaluate "
            "all final checkpoints of the cell on one pinned dexmachina commit and device."
        )
    arrays_a = _checked(f"{metric} ({name_a})", {clip: clips_a[clip].values for clip in shared})
    arrays_b = _checked(f"{metric} ({name_b})", {clip: clips_b[clip].values for clip in shared})

    per_clip: list[ClipDifference] = []
    pooled_variances: list[float] = []
    pooled_seed_counts: list[int] = []
    num_clips = len(shared)
    for clip in shared:
        a, b = arrays_a[clip], arrays_b[clip]
        variances = [_variance_of_mean(a), _variance_of_mean(b)]
        seed_counts = [len(a), len(b)]
        interval, degrees_of_freedom = _interval(float(a.mean() - b.mean()), variances, seed_counts)
        per_clip.append(ClipDifference(clip, interval, degrees_of_freedom))
        pooled_variances += [variance / num_clips**2 for variance in variances]
        pooled_seed_counts += seed_counts

    pooled_mean = sum(entry.interval.mean for entry in per_clip) / num_clips
    pooled_interval, pooled_degrees_of_freedom = _interval(pooled_mean, pooled_variances, pooled_seed_counts)
    return Comparison(
        metric=metric,
        methods=(name_a, name_b),
        per_clip=tuple(per_clip),
        pooled=PooledDifference(tuple(shared), pooled_interval, pooled_degrees_of_freedom),
        clips_only_in={
            name_a: tuple(clip for clip in clips_a if clip not in clips_b),
            name_b: tuple(clip for clip in clips_b if clip not in clips_a),
        },
    )

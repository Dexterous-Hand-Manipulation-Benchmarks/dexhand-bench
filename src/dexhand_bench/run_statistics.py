"""Summaries of one metric of one method over seeds and clips.

The headline (per-clip and pooled mean with a Student-t interval) is the only part used to rank methods. Insights
(median, IQR, std, custom functions) are computed separately and recorded under their names, for tracking only.

Reading the intervals:
- The per-clip intervals are descriptive. With few seeds and values near 0 or 1 they cover the true mean far less
  often than 95 % of the time. Only the pooled interval is meant for ranking.
- The pooled interval reflects the variance between seeds on exactly this set of clips. Adding or removing clips
  changes what is estimated, not only how precisely.
- Each seed must contribute one value from the same evaluation protocol for every method: final checkpoint, same
  evaluation episodes, same environment config and same step budget.
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
        degrees_of_freedom = pooled_variance**2 / sum(
            term**2 / (entry.num_seeds - 1) for term, entry in zip(terms, per_clip, strict=True)
        )
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

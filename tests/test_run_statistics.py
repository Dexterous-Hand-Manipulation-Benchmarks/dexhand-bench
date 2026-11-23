import dataclasses
import math

import numpy as np
import pytest
from scipy import stats  # pyright: ignore[reportMissingTypeStubs]

from dexhand_bench.run_statistics import (
    ClipHeadline,
    ClipResults,
    Headline,
    MethodResults,
    PooledHeadline,
    compare_methods,
    compute_insights,
    normalized_by_floor,
    summarize_headline,
)

METRIC = "auc_add"
# 97.5 % quantiles of Student's t with 2 and 1 degrees of freedom, from a t table.
T_975_DF2 = 4.302653
T_975_DF1 = 12.706205
COVERAGE_REPETITIONS = 2000
COVERAGE_SEEDS = 5
MIN_COVERAGE = 0.93
POOLED_COVERAGE_REPETITIONS = 3000
POOLED_MIN_COVERAGE = 0.92
POOLED_SEEDS = 3
# Beta(a, b) per clip: different spreads and means near the bounds, as for success rates.
POOLED_CLIP_BETAS = ((1.0, 4.0), (8.0, 2.0), (0.5, 0.5), (3.0, 3.0))
TASK = "task0"
# Hand-computed for first [1, 2, 3] vs second [0, 0, 3]: variances of the mean 1/3 and 1, so the difference 1 has
# variance 4/3 and Welch degrees of freedom (4/3)^2 / ((1/3)^2 / 2 + 1^2 / 2) = 3.2.
WELCH_DF = 3.2
FLOOR = 0.2


def test_the_clip_interval_is_the_t_interval_on_three_values() -> None:
    # Values 1, 2, 6: mean 3; deviations -2, -1, 3 -> sum of squares 14, s^2 = 14 / 2 = 7; s / sqrt(3) = sqrt(7 / 3).
    headline = summarize_headline(METRIC, {"clip": [1.0, 2.0, 6.0]})
    clip = headline.per_clip[0]
    half_width = T_975_DF2 * math.sqrt(7 / 3)
    assert clip.num_seeds == 3
    assert clip.interval.mean == pytest.approx(3.0)
    assert clip.interval.low == pytest.approx(3.0 - half_width, abs=1e-4)
    assert clip.interval.high == pytest.approx(3.0 + half_width, abs=1e-4)


def test_the_pooled_mean_weights_clips_equally_with_unequal_seed_counts() -> None:
    # Clip means 2 (two seeds) and 10 (four seeds): the pooled mean is (2 + 10) / 2 = 6, not the seed-weighted 7.333.
    headline = summarize_headline(METRIC, {"a": [1.0, 3.0], "b": [8.0, 9.0, 11.0, 12.0]})
    assert headline.pooled.interval.mean == pytest.approx(6.0)
    assert headline.pooled.clips == ("a", "b")
    assert headline.pooled.seeds_per_clip == (2, 4)


def test_the_pooled_variance_and_degrees_of_freedom_on_two_clips() -> None:
    # a = [1, 3]: s^2 = 2, n = 2 -> s^2 / n = 1.   b = [0, 2, 4, 6]: s^2 = (9 + 1 + 1 + 9) / 3 = 20 / 3, n = 4 -> 5 / 3.
    # Var = (1 + 5/3) / K^2 = (8/3) / 4 = 2/3.
    # Per-clip terms s^2 / (n K^2): 1/4 and 5/12. Welch-Satterthwaite:
    # df = (2/3)^2 / ((1/4)^2 / 1 + (5/12)^2 / 3) = (4/9) / (1/16 + 25/432) = (4/9) / (52/432) = 192 / 52.
    terms = (1 / 4, 5 / 12)
    expected_df = (2 / 3) ** 2 / (terms[0] ** 2 / 1 + terms[1] ** 2 / 3)
    assert expected_df == pytest.approx(192 / 52)
    headline = summarize_headline(METRIC, {"a": [1.0, 3.0], "b": [0.0, 2.0, 4.0, 6.0]})
    assert headline.pooled.degrees_of_freedom == pytest.approx(expected_df)
    t_quantile = float(stats.t.ppf(0.975, expected_df))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    # Pooled mean (2 + 3) / 2 = 2.5.
    assert headline.pooled.interval.mean == pytest.approx(2.5)
    assert headline.pooled.interval.high - 2.5 == pytest.approx(t_quantile * math.sqrt(2 / 3))
    assert 2.5 - headline.pooled.interval.low == pytest.approx(t_quantile * math.sqrt(2 / 3))


def test_a_clip_with_one_seed_is_rejected_naming_metric_and_clip() -> None:
    with pytest.raises(ValueError, match=r"auc_add, clip lonely.*Run another seed"):
        summarize_headline(METRIC, {"ok": [1.0, 2.0], "lonely": [1.0]})


def test_an_infinite_value_is_rejected_naming_metric_and_clip() -> None:
    with pytest.raises(ValueError, match=r"auc_add, clip broken.*NaN or infinite"):
        summarize_headline(METRIC, {"ok": [1.0, 2.0], "broken": [1.0, float("inf"), 3.0]})


def test_clips_where_every_seed_agrees_have_no_spread_and_pooled_has_no_degrees_of_freedom() -> None:
    headline = summarize_headline(METRIC, {"a": [0.1] * 3, "b": [0.7] * 3})
    assert [clip.no_spread for clip in headline.per_clip] == [True, True]
    assert headline.pooled.clips_without_spread == ("a", "b")
    assert headline.pooled.degrees_of_freedom is None
    assert headline.pooled.interval.low == headline.pooled.interval.high == pytest.approx(0.4)


def test_all_zero_seeds_have_no_spread() -> None:
    clip = summarize_headline(METRIC, {"a": [0.0, 0.0, 0.0]}).per_clip[0]
    assert clip.no_spread
    assert clip.interval.low == clip.interval.high == 0.0


def test_a_mixed_case_lists_the_clip_without_spread_and_still_has_degrees_of_freedom() -> None:
    # Only b has variance: s^2 / (n K^2) = 2 / (2 * 4) = 1/4 for b = [1, 3].
    # One term, so df = (1/4)^2 / ((1/4)^2 / 1) = 1, and the half-width is t_{0.975,1} * sqrt(1/4).
    headline = summarize_headline(METRIC, {"a": [5.0, 5.0], "b": [1.0, 3.0]})
    assert headline.pooled.clips_without_spread == ("a",)
    assert headline.pooled.degrees_of_freedom == pytest.approx(1.0)
    assert headline.pooled.interval.mean == pytest.approx(3.5)
    assert headline.pooled.interval.high == pytest.approx(3.5 + T_975_DF1 * 0.5)


def test_a_custom_insight_named_like_a_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\['median'\].*already presets"):
        compute_insights(METRIC, {"a": [1.0, 2.0]}, custom={"median": min})


def test_a_nan_is_rejected_naming_metric_and_clip() -> None:
    with pytest.raises(ValueError, match=r"auc_add, clip broken.*NaN"):
        summarize_headline(METRIC, {"ok": [1.0, 2.0], "broken": [1.0, float("nan"), 3.0]})


def test_insights_carry_custom_names_and_stay_out_of_the_headline() -> None:
    values = {"a": [1.0, 2.0, 3.0, 10.0], "b": [2.0, 4.0]}
    insights = compute_insights(METRIC, values, custom={"worst": min})
    # Clip a: median (2 + 3) / 2 = 2.5; clip b: median 3. Worst seed: 1 and 2, pooled (1 + 2) / 2 = 1.5.
    assert insights.per_clip["median"] == pytest.approx({"a": 2.5, "b": 3.0})
    assert insights.per_clip["worst"] == {"a": 1.0, "b": 2.0}
    assert insights.pooled["median"] == pytest.approx((2.5 + 3.0) / 2)
    assert insights.pooled["worst"] == pytest.approx(1.5)
    assert {"median", "iqr", "std", "worst"} <= set(insights.pooled)
    headline = summarize_headline(METRIC, values)
    insight_names = {"median", "iqr", "std", "worst"}
    for headline_type in (Headline, ClipHeadline, PooledHeadline):
        assert insight_names.isdisjoint(field.name for field in dataclasses.fields(headline_type))
    assert headline.metric == METRIC


def test_the_presets_on_four_seeds() -> None:
    # Values 1, 2, 3, 10: quartiles by linear interpolation are 1.75 and 4.75 -> IQR 3; mean 4, sum of squares
    # 9 + 4 + 1 + 36 = 50 -> s = sqrt(50 / 3).
    per_clip = compute_insights(METRIC, {"a": [1.0, 2.0, 3.0, 10.0]}).per_clip
    assert per_clip["iqr"]["a"] == pytest.approx(3.0)
    assert per_clip["std"]["a"] == pytest.approx(math.sqrt(50 / 3))


def test_the_ninety_five_percent_interval_covers_the_true_mean_in_simulation() -> None:
    rng = np.random.default_rng(0)
    true_mean = 0.7
    covered = 0
    for _ in range(COVERAGE_REPETITIONS):
        seeds = rng.normal(true_mean, 0.1, size=COVERAGE_SEEDS).tolist()
        interval = summarize_headline(METRIC, {"clip": seeds}).per_clip[0].interval
        covered += interval.low <= true_mean <= interval.high
    assert covered / COVERAGE_REPETITIONS >= MIN_COVERAGE


def test_the_pooled_interval_covers_the_mean_of_clip_means_in_simulation() -> None:
    rng = np.random.default_rng(1)
    true_mean = sum(a / (a + b) for a, b in POOLED_CLIP_BETAS) / len(POOLED_CLIP_BETAS)
    covered = 0
    for _ in range(POOLED_COVERAGE_REPETITIONS):
        values = {
            f"clip{index}": rng.beta(a, b, size=POOLED_SEEDS).tolist() for index, (a, b) in enumerate(POOLED_CLIP_BETAS)
        }
        interval = summarize_headline(METRIC, values).pooled.interval
        covered += interval.low <= true_mean <= interval.high
    assert covered / POOLED_COVERAGE_REPETITIONS >= POOLED_MIN_COVERAGE


def _method(name: str, clips: dict[str, list[float]], task_hash: str = TASK, floor: float = FLOOR) -> MethodResults:
    return MethodResults(
        name, {clip: ClipResults(task_hash, values, [floor] * len(values)) for clip, values in clips.items()}
    )


def _t_975(degrees_of_freedom: float) -> float:
    return float(stats.t.ppf(0.975, degrees_of_freedom))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def test_the_clip_difference_is_first_minus_second_with_a_welch_interval() -> None:
    comparison = compare_methods(METRIC, _method("a", {"c1": [1, 2, 3]}), _method("b", {"c1": [0, 0, 3]}))
    (clip,) = comparison.per_clip
    half_width = _t_975(WELCH_DF) * math.sqrt(4 / 3)
    assert clip.interval.mean == pytest.approx(1.0)
    assert clip.degrees_of_freedom == pytest.approx(WELCH_DF)
    assert (clip.interval.low, clip.interval.high) == pytest.approx((1 - half_width, 1 + half_width))
    assert comparison.methods == ("a", "b")


def test_the_pooled_difference_is_the_mean_of_the_clip_differences() -> None:
    first = _method("a", {"c1": [1, 2, 3], "c2": [0.5, 0.5, 0.5]})
    second = _method("b", {"c1": [0, 0, 3], "c2": [0.5, 0.5, 0.5]})
    pooled = compare_methods(METRIC, first, second).pooled
    # Mean of the differences 1 and 0; variance (1/3 + 1) / 2^2 = 1/3; Welch df (1/3)^2 / ((1/12)^2/2 + (1/4)^2/2).
    half_width = _t_975(WELCH_DF) * math.sqrt(1 / 3)
    assert pooled.clips == ("c1", "c2")
    assert pooled.interval.mean == pytest.approx(0.5)
    assert pooled.degrees_of_freedom == pytest.approx(WELCH_DF)
    assert (pooled.interval.low, pooled.interval.high) == pytest.approx((0.5 - half_width, 0.5 + half_width))


def test_clips_of_only_one_method_are_left_out_and_listed() -> None:
    first = _method("a", {"c1": [1, 2], "only_a": [1, 1]})
    second = _method("b", {"c1": [0, 1], "only_b": [0, 0]})
    comparison = compare_methods(METRIC, first, second)
    assert comparison.pooled.clips == ("c1",)
    assert comparison.clips_only_in == {"a": ("only_a",), "b": ("only_b",)}


def test_without_any_spread_the_difference_has_no_interval() -> None:
    comparison = compare_methods(METRIC, _method("a", {"c1": [1, 1]}), _method("b", {"c1": [0, 0]}))
    assert comparison.pooled.interval == comparison.per_clip[0].interval
    assert (comparison.pooled.interval.low, comparison.pooled.interval.high) == (1.0, 1.0)
    assert comparison.pooled.degrees_of_freedom is None


def test_methods_evaluated_on_different_tasks_are_refused() -> None:
    first = _method("a", {"c1": [1, 2]})
    second = _method("b", {"c1": [0, 1]}, task_hash="other")
    with pytest.raises(ValueError, match=r"different tasks for clip\(s\) \['c1'\]"):
        compare_methods(METRIC, first, second)


def test_methods_whose_floors_differ_within_a_clip_are_refused() -> None:
    first = _method("a", {"c1": [1, 2]})
    second = _method("b", {"c1": [0, 1]}, floor=FLOOR + 0.01)
    with pytest.raises(ValueError, match=r"floors of a and b differ .* clip\(s\) \{'c1'"):
        compare_methods(METRIC, first, second)


def test_normalizing_maps_the_floor_to_zero_and_perfect_tracking_to_one() -> None:
    normalized = normalized_by_floor(_method("a", {"c1": [FLOOR, 1.0, 0.0]}))
    assert list(normalized.clips["c1"].values) == pytest.approx([0.0, 1.0, -FLOOR / (1 - FLOOR)])


def test_methods_without_a_common_clip_are_refused() -> None:
    with pytest.raises(ValueError, match="no clip in common"):
        compare_methods(METRIC, _method("a", {"c1": [1, 2]}), _method("b", {"c2": [0, 1]}))


def test_a_method_with_a_single_seed_on_a_shared_clip_is_refused() -> None:
    with pytest.raises(ValueError, match=r"auc_add \(b\), clip c1: 1 seed"):
        compare_methods(METRIC, _method("a", {"c1": [1, 2]}), _method("b", {"c1": [0]}))


def test_the_pooled_difference_covers_the_true_difference_in_simulation() -> None:
    rng = np.random.default_rng(2)
    # The second method has a higher mean and a different spread on every clip.
    second_betas = tuple((a + 1.0, b) for a, b in POOLED_CLIP_BETAS)
    true_difference = sum(
        a / (a + b) - c / (c + d) for (a, b), (c, d) in zip(POOLED_CLIP_BETAS, second_betas, strict=True)
    ) / len(POOLED_CLIP_BETAS)
    covered = 0
    for _ in range(POOLED_COVERAGE_REPETITIONS):
        first, second = (
            _method(name, {f"clip{i}": rng.beta(a, b, size=POOLED_SEEDS).tolist() for i, (a, b) in enumerate(betas)})
            for name, betas in (("a", POOLED_CLIP_BETAS), ("b", second_betas))
        )
        interval = compare_methods(METRIC, first, second).pooled.interval
        covered += interval.low <= true_difference <= interval.high
    assert covered / POOLED_COVERAGE_REPETITIONS >= POOLED_MIN_COVERAGE

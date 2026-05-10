"""Tests for sign-aware score_improvement_ratio (feature 019).

Covers FR-012, FR-013 (US6, SC-006). Replaces the legacy abs()-trick metric.
"""

from __future__ import annotations

import polars as pl
import pytest


pytestmark = pytest.mark.unit


def _gt(ids: list[str], values: list[float], col: str = "target") -> pl.DataFrame:
    return pl.DataFrame({"ID": ids, col: values})


def test_score_improvement_higher_better_with_better_selection_positive() -> None:
    from learnm8.evaluation.metrics.enrichment import calculate_score_improvement_ratio

    pop_ids = ["A", "B", "C", "D", "E", "F"]
    pop = _gt(pop_ids, [10.0, 9.0, 8.0, 5.0, 4.0, 3.0])
    selected = {"A", "B", "C"}
    out = calculate_score_improvement_ratio(selected, pop, "target", score_direction="higher")
    assert out > 0.0


def test_score_improvement_higher_better_doubles_pop_returns_at_least_one() -> None:
    from learnm8.evaluation.metrics.enrichment import calculate_score_improvement_ratio

    # Selected mean is large compared to pop mean → ratio > 1 means >100% improvement.
    # pop_mean = 30/7 ≈ 4.29; selected_mean = 10; (10-4.29)/4.29 ≈ 1.33.
    ids = ["A", "B", "C", "D", "E", "F", "G"]
    pop = _gt(ids, [10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    out = calculate_score_improvement_ratio(
        {"A", "B", "C"}, pop, "target", score_direction="higher"
    )
    assert out > 1.0


def test_score_improvement_higher_better_with_worse_selection_negative() -> None:
    from learnm8.evaluation.metrics.enrichment import calculate_score_improvement_ratio

    pop_ids = ["A", "B", "C", "D"]
    pop = _gt(pop_ids, [5.0, 4.0, 1.0, 2.0])
    selected = {"C", "D"}
    out = calculate_score_improvement_ratio(selected, pop, "target", score_direction="higher")
    assert out < 0.0


def test_score_improvement_lower_better_docking_negative_targets() -> None:
    from learnm8.evaluation.metrics.enrichment import calculate_score_improvement_ratio

    ids = ["A", "B", "C", "D"]
    pop = _gt(ids, [-9.0, -8.0, -5.0, -4.0])  # pop_mean = -6.5
    selected = {"A", "B"}  # selected_mean = -8.5
    out = calculate_score_improvement_ratio(selected, pop, "target", score_direction="lower")
    expected = (-6.5 - (-8.5)) / abs(-6.5)
    assert out == pytest.approx(expected, rel=1e-6)
    assert out > 0.0


def test_score_improvement_lower_better_docking_doubles_negativity_returns_at_least_one() -> None:
    from learnm8.evaluation.metrics.enrichment import calculate_score_improvement_ratio

    # Docking-style: selected_mean = -8; pop_mean = -24/7 ≈ -3.43.
    # ratio = (pop_mean − selected_mean)/|pop_mean| = (−3.43+8)/3.43 ≈ 1.33 > 1.
    ids = ["A", "B", "C", "D", "E", "F", "G"]
    pop = _gt(ids, [-8.0, -8.0, -8.0, 0.0, 0.0, 0.0, 0.0])
    out = calculate_score_improvement_ratio(
        {"A", "B", "C"}, pop, "target", score_direction="lower"
    )
    assert out > 1.0


def test_score_improvement_lower_better_negative_selection_worse_negative() -> None:
    from learnm8.evaluation.metrics.enrichment import calculate_score_improvement_ratio

    ids = ["A", "B", "C", "D"]
    pop = _gt(ids, [-5.0, -4.0, -1.0, -2.0])
    selected = {"C", "D"}
    out = calculate_score_improvement_ratio(selected, pop, "target", score_direction="lower")
    assert out < 0.0


def test_score_improvement_pop_mean_zero_returns_signed_inf() -> None:
    from learnm8.evaluation.metrics.enrichment import calculate_score_improvement_ratio

    ids = ["A", "B"]
    pop = _gt(ids, [-1.0, 1.0])  # pop_mean = 0
    selected = {"A"}  # selected_mean = -1, lower_better → improvement direction is positive
    out = calculate_score_improvement_ratio(selected, pop, "target", score_direction="lower")
    import numpy as np
    assert np.isinf(out)


def test_score_improvement_pop_mean_zero_zero_selection_returns_zero() -> None:
    from learnm8.evaluation.metrics.enrichment import calculate_score_improvement_ratio

    # Feature 019: neutral baseline is 0.0 ("no improvement"), not 1.0.
    ids = ["A", "B"]
    pop = _gt(ids, [-1.0, 1.0])  # pop_mean = 0
    selected = {"A", "B"}
    out = calculate_score_improvement_ratio(selected, pop, "target", score_direction="higher")
    assert out == pytest.approx(0.0)


def test_score_improvement_empty_selection_raises_value_error() -> None:
    from learnm8.evaluation.metrics.enrichment import calculate_score_improvement_ratio

    pop = _gt(["A", "B"], [1.0, 2.0])
    with pytest.raises(ValueError) as exc_info:
        calculate_score_improvement_ratio(set(), pop, "target", score_direction="higher")
    assert "empty" in str(exc_info.value).lower()


def test_score_improvement_invalid_direction_raises() -> None:
    from learnm8.evaluation.metrics.enrichment import calculate_score_improvement_ratio

    pop = _gt(["A", "B"], [1.0, 2.0])
    with pytest.raises(ValueError):
        calculate_score_improvement_ratio({"A"}, pop, "target", score_direction="higher_better")  # type: ignore[arg-type]


def test_score_improvement_required_direction_no_default() -> None:
    """Calling without score_direction should fail (positional/kw-only required)."""
    from learnm8.evaluation.metrics.enrichment import calculate_score_improvement_ratio

    pop = _gt(["A", "B"], [1.0, 2.0])
    with pytest.raises(TypeError):
        calculate_score_improvement_ratio({"A"}, pop, "target")  # type: ignore[call-arg]


def test_batch_score_improvement_parallels_cumulative() -> None:
    from learnm8.evaluation.metrics.enrichment import (
        calculate_batch_score_improvement_ratio,
        calculate_score_improvement_ratio,
    )

    ids = ["A", "B", "C", "D"]
    pop = _gt(ids, [10.0, 5.0, 3.0, 2.0])
    selected_df = pop.filter(pl.col("ID").is_in(["A", "B"]))
    cum = calculate_score_improvement_ratio({"A", "B"}, pop, "target", score_direction="higher")
    bat = calculate_batch_score_improvement_ratio(selected_df, pop, "target", score_direction="higher")
    assert cum == pytest.approx(bat)


def test_old_function_names_removed() -> None:
    """Confirm the legacy abs()-trick functions were removed (no aliases per FR-012)."""
    enrichment = pytest.importorskip("learnm8.evaluation.metrics.enrichment")
    legacy_higher = "calculate_" + "average_score_ratio"
    legacy_batch = "calculate_batch_" + "average_score_ratio"
    assert not hasattr(enrichment, legacy_higher)
    assert not hasattr(enrichment, legacy_batch)

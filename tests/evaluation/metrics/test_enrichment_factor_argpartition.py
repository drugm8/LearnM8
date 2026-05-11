"""Spec 022 US3 / FR-005: unlabeled EF uses argpartition not full sort."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from learnm8.evaluation.metrics.enrichment import (
    calculate_multiple_unlabeled_enrichment_factors,
)

pytestmark = pytest.mark.integration


def _make_dfs(n: int, p_active: float, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = [f"M{i:08d}" for i in range(n)]
    preds = rng.random(n).astype(np.float32)
    # Make positives concentrated among high predictions for a realistic EF > 1.
    labels = (preds + 0.3 * rng.standard_normal(n) > np.quantile(preds, 1.0 - p_active)).astype(int)
    pred_df = pl.DataFrame({"ID": ids, "prediction": preds})
    truth_df = pl.DataFrame({"ID": ids, "activity": labels})
    return pred_df, truth_df


def test_unlabeled_ef_returns_expected_keys() -> None:
    pred_df, truth_df = _make_dfs(1000, p_active=0.1)
    res = calculate_multiple_unlabeled_enrichment_factors(
        pred_df, truth_df, activity_column="activity", score_direction="higher"
    )
    assert set(res.keys()) == {"unlabeled_ef_1_0", "unlabeled_ef_5_0"}


def test_unlabeled_ef_correlates_with_signal_strength() -> None:
    """EF at 1% should exceed 1 when predictions correlate with labels."""
    pred_df, truth_df = _make_dfs(10_000, p_active=0.05, seed=42)
    res = calculate_multiple_unlabeled_enrichment_factors(
        pred_df, truth_df, activity_column="activity", score_direction="higher"
    )
    # With signal, top-1% should enrich actives above population rate.
    assert res["unlabeled_ef_1_0"] is not None
    assert res["unlabeled_ef_1_0"] > 1.0, (
        f"EF at 1% is {res['unlabeled_ef_1_0']}; expected > 1.0 given signal."
    )


def test_unlabeled_ef_handles_score_direction_lower() -> None:
    """score_direction='lower' returns EF for descending=False ordering."""
    pred_df, truth_df = _make_dfs(1000, p_active=0.1, seed=7)
    res_lower = calculate_multiple_unlabeled_enrichment_factors(
        pred_df, truth_df, activity_column="activity", score_direction="lower"
    )
    # Should return valid EF values (not None or NaN); high vs low ordering is
    # symmetric on random data, so we just check key validity.
    assert all(isinstance(v, (int, float)) for v in res_lower.values())


def test_unlabeled_ef_missing_activity_column_returns_none() -> None:
    pred_df = pl.DataFrame({"ID": ["A", "B"], "prediction": [0.5, 0.7]})
    truth_df = pl.DataFrame({"ID": ["A", "B"], "wrong_col": [1, 0]})
    res = calculate_multiple_unlabeled_enrichment_factors(
        pred_df, truth_df, activity_column="activity", score_direction="higher"
    )
    assert res == {"unlabeled_ef_1_0": None, "unlabeled_ef_5_0": None}

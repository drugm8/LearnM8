"""Unit tests for the diagnostic metrics added in feature 014.

Covers `compute_selected_percentile` and `compute_prediction_entropy`.

Edge cases asserted (REQ-1..REQ-4 in specs/014-diagnostic-metrics/spec.md):
- benchmark-mode happy path with known expected values
- score_direction='lower' inverts the percentile
- empty / missing inputs return None silently (mode-neutral)
- all-identical predictions yield entropy ≈ 0
- uniform predictions yield entropy ≈ log2(n_bins)
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from learnm8.evaluation.metrics.diagnostics import (
    compute_prediction_entropy,
    compute_selected_percentile,
)

# ---------- compute_selected_percentile ---------------------------------------

@pytest.mark.unit
class TestSelectedPercentile:
    """Unit tests for compute_selected_percentile."""

    def test_top_selection_higher_better(self):
        # Pool true scores 1..10, select the top three (8, 9, 10).
        # higher-is-better: each selected percentile = (s >= pool).mean()
        # For s=10: 10/10 = 100; s=9: 9/10 = 90; s=8: 8/10 = 80.
        # Mean = 90.
        gt = pl.DataFrame({'ID': [f'm{i}' for i in range(10)],
                           'Activity': list(range(1, 11))})
        sel = pl.DataFrame({'ID': ['m9', 'm8', 'm7'],
                            'Activity': [10, 9, 8]})
        result = compute_selected_percentile(sel, gt, 'Activity', 'higher')
        assert result == pytest.approx(90.0, abs=1e-9)

    def test_top_selection_lower_better(self):
        # lower-is-better: best score = lowest. Select the lowest three (1, 2, 3).
        # percentile for selected s = (s <= pool).mean()
        # For s=1: (1<=[1..10]).mean = 10/10 = 100; s=2: 9/10 = 90; s=3: 8/10 = 80.
        # Mean = 90.
        gt = pl.DataFrame({'ID': [f'm{i}' for i in range(10)],
                           'DockingScore': list(range(1, 11))})
        sel = pl.DataFrame({'ID': ['m0', 'm1', 'm2'],
                            'DockingScore': [1, 2, 3]})
        result = compute_selected_percentile(sel, gt, 'DockingScore', 'lower')
        assert result == pytest.approx(90.0, abs=1e-9)

    def test_random_selection_centers_on_50(self):
        # Selecting the median compound should give ~50th percentile.
        gt = pl.DataFrame({'ID': [f'm{i}' for i in range(11)],
                           'Activity': list(range(11))})
        sel = pl.DataFrame({'ID': ['m5'], 'Activity': [5]})
        # higher-better: (5 >= [0..10]).mean = 6/11 ≈ 54.5
        result = compute_selected_percentile(sel, gt, 'Activity', 'higher')
        assert result == pytest.approx(6 / 11 * 100, abs=1e-6)

    def test_returns_value_in_range_0_100(self):
        rng = np.random.default_rng(42)
        gt = pl.DataFrame({'ID': [f'm{i}' for i in range(1000)],
                           'Activity': rng.normal(size=1000)})
        sel = gt.sample(n=50, seed=7)
        result = compute_selected_percentile(sel, gt, 'Activity', 'higher')
        assert result is not None
        assert 0.0 <= result <= 100.0

    def test_missing_ground_truth_returns_none(self):
        sel = pl.DataFrame({'ID': ['a'], 'Activity': [1.0]})
        assert compute_selected_percentile(sel, None, 'Activity', 'higher') is None

    def test_empty_selection_returns_none(self):
        gt = pl.DataFrame({'ID': ['a', 'b'], 'Activity': [1.0, 2.0]})
        sel = pl.DataFrame(schema={'ID': pl.Utf8, 'Activity': pl.Float64})
        assert compute_selected_percentile(sel, gt, 'Activity', 'higher') is None

    def test_empty_ground_truth_returns_none(self):
        gt = pl.DataFrame(schema={'ID': pl.Utf8, 'Activity': pl.Float64})
        sel = pl.DataFrame({'ID': ['a'], 'Activity': [1.0]})
        assert compute_selected_percentile(sel, gt, 'Activity', 'higher') is None

    def test_target_col_missing_returns_none(self):
        gt = pl.DataFrame({'ID': ['a', 'b'], 'OtherCol': [1.0, 2.0]})
        sel = pl.DataFrame({'ID': ['a'], 'Activity': [1.0]})
        assert compute_selected_percentile(sel, gt, 'Activity', 'higher') is None

    def test_handles_nan_scores(self):
        # NaNs in either pool or selection are dropped; computation proceeds.
        gt = pl.DataFrame({'ID': [f'm{i}' for i in range(5)],
                           'Activity': [1.0, 2.0, 3.0, 4.0, 5.0]})
        sel = pl.DataFrame({'ID': ['a', 'b'], 'Activity': [5.0, float('nan')]})
        # Only the 5.0 selection counts → percentile = 5/5 = 100
        result = compute_selected_percentile(sel, gt, 'Activity', 'higher')
        assert result == pytest.approx(100.0, abs=1e-9)


# ---------- compute_prediction_entropy ----------------------------------------

@pytest.mark.unit
class TestPredictionEntropy:
    """Unit tests for compute_prediction_entropy."""

    def test_uniform_predictions_high_entropy(self):
        # Predictions spread uniformly → entropy ≈ log2(n_bins).
        rng = np.random.default_rng(0)
        pool = pl.DataFrame({
            'ID': [f'm{i}' for i in range(10_000)],
            'prediction': rng.uniform(0.0, 1.0, size=10_000),
        })
        result = compute_prediction_entropy(pool, cumulative_selected_ids=set(),
                                            n_bins=50)
        assert result is not None
        # log2(50) ≈ 5.64. Allow generous tolerance for sampling noise.
        assert result == pytest.approx(math.log2(50), abs=0.2)

    def test_all_identical_predictions_low_entropy(self):
        # All predictions equal → density histogram has all mass in one bin.
        # Entropy should be very small (smoothing keeps it from being exactly 0).
        pool = pl.DataFrame({
            'ID': [f'm{i}' for i in range(1000)],
            'prediction': [0.5] * 1000,
        })
        result = compute_prediction_entropy(pool, cumulative_selected_ids=set(),
                                            n_bins=50)
        assert result is not None
        assert result < 0.1, f"all-identical predictions should give near-zero entropy, got {result}"

    def test_filters_to_unlabeled_subset(self):
        # If half the pool is "labeled" (in cumulative_selected_ids), entropy is
        # computed over the remaining half only.
        pool = pl.DataFrame({
            'ID': [f'm{i}' for i in range(10)],
            'prediction': [0.0] * 5 + [1.0] * 5,
        })
        # Mark the first five (all-zero predictions) as already labeled.
        # The unlabeled subset is constant 1.0 → near-zero entropy.
        labeled_ids = {f'm{i}' for i in range(5)}
        result = compute_prediction_entropy(pool, cumulative_selected_ids=labeled_ids,
                                            n_bins=50)
        assert result is not None
        assert result < 0.1

    def test_no_filter_when_selected_ids_none(self):
        pool = pl.DataFrame({
            'ID': [f'm{i}' for i in range(100)],
            'prediction': [0.5] * 100,
        })
        result = compute_prediction_entropy(pool, cumulative_selected_ids=None,
                                            n_bins=50)
        assert result is not None

    def test_missing_pool_returns_none(self):
        assert compute_prediction_entropy(None, cumulative_selected_ids=set()) is None

    def test_empty_pool_returns_none(self):
        empty = pl.DataFrame(schema={'ID': pl.Utf8, 'prediction': pl.Float64})
        assert compute_prediction_entropy(empty, cumulative_selected_ids=set()) is None

    def test_all_labeled_returns_none(self):
        # Every compound in the pool is also in cumulative_selected_ids.
        pool = pl.DataFrame({
            'ID': [f'm{i}' for i in range(5)],
            'prediction': [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        all_labeled = {f'm{i}' for i in range(5)}
        result = compute_prediction_entropy(pool, cumulative_selected_ids=all_labeled,
                                            n_bins=50)
        assert result is None

    def test_missing_prediction_column_returns_none(self):
        pool = pl.DataFrame({'ID': ['a', 'b'], 'OtherCol': [1.0, 2.0]})
        assert compute_prediction_entropy(pool, cumulative_selected_ids=set()) is None

    def test_single_compound_pool_no_crash(self):
        pool = pl.DataFrame({'ID': ['only'], 'prediction': [0.42]})
        # Should return a value (not None) and not crash.
        result = compute_prediction_entropy(pool, cumulative_selected_ids=set(),
                                            n_bins=50)
        assert result is not None
        assert result >= 0.0

    def test_returns_non_negative(self):
        rng = np.random.default_rng(123)
        for _ in range(5):
            preds = rng.normal(size=500)
            pool = pl.DataFrame({'ID': [f'm{i}' for i in range(500)],
                                 'prediction': preds})
            result = compute_prediction_entropy(pool, cumulative_selected_ids=set(),
                                                n_bins=50)
            assert result is not None
            assert result >= 0.0

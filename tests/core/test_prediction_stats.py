"""Tests for PredictionStats (streaming metrics boundary)."""

import numpy as np
import polars as pl
import pytest

from learnm8.core.prediction_stats import PredictionStats

pytestmark = pytest.mark.unit


def _legacy_pred_block(predictions, uncertainties):
    """Reproduce the legacy inline reductions from _calculate_cycle_metrics."""
    out = {}
    p = predictions
    out['prediction_mean'] = float(np.nanmean(p, dtype=np.float64))
    out['prediction_std'] = float(np.nanstd(p, dtype=np.float64))
    out['prediction_min'] = float(np.nanmin(p))
    out['prediction_max'] = float(np.nanmax(p))
    out['prediction_median'] = float(np.nanmedian(p))
    if uncertainties is not None and len(uncertainties) > 0:
        out['uncertainty_mean'] = float(np.nanmean(uncertainties, dtype=np.float64))
        out['uncertainty_std'] = float(np.nanstd(uncertainties, dtype=np.float64))
    return out


class TestFromArraysLegacyParity:
    def test_matches_legacy_inline_reductions(self):
        rng = np.random.default_rng(0)
        preds = rng.normal(size=1000)
        unc = rng.uniform(0, 1, size=1000)
        stats = PredictionStats.from_arrays(preds, unc)
        legacy = _legacy_pred_block(preds, unc)
        assert stats.pred_mean == legacy['prediction_mean']
        assert stats.pred_std == legacy['prediction_std']
        assert stats.pred_min == legacy['prediction_min']
        assert stats.pred_max == legacy['prediction_max']
        assert stats.pred_median == legacy['prediction_median']
        assert stats.unc_mean == legacy['uncertainty_mean']
        assert stats.unc_std == legacy['uncertainty_std']
        assert stats.has_uncertainty is True

    def test_no_uncertainty(self):
        stats = PredictionStats.from_arrays(np.array([1.0, 2.0, 3.0]), None)
        assert stats.has_uncertainty is False
        assert stats.unc_mean is None

    def test_empty_predictions(self):
        stats = PredictionStats.from_arrays(np.array([]), None)
        assert stats.pred_mean is None
        assert stats.count == 0


class TestFromLazyParity:
    @pytest.mark.parametrize('n', [999, 1000])  # odd/even for median interpolation
    def test_lazy_matches_arrays(self, n, tmp_path):
        rng = np.random.default_rng(1)
        preds = rng.normal(size=n).astype(np.float32)
        unc = rng.uniform(0, 1, size=n).astype(np.float32)
        df = pl.DataFrame({'ID': [f'c{i}' for i in range(n)], 'prediction': preds,
                           'uncertainty': unc})
        pq = tmp_path / 'p.parquet'
        df.write_parquet(pq)

        from_arr = PredictionStats.from_arrays(
            preds.astype(np.float64), unc.astype(np.float64)
        )
        from_lazy = PredictionStats.from_lazy(
            pl.scan_parquet(pq), with_uncertainty=True
        )
        # float32 storage → compute in f64 both sides; allow tiny float tolerance.
        assert from_lazy.count == n
        np.testing.assert_allclose(from_lazy.pred_mean, from_arr.pred_mean, rtol=1e-6)
        np.testing.assert_allclose(from_lazy.pred_std, from_arr.pred_std, rtol=1e-6)
        np.testing.assert_allclose(from_lazy.pred_median, from_arr.pred_median, rtol=1e-6)
        np.testing.assert_allclose(from_lazy.pred_min, from_arr.pred_min, rtol=1e-6)
        np.testing.assert_allclose(from_lazy.pred_max, from_arr.pred_max, rtol=1e-6)
        np.testing.assert_allclose(from_lazy.unc_mean, from_arr.unc_mean, rtol=1e-6)

    def test_lazy_without_uncertainty(self, tmp_path):
        df = pl.DataFrame({'ID': ['a', 'b'], 'prediction': [1.0, 2.0]})
        pq = tmp_path / 'p.parquet'
        df.write_parquet(pq)
        stats = PredictionStats.from_lazy(pl.scan_parquet(pq), with_uncertainty=False)
        assert stats.has_uncertainty is False
        assert stats.unc_mean is None
        assert stats.pred_mean == 1.5

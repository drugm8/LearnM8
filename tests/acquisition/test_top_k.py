"""Tests for top-k acquisition."""

import numpy as np
import polars as pl
import pytest

from learnm8.acquisition import TopKAcquisition


def _pool_with_predictions(compounds: pl.DataFrame, predictions: list[float] | np.ndarray) -> pl.DataFrame:
    return compounds.with_columns(pl.Series('prediction', predictions))


@pytest.mark.unit
class TestTopKAcquisition:
    def test_topk_selects_requested_compounds_from_top_fraction(self, small_real_compounds):
        compounds = _pool_with_predictions(
            small_real_compounds.head(10).clone(),
            [0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0],
        )

        selected = TopKAcquisition(k_fraction=0.3).select(compounds, n_select=2)
        top_ids = set(compounds.sort('prediction', descending=True).head(3).get_column('ID').to_list())

        assert len(selected) == 2
        assert set(selected.get_column('ID').to_list()).issubset(top_ids)
        assert set(selected.get_column('acquisition_score').to_list()) == set(selected.get_column('prediction').to_list())

    def test_topk_selects_requested_compounds_from_bottom_fraction_when_score_direction_is_lower(self, small_real_compounds):
        compounds = _pool_with_predictions(
            small_real_compounds.head(10).clone(),
            [0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0],
        )

        selected = TopKAcquisition(k_fraction=0.3, score_direction='lower').select(compounds, n_select=2)
        bottom_ids = set(compounds.sort('prediction').head(3).get_column('ID').to_list())

        assert len(selected) == 2
        assert set(selected.get_column('ID').to_list()).issubset(bottom_ids)

    def test_topk_is_effectively_deterministic_when_candidate_pool_equals_batch_size(self, small_real_compounds):
        compounds = _pool_with_predictions(
            small_real_compounds.head(8).clone(),
            np.linspace(0.1, 0.8, 8),
        )

        acquisition = TopKAcquisition(k_fraction=0.25)
        first = acquisition.select(compounds, n_select=2)
        second = acquisition.select(compounds, n_select=2)

        assert first.get_column('ID').to_list() == second.get_column('ID').to_list()

    def test_topk_returns_all_compounds_when_requested_batch_exceeds_pool(self, small_real_compounds):
        compounds = _pool_with_predictions(small_real_compounds.head(4).clone(), [0.1, 0.3, 0.2, 0.4])

        selected = TopKAcquisition(k_fraction=0.5).select(compounds, n_select=10)

        assert len(selected) == 4

    def test_topk_preserves_requested_batch_size_when_scores_are_identical(self, small_real_compounds):
        compounds = _pool_with_predictions(small_real_compounds.head(6).clone(), np.ones(6))

        selected = TopKAcquisition(k_fraction=0.5).select(compounds, n_select=3)

        assert len(selected) == 3
        assert np.allclose(selected.get_column('acquisition_score').to_numpy(), 1.0)

    def test_topk_rejects_empty_pool(self, empty_compounds):
        with pytest.raises(ValueError, match='compounds DataFrame is empty'):
            TopKAcquisition().select(empty_compounds, n_select=1)

    def test_topk_rejects_invalid_fraction(self):
        with pytest.raises(ValueError, match='k_fraction must be between 0 and 1'):
            TopKAcquisition(k_fraction=0.0)

    def test_topk_rejects_missing_prediction_column(self, small_real_compounds):
        with pytest.raises(ValueError, match='Missing required columns'):
            TopKAcquisition().select(small_real_compounds.head(5).clone(), n_select=2)

    def test_topk_rejects_non_positive_batch_size(self, small_real_compounds):
        compounds = _pool_with_predictions(small_real_compounds.head(5).clone(), np.linspace(0.1, 0.5, 5))

        with pytest.raises(ValueError, match='n_select must be positive'):
            TopKAcquisition().select(compounds, n_select=0)

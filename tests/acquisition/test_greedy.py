"""Tests for greedy acquisition."""

import numpy as np
import polars as pl
import pytest

from learnm8.acquisition import GreedyAcquisition


def _pool_with_predictions(compounds: pl.DataFrame, predictions: list[float] | np.ndarray) -> pl.DataFrame:
    return compounds.with_columns(pl.Series('prediction', predictions))


@pytest.mark.unit
class TestGreedyAcquisition:
    def test_greedy_selects_highest_prediction_scores(self, small_real_compounds):
        compounds = _pool_with_predictions(
            small_real_compounds.head(6).clone(),
            [0.2, 0.9, 0.5, 0.8, 0.1, 0.7],
        )

        selected = GreedyAcquisition().select(compounds, n_select=3)

        expected_ids = [
            compounds.get_column('ID')[1],
            compounds.get_column('ID')[3],
            compounds.get_column('ID')[5],
        ]

        assert selected.get_column('ID').to_list() == expected_ids
        assert selected.get_column('prediction').to_list() == [0.9, 0.8, 0.7]
        assert selected.get_column('acquisition_score').to_list() == [0.9, 0.8, 0.7]

    def test_greedy_selects_lowest_prediction_scores_when_score_direction_is_lower(self, small_real_compounds):
        compounds = _pool_with_predictions(
            small_real_compounds.head(6).clone(),
            [2.0, 0.9, 1.5, 0.3, 1.0, 0.2],
        )

        selected = GreedyAcquisition(score_direction='lower').select(compounds, n_select=3)

        assert selected.get_column('prediction').to_list() == [0.2, 0.3, 0.9]

    def test_greedy_returns_all_compounds_when_requested_batch_exceeds_pool(self, small_real_compounds):
        compounds = _pool_with_predictions(
            small_real_compounds.head(4).clone(),
            [0.4, 0.1, 0.7, 0.2],
        )

        selected = GreedyAcquisition().select(compounds, n_select=10)

        assert len(selected) == 4
        assert set(selected.get_column('ID').to_list()) == set(compounds.get_column('ID').to_list())

    def test_greedy_is_deterministic_for_identical_inputs(self, small_real_compounds):
        compounds = _pool_with_predictions(
            small_real_compounds.head(8).clone(),
            np.linspace(0.1, 0.8, 8),
        )

        acquisition = GreedyAcquisition()

        first = acquisition.select(compounds, n_select=5)
        second = acquisition.select(compounds, n_select=5)

        assert first.get_column('ID').to_list() == second.get_column('ID').to_list()
        assert first.get_column('acquisition_score').to_list() == second.get_column('acquisition_score').to_list()

    def test_greedy_preserves_requested_batch_size_when_scores_are_identical(self, small_real_compounds):
        compounds = _pool_with_predictions(
            small_real_compounds.head(7).clone(),
            np.ones(7),
        )

        selected = GreedyAcquisition().select(compounds, n_select=4)

        assert len(selected) == 4
        assert np.allclose(selected.get_column('acquisition_score').to_numpy(), 1.0)

    def test_greedy_rejects_empty_pool(self, empty_compounds):
        with pytest.raises(ValueError, match='compounds DataFrame is empty'):
            GreedyAcquisition().select(empty_compounds, n_select=1)

    def test_greedy_rejects_missing_prediction_column(self, small_real_compounds):
        with pytest.raises(ValueError, match='Missing required columns'):
            GreedyAcquisition().select(small_real_compounds.head(5).clone(), n_select=2)

    def test_greedy_rejects_non_positive_batch_size(self, small_real_compounds):
        compounds = _pool_with_predictions(small_real_compounds.head(5).clone(), np.linspace(0.1, 0.5, 5))

        with pytest.raises(ValueError, match='n_select must be positive'):
            GreedyAcquisition().select(compounds, n_select=0)

    def test_greedy_tiebreaking_at_boundary_is_index_stable(self, small_real_compounds):
        compounds = _pool_with_predictions(
            small_real_compounds.head(5).clone(),
            [0.5, 0.5, 0.5, 0.5, 0.5],
        )

        result1 = GreedyAcquisition().select(compounds, n_select=3)
        result2 = GreedyAcquisition().select(compounds, n_select=3)

        assert result1.get_column('ID').to_list() == result2.get_column('ID').to_list()
        assert len(result1) == 3

    def test_greedy_rejects_nan_predictions(self, small_real_compounds):
        compounds = _pool_with_predictions(
            small_real_compounds.head(5).clone(),
            [0.1, 0.2, np.nan, 0.4, 0.5],
        )

        with pytest.raises(ValueError, match='Predictions contain NaN values'):
            GreedyAcquisition().select(compounds, n_select=2)

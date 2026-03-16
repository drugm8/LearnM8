"""Tests for entropy acquisition."""

import numpy as np
import polars as pl
import pytest

from learnm8.acquisition import EntropyAcquisition


def _pool_with_predictions_and_uncertainty(
    compounds: pl.DataFrame,
    predictions: list[float] | np.ndarray,
    uncertainty: list[float] | np.ndarray,
) -> pl.DataFrame:
    return compounds.with_columns([
        pl.Series('prediction', predictions),
        pl.Series('uncertainty', uncertainty),
    ])


@pytest.mark.unit
class TestEntropyAcquisition:
    def test_entropy_selects_highest_uncertainty_when_entropy_type_is_uncertainty(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(5).clone(),
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.1, 0.6, 0.2, 0.5, 0.3],
        )

        selected = EntropyAcquisition(entropy_type='uncertainty').select(compounds, n_select=2)

        assert selected.get_column('acquisition_score').to_list() == [0.6, 0.5]

    def test_entropy_selects_highest_variance_when_entropy_type_is_variance(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(5).clone(),
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.1, 0.6, 0.2, 0.5, 0.3],
        )

        selected = EntropyAcquisition(entropy_type='variance').select(compounds, n_select=2)

        assert np.allclose(selected.get_column('acquisition_score').to_numpy(), [0.36, 0.25])

    def test_entropy_returns_all_compounds_when_requested_batch_exceeds_pool(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(4).clone(),
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4],
        )

        selected = EntropyAcquisition().select(compounds, n_select=10)

        assert len(selected) == 4

    def test_entropy_is_deterministic_for_identical_inputs(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(6).clone(),
            [0.1, 0.3, 0.4, 0.2, 0.8, 0.7],
            [0.2, 0.4, 0.1, 0.3, 0.2, 0.1],
        )

        acquisition = EntropyAcquisition()
        first = acquisition.select(compounds, n_select=3)
        second = acquisition.select(compounds, n_select=3)

        assert first.get_column('ID').to_list() == second.get_column('ID').to_list()

    def test_entropy_handles_identical_uncertainty_values(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(5).clone(),
            [0.1, 0.2, 0.3, 0.4, 0.5],
            np.full(5, 0.4),
        )

        selected = EntropyAcquisition().select(compounds, n_select=3)

        assert len(selected) == 3
        assert np.allclose(selected.get_column('acquisition_score').to_numpy(), 0.4)

    def test_entropy_rejects_invalid_entropy_type(self):
        with pytest.raises(ValueError, match='entropy_type must be'):
            EntropyAcquisition(entropy_type='invalid')

    def test_entropy_rejects_missing_uncertainty_column(self, small_real_compounds):
        compounds = small_real_compounds.head(5).clone().with_columns(pl.Series('prediction', np.linspace(0.1, 0.5, 5)))

        with pytest.raises(ValueError, match='requires an \'uncertainty\' column'):
            EntropyAcquisition().select(compounds, n_select=2)

    def test_entropy_rejects_non_positive_batch_size(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(5).clone(),
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
        )

        with pytest.raises(ValueError, match='n_select must be positive'):
            EntropyAcquisition().select(compounds, n_select=0)

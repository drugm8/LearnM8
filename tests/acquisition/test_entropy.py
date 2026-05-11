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
        # Feature 019: scores are now differential entropy in nats, not raw σ.
        # Selection rank still matches σ-descending baseline.
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(5).clone(),
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.1, 0.6, 0.2, 0.5, 0.3],
        )

        selected = EntropyAcquisition(entropy_type='uncertainty').select(compounds, n_select=2)

        scores = selected.get_column('acquisition_score').to_numpy()
        half_log_2pi_e = 0.5 * float(np.log(2.0 * np.pi * np.e))
        np.testing.assert_allclose(scores, [half_log_2pi_e + np.log(0.6), half_log_2pi_e + np.log(0.5)])

    def test_entropy_selects_highest_variance_when_entropy_type_is_variance(self, small_real_compounds):
        # Feature 019: 'variance' branch returns 0.5·log(2πe·σ⁴) = log(2πe) + 2·log(σ).
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(5).clone(),
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.1, 0.6, 0.2, 0.5, 0.3],
        )

        selected = EntropyAcquisition(entropy_type='variance').select(compounds, n_select=2)

        scores = selected.get_column('acquisition_score').to_numpy()
        half_log_2pi_e = 0.5 * float(np.log(2.0 * np.pi * np.e))
        np.testing.assert_allclose(scores, [half_log_2pi_e + 2.0 * np.log(0.6), half_log_2pi_e + 2.0 * np.log(0.5)])

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
        # Feature 019: scores are differential entropy (nats); identical σ → identical entropy.
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(5).clone(),
            [0.1, 0.2, 0.3, 0.4, 0.5],
            np.full(5, 0.4),
        )

        selected = EntropyAcquisition().select(compounds, n_select=3)

        assert len(selected) == 3
        half_log_2pi_e = 0.5 * float(np.log(2.0 * np.pi * np.e))
        expected = half_log_2pi_e + np.log(0.4)
        np.testing.assert_allclose(selected.get_column('acquisition_score').to_numpy(), expected)

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

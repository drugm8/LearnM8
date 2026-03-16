"""Tests for Thompson sampling acquisition."""

import numpy as np
import polars as pl
import pytest

from learnm8.acquisition import ThompsonSamplingAcquisition


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
class TestThompsonSamplingAcquisition:
    def test_thompson_sampling_returns_requested_number_of_compounds(self, compounds_with_uncertainty):
        selected = ThompsonSamplingAcquisition(random_state=42).select(compounds_with_uncertainty.head(10).clone(), n_select=5)

        assert len(selected) == 5
        assert 'acquisition_score' in selected.columns

    def test_thompson_sampling_is_reproducible_with_fixed_seed(self, compounds_with_uncertainty):
        compounds = compounds_with_uncertainty.head(10).clone()

        first = ThompsonSamplingAcquisition(random_state=42).select(compounds, n_select=6)
        second = ThompsonSamplingAcquisition(random_state=42).select(compounds, n_select=6)

        assert first.get_column('ID').to_list() == second.get_column('ID').to_list()
        assert np.allclose(first.get_column('acquisition_score').to_numpy(), second.get_column('acquisition_score').to_numpy())

    def test_thompson_sampling_differs_with_different_seeds(self, compounds_with_uncertainty):
        compounds = compounds_with_uncertainty.head(10).clone()

        first = ThompsonSamplingAcquisition(random_state=42).select(compounds, n_select=6)
        second = ThompsonSamplingAcquisition(random_state=123).select(compounds, n_select=6)

        assert first.get_column('ID').to_list() != second.get_column('ID').to_list()

    def test_thompson_sampling_returns_all_compounds_when_requested_batch_exceeds_pool(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(4).clone(),
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4],
        )

        selected = ThompsonSamplingAcquisition(random_state=42).select(compounds, n_select=10)

        assert len(selected) == 4

    def test_thompson_sampling_handles_identical_predictions_with_requested_batch_size(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(6).clone(),
            np.full(6, 1.0),
            np.full(6, 0.2),
        )

        selected = ThompsonSamplingAcquisition(random_state=42).select(compounds, n_select=3)

        assert len(selected) == 3

    def test_thompson_sampling_rejects_missing_uncertainty_column(self, small_real_compounds):
        compounds = small_real_compounds.head(5).clone().with_columns(pl.Series('prediction', np.linspace(0.1, 0.5, 5)))

        with pytest.raises(ValueError, match='requires an \'uncertainty\' column'):
            ThompsonSamplingAcquisition(random_state=42).select(compounds, n_select=2)

    def test_thompson_sampling_distribution_uses_uncertainty_as_standard_deviation(self, small_real_compounds):
        base = small_real_compounds.head(5).clone()
        compounds = _pool_with_predictions_and_uncertainty(
            base,
            [10.0, 12.0, 8.0, 11.0, 9.0],
            np.full(5, 3.0),
        )

        acquisition_scores = []
        for seed in range(200):
            result = ThompsonSamplingAcquisition(random_state=seed).select(compounds, n_select=5)
            acquisition_scores.append(
                result.filter(pl.col('prediction') == 10.0).get_column('acquisition_score')[0]
            )

        sample_mean = np.mean(acquisition_scores)
        sample_std = np.std(acquisition_scores)
        dist_to_correct = abs(sample_std - 3.0)
        dist_to_bug = abs(sample_std - np.sqrt(3.0))

        assert abs(sample_mean - 10.0) < 0.5
        assert dist_to_correct < dist_to_bug
        assert abs(sample_std - 3.0) < 0.8

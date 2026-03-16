"""Tests for expected improvement acquisition."""

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from learnm8.acquisition import ExpectedImprovementAcquisition


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
class TestExpectedImprovementAcquisition:
    def test_expected_improvement_returns_non_negative_scores(self, compounds_with_uncertainty):
        compounds = compounds_with_uncertainty.head(10).clone()
        current_best = compounds.get_column('prediction').max()

        selected = ExpectedImprovementAcquisition(current_best=current_best).select(compounds, n_select=5)

        assert len(selected) == 5
        assert np.all(selected.get_column('acquisition_score').to_numpy() >= 0)

    def test_expected_improvement_uses_score_direction_for_minimization(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(4).clone(),
            [1.0, 0.8, 0.4, 1.2],
            [0.1, 0.2, 0.4, 0.1],
        )

        selected = ExpectedImprovementAcquisition(
            current_best=0.9,
            score_direction='lower',
            xi=0.0,
        ).select(compounds, n_select=2)

        assert len(selected) == 2
        assert selected.get_column('acquisition_score')[0] >= selected.get_column('acquisition_score')[1]

    def test_expected_improvement_zero_uncertainty_uses_positive_improvement_rule(self, small_real_compounds):
        base = small_real_compounds.head(2).clone()
        compounds = _pool_with_predictions_and_uncertainty(base, [12.0, 8.0], [0.0, 0.0])

        result = ExpectedImprovementAcquisition(current_best=10.0, xi=0.0).select(compounds, n_select=2).sort('ID')

        assert result.get_column('acquisition_score').to_list() == [2.0, 0.0]

    def test_expected_improvement_rejects_missing_current_best(self, compounds_with_uncertainty):
        with pytest.raises(ValueError, match='requires \'current_best\' parameter'):
            ExpectedImprovementAcquisition().select(compounds_with_uncertainty.head(5).clone(), n_select=2)

    def test_expected_improvement_rejects_missing_uncertainty(self, small_real_compounds):
        compounds = small_real_compounds.head(5).clone().with_columns(pl.Series('prediction', np.linspace(0.1, 0.5, 5)))

        with pytest.raises(ValueError, match='requires an \'uncertainty\' column'):
            ExpectedImprovementAcquisition(current_best=0.5).select(compounds, n_select=2)

    def test_expected_improvement_rejects_negative_xi(self):
        with pytest.raises(ValueError, match='xi must be non-negative'):
            ExpectedImprovementAcquisition(xi=-0.1, current_best=0.5)

    def test_expected_improvement_matches_manual_formula(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(4).clone(),
            [10.0, 12.0, 8.0, 15.0],
            [1.0, 2.0, 0.5, 3.0],
        )
        current_best = 11.0
        xi = 0.01

        result = ExpectedImprovementAcquisition(
            current_best=current_best,
            xi=xi,
            score_direction='higher',
        ).select(compounds, n_select=4)

        mu = 12.0
        sigma = 2.0
        improvement = mu - current_best - xi
        z_score = improvement / sigma
        expected_score = improvement * norm.cdf(z_score) + sigma * norm.pdf(z_score)

        score = result.filter(pl.col('prediction') == 12.0).get_column('acquisition_score')[0]
        assert np.isclose(score, expected_score, rtol=1e-5)

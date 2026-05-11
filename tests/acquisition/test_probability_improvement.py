"""Tests for probability of improvement acquisition."""

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from learnm8.acquisition import ProbabilityImprovementAcquisition
from learnm8.exceptions import LearnerError


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
class TestProbabilityImprovementAcquisition:
    def test_probability_improvement_returns_scores_between_zero_and_one(self, compounds_with_uncertainty):
        compounds = compounds_with_uncertainty.head(10).clone()
        current_best = compounds.get_column('prediction').max()

        selected = ProbabilityImprovementAcquisition(current_best=current_best).select(compounds, n_select=5)

        scores = selected.get_column('acquisition_score').to_numpy()
        assert len(selected) == 5
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_probability_improvement_uses_score_direction_for_minimization(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(4).clone(),
            [1.0, 0.8, 0.4, 1.2],
            [0.1, 0.2, 0.4, 0.1],
        )

        selected = ProbabilityImprovementAcquisition(
            current_best=0.9,
            score_direction='lower',
            xi=0.0,
        ).select(compounds, n_select=2)

        assert len(selected) == 2
        assert selected.get_column('acquisition_score')[0] >= selected.get_column('acquisition_score')[1]

    def test_pi_returns_zero_when_sigma_is_zero(self, small_real_compounds):
        """Spec 022 FR-006: Botorch σ=0 → PI=0 convention.

        Replaces the prior σ=0 → indicator(improvement > 0) semantics.
        PI of a perfectly-deterministic candidate is 0 by the same Botorch
        convention as EI — when σ=0 there is no uncertainty information.
        """
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(2).clone(),
            [12.0, 8.0],
            [0.0, 0.0],
        )

        result = ProbabilityImprovementAcquisition(current_best=10.0, xi=0.0).select(compounds, n_select=2).sort('ID')

        assert result.get_column('acquisition_score').to_list() == [0.0, 0.0]

    def test_probability_improvement_rejects_missing_current_best(self, compounds_with_uncertainty):
        with pytest.raises(ValueError, match='requires \'current_best\' parameter'):
            ProbabilityImprovementAcquisition().select(compounds_with_uncertainty.head(5).clone(), n_select=2)

    def test_probability_improvement_rejects_missing_uncertainty(self, small_real_compounds):
        compounds = small_real_compounds.head(5).clone().with_columns(pl.Series('prediction', np.linspace(0.1, 0.5, 5)))

        with pytest.raises(ValueError, match='requires an \'uncertainty\' column'):
            ProbabilityImprovementAcquisition(current_best=0.5).select(compounds, n_select=2)

    def test_probability_improvement_rejects_negative_xi(self):
        with pytest.raises(ValueError, match='xi must be non-negative'):
            ProbabilityImprovementAcquisition(xi=-0.1, current_best=0.5)

    def test_probability_improvement_matches_manual_formula(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(4).clone(),
            [10.0, 12.0, 8.0, 15.0],
            [1.0, 2.0, 0.5, 3.0],
        )
        current_best = 11.0
        xi = 0.01

        result = ProbabilityImprovementAcquisition(
            current_best=current_best,
            xi=xi,
            score_direction='higher',
        ).select(compounds, n_select=4)

        improvement = 12.0 - current_best - xi
        expected_score = norm.cdf(improvement / 2.0)

        score = result.filter(pl.col('prediction') == 12.0).get_column('acquisition_score')[0]
        assert np.isclose(score, expected_score, rtol=1e-5)

    def test_inf_uncertainty_raises_learner_error(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(3).clone(),
            [1.0, 2.0, 3.0],
            [1.0, np.inf, 1.0],
        )
        with pytest.raises(LearnerError, match='Inf uncertainties'):
            ProbabilityImprovementAcquisition(current_best=0.5).select(compounds, n_select=2)

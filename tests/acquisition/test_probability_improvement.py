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
    return compounds.with_columns(
        [
            pl.Series('prediction', predictions),
            pl.Series('uncertainty', uncertainty),
        ]
    )


@pytest.mark.unit
class TestProbabilityImprovementAcquisition:
    def test_probability_improvement_returns_scores_between_zero_and_one(
        self, compounds_with_uncertainty
    ):
        compounds = compounds_with_uncertainty.head(10).clone()
        current_best = compounds.get_column('prediction').max()

        selected = ProbabilityImprovementAcquisition(current_best=current_best).select(
            compounds, n_select=5
        )

        scores = selected.get_column('acquisition_score').to_numpy()
        assert len(selected) == 5
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_probability_improvement_uses_score_direction_for_minimization(
        self, small_real_compounds
    ):
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
        assert (
            selected.get_column('acquisition_score')[0]
            >= selected.get_column('acquisition_score')[1]
        )

    def test_pi_at_zero_sigma_equals_improvement_indicator(self, small_real_compounds):
        """At sigma=0 PI is the deterministic improvement indicator.

        The sigma->0 limit of Probability of Improvement is ``1.0`` when the
        deterministic prediction improves over the current best and ``0.0``
        otherwise. A confident improving compound must score 1.0, not 0.
        """
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(2).clone(),
            [12.0, 8.0],
            [0.0, 0.0],
        )

        result = (
            ProbabilityImprovementAcquisition(current_best=10.0, xi=0.0)
            .select(compounds, n_select=2)
            .sort('ID')
        )

        assert result.get_column('acquisition_score').to_list() == [1.0, 0.0]

    def test_pi_deterministic_winner_outranks_uncertain_non_improver(
        self, small_real_compounds
    ):
        """A confident improving compound must outrank an uncertain non-improver.

        sigma=0 with mean above current_best yields PI=1.0; it must rank ahead
        of a sigma>0 compound whose mean is below current_best (PI<0.5).
        """
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(2).clone(),
            [20.0, 4.0],
            [0.0, 1.5],
        )

        result = ProbabilityImprovementAcquisition(current_best=10.0, xi=0.0).select(
            compounds, n_select=2
        )

        winner = result.filter(pl.col('prediction') == 20.0)
        loser = result.filter(pl.col('prediction') == 4.0)
        winner_score = winner.get_column('acquisition_score')[0]
        loser_score = loser.get_column('acquisition_score')[0]
        assert winner_score > 0
        assert winner_score > loser_score
        assert result.get_column('prediction')[0] == 20.0

    def test_probability_improvement_rejects_missing_current_best(
        self, compounds_with_uncertainty
    ):
        with pytest.raises(ValueError, match="requires 'current_best' parameter"):
            ProbabilityImprovementAcquisition().select(
                compounds_with_uncertainty.head(5).clone(), n_select=2
            )

    def test_probability_improvement_rejects_missing_uncertainty(
        self, small_real_compounds
    ):
        compounds = (
            small_real_compounds.head(5)
            .clone()
            .with_columns(pl.Series('prediction', np.linspace(0.1, 0.5, 5)))
        )

        with pytest.raises(ValueError, match="requires an 'uncertainty' column"):
            ProbabilityImprovementAcquisition(current_best=0.5).select(
                compounds, n_select=2
            )

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

        score = result.filter(pl.col('prediction') == 12.0).get_column(
            'acquisition_score'
        )[0]
        assert np.isclose(score, expected_score, rtol=1e-5)

    def test_inf_uncertainty_raises_learner_error(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(3).clone(),
            [1.0, 2.0, 3.0],
            [1.0, np.inf, 1.0],
        )
        with pytest.raises(LearnerError, match='Inf uncertainties'):
            ProbabilityImprovementAcquisition(current_best=0.5).select(
                compounds, n_select=2
            )

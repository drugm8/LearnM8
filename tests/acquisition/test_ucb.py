"""Tests for upper confidence bound acquisition."""

import numpy as np
import polars as pl
import pytest

from learnm8.acquisition import UCBAcquisition


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
class TestUCBAcquisition:
    def test_ucb_selects_highest_upper_confidence_bound_scores(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(5).clone(),
            [0.1, 0.6, 0.4, 0.2, 0.5],
            [0.0, 0.1, 0.5, 0.2, 0.1],
        )

        selected = UCBAcquisition(beta=1.0).select(compounds, n_select=2)

        assert selected.get_column('acquisition_score').to_list() == [0.9, 0.7]

    def test_ucb_selects_lowest_lower_confidence_bound_scores_for_minimization(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(5).clone(),
            [2.0, 1.0, 3.0, 1.5, 2.5],
            [0.1, 0.3, 0.2, 0.4, 0.1],
        )

        selected = UCBAcquisition(beta=1.0, score_direction='lower').select(compounds, n_select=2)

        assert selected.get_column('acquisition_score').to_list() == [0.7, 1.1]

    def test_ucb_beta_parameter_changes_score_scale(self, compounds_with_uncertainty):
        compounds = compounds_with_uncertainty.head(10).clone()

        conservative = UCBAcquisition(beta=0.1).select(compounds, n_select=5)
        aggressive = UCBAcquisition(beta=2.0).select(compounds, n_select=5)

        assert len(conservative) == 5
        assert len(aggressive) == 5
        assert aggressive.get_column('acquisition_score').mean() != conservative.get_column('acquisition_score').mean()

    def test_ucb_returns_all_compounds_when_requested_batch_exceeds_pool(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(4).clone(),
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4],
        )

        selected = UCBAcquisition(beta=1.0).select(compounds, n_select=10)

        assert len(selected) == 4

    def test_ucb_is_deterministic_for_identical_inputs(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(6).clone(),
            [0.1, 0.3, 0.4, 0.2, 0.8, 0.7],
            [0.2, 0.4, 0.1, 0.3, 0.2, 0.1],
        )

        acquisition = UCBAcquisition(beta=1.5)
        first = acquisition.select(compounds, n_select=3)
        second = acquisition.select(compounds, n_select=3)

        assert first.get_column('ID').to_list() == second.get_column('ID').to_list()

    def test_ucb_rejects_missing_uncertainty_column(self, small_real_compounds):
        compounds = small_real_compounds.head(5).clone().with_columns(pl.Series('prediction', np.linspace(0.1, 0.5, 5)))

        with pytest.raises(ValueError, match='requires an \'uncertainty\' column'):
            UCBAcquisition().select(compounds, n_select=2)

    def test_ucb_rejects_negative_uncertainty_values(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(5).clone(),
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.1, -0.2, 0.3, 0.4, 0.5],
        )

        with pytest.raises(ValueError, match='negative values'):
            UCBAcquisition().select(compounds, n_select=2)

    def test_ucb_rejects_nan_uncertainty_values(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(5).clone(),
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.1, np.nan, 0.3, 0.4, 0.5],
        )

        # Feature 019: NaN at acquisition layer hits the defence-in-depth
        # secondary guard (LearnerError); cycle.py is the canonical fail-fast.
        from learnm8.exceptions import LearnerError
        with pytest.raises(LearnerError, match=r"\[secondary-guard\] Uncertainties contain"):
            UCBAcquisition().select(compounds, n_select=2)

    def test_ucb_rejects_inf_uncertainty_values(self, small_real_compounds):
        # Item 9: UCB must reject ±inf σ like EI/PI/Entropy, not propagate ±inf.
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(5).clone(),
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.1, np.inf, 0.3, 0.4, 0.5],
        )

        from learnm8.exceptions import LearnerError
        with pytest.raises(LearnerError, match='Inf uncertainties not supported'):
            UCBAcquisition().select(compounds, n_select=2)

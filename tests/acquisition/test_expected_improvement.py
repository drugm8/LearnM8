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

    def test_ei_returns_zero_when_sigma_is_zero(self, small_real_compounds):
        """Spec 022 FR-006: Botorch σ=0 → EI=0 convention.

        Replaces the prior Močkus ``max(improvement, 0)`` semantics.
        When σ=0 the model is fully certain — no improvement opportunity
        beyond the deterministic prediction. This convention is locked in
        per ``contracts/inline-norm.md``.
        """
        base = small_real_compounds.head(2).clone()
        compounds = _pool_with_predictions_and_uncertainty(base, [12.0, 8.0], [0.0, 0.0])

        result = ExpectedImprovementAcquisition(current_best=10.0, xi=0.0).select(compounds, n_select=2).sort('ID')

        assert result.get_column('acquisition_score').to_list() == [0.0, 0.0]

    def test_ei_no_nan_intermediates_when_sigma_zero(self, small_real_compounds):
        """Spec 022 FR-006: σ=0 must NOT produce NaN intermediates.

        The σ-guard via ``np.where(sigma > 0, ...)`` ensures division by zero
        never propagates into the EI output.
        """
        base = small_real_compounds.head(3).clone()
        compounds = _pool_with_predictions_and_uncertainty(
            base, [12.0, 5.0, 11.0], [0.0, 0.0, 0.5]
        )
        result = ExpectedImprovementAcquisition(current_best=10.0, xi=0.0).select(compounds, n_select=3)
        scores = result.get_column('acquisition_score').to_numpy()
        assert not np.isnan(scores).any(), f"NaN in EI scores: {scores}"

    def test_ei_symmetric_clipping_at_extreme_z(self, small_real_compounds):
        """Spec 022 FR-007: z clipped to ±37 must not produce NaN or inf.

        Extreme μ-f_best ratios where z would naturally exceed ±100 still
        produce finite EI under symmetric clipping.
        """
        base = small_real_compounds.head(2).clone()
        # σ=0.001 makes z = (μ - f_best - xi) / σ ≈ ±10000 — would underflow
        # exp(-z²/2) without clipping.
        compounds = _pool_with_predictions_and_uncertainty(base, [50.0, -50.0], [0.001, 0.001])
        result = ExpectedImprovementAcquisition(current_best=0.0, xi=0.0).select(compounds, n_select=2)
        scores = result.get_column('acquisition_score').to_numpy()
        assert np.all(np.isfinite(scores)), f"Non-finite EI under extreme z: {scores}"

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

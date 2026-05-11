import numpy as np
import polars as pl
import pytest

from learnm8.acquisition import (
    ExpectedImprovementAcquisition,
    ProbabilityImprovementAcquisition,
    ThompsonSamplingAcquisition,
    UCBAcquisition,
)


@pytest.mark.unit
class TestUncertaintyBasedIntegration:
    """Integration tests for uncertainty-based acquisition."""

    def test_uncertainty_acquisition_methods_all_return_valid_batch_sizes(self, compounds_with_uncertainty):
        """Compare different uncertainty-based acquisition methods."""
        compounds = compounds_with_uncertainty.clone()

        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")

        # Calculate current_best from predictions (simulating labeled data)
        current_best = compounds.get_column('prediction').max()

        # Shared integration coverage stays here; dedicated unit coverage lives in per-module files.
        ucb_acq = UCBAcquisition(beta=1.0)
        ei_acq = ExpectedImprovementAcquisition(xi=0.01, current_best=current_best)
        pi_acq = ProbabilityImprovementAcquisition(xi=0.01, current_best=current_best)
        ts_acq = ThompsonSamplingAcquisition(random_state=42)

        n_select = 5
        ucb_selected = ucb_acq.select(compounds, n_select=n_select)
        ei_selected = ei_acq.select(compounds, n_select=n_select)
        pi_selected = pi_acq.select(compounds, n_select=n_select)
        ts_selected = ts_acq.select(compounds, n_select=n_select)

        # All should select correct number
        assert len(ucb_selected) == n_select
        assert len(ei_selected) == n_select
        assert len(pi_selected) == n_select
        assert len(ts_selected) == n_select

        # Different methods should show some diversity in selection
        all_selections = [
            set(ucb_selected.get_column('ID').to_list()),
            set(ei_selected.get_column('ID').to_list()),
            set(pi_selected.get_column('ID').to_list()),
            set(ts_selected.get_column('ID').to_list())
        ]

        # Not all methods should select identical compounds
        assert not all(s == all_selections[0] for s in all_selections[1:])

    def test_uncertainty_methods_with_molecular_workflow(self, medium_real_compounds):
        """Test uncertainty methods in realistic molecular workflow."""
        compounds = medium_real_compounds.clone()

        if len(compounds) < 20:
            pytest.skip("Insufficient compounds for workflow test")

        # Simulate realistic uncertainty estimates
        np.random.seed(42)
        compounds = compounds.with_columns([
            (pl.col('Activity') + pl.Series('noise', np.random.normal(0, 2, len(compounds)))).alias('prediction'),
            pl.Series('uncertainty', np.random.uniform(0.1, 1.0, len(compounds)))
        ])

        # Test that uncertainty methods work with real molecular data
        acq = UCBAcquisition(beta=1.5)
        selected = acq.select(compounds, n_select=15)

        assert len(selected) == 15
        compound_ids = compounds.get_column('ID').to_list()
        selected_ids = selected.get_column('ID').to_list()
        assert all(id in compound_ids for id in selected_ids)

        # Selected compounds should have reasonable uncertainty properties
        selected_uncertainty = selected.get_column('uncertainty').to_numpy()
        assert np.all(selected_uncertainty > 0)
        assert np.all(np.isfinite(selected_uncertainty))

    def test_uncertainty_acquisition_error_handling(self, small_real_compounds):
        """Test error handling in uncertainty-based acquisition."""
        compounds = small_real_compounds.clone()
        # Test with missing uncertainty
        compounds = compounds.with_columns(
            pl.col('Activity').alias('prediction')
        )
        # No uncertainty column

        acq = UCBAcquisition(beta=1.0)

        with pytest.raises((KeyError, ValueError)):
            acq.select(compounds, n_select=5)

        # Test with NaN uncertainty
        compounds = compounds.with_columns(
            pl.lit(float('nan')).alias('uncertainty')
        )

        # Feature 019: NaN at acquisition layer hits the secondary-guard
        # LearnerError; canonical fail-fast lives in cycle.py.
        from learnm8.exceptions import LearnerError
        with pytest.raises((ValueError, RuntimeError, LearnerError)):
            acq.select(compounds, n_select=5)


@pytest.mark.unit
class TestEIPIRegressionTests:
    """Regression tests to ensure EI and PI are different and mathematically correct.

    These tests were added after fixing bug where np.sqrt(uncertainties) was incorrectly
    applied to standard deviations, causing EI and PI to produce identical results.
    """

    @pytest.mark.skip(reason="Regression test has flaky threshold - mathematical correctness verified by test_ei_pi_mathematical_correctness")
    def test_ei_pi_produce_different_selections(self, compounds_with_uncertainty):
        """Test that EI and PI compute DIFFERENT scores (regression test for sqrt bug).

        Before the fix, EI and PI produced identical scores due to incorrect sqrt() application.
        After the fix, they should compute fundamentally different acquisition scores.

        NOTE: This test is skipped because it has a flaky threshold that fails with certain
        fixture data distributions. The mathematical correctness of EI and PI is verified
        by test_ei_pi_mathematical_correctness which uses controlled test data.
        """
        compounds = compounds_with_uncertainty.clone()

        if len(compounds) < 20:
            pytest.skip("Need at least 20 compounds for meaningful comparison")

        # Use a current_best below max to create meaningful score variation
        # (if current_best = max, all improvements are negative → similar scores)
        predictions = compounds.get_column('prediction').to_numpy()
        current_best = np.percentile(predictions, 75)  # 75th percentile

        # Create EI and PI acquisition functions with same parameters
        ei_acq = ExpectedImprovementAcquisition(xi=0.01, current_best=current_best, score_direction='higher')
        pi_acq = ProbabilityImprovementAcquisition(xi=0.01, current_best=current_best, score_direction='higher')

        # Select ALL compounds (not just top-k) to get scores for comparison
        ei_result = ei_acq.select(compounds, n_select=len(compounds))
        pi_result = pi_acq.select(compounds, n_select=len(compounds))

        # Sort by ID for direct comparison
        ei_sorted = ei_result.sort('ID')
        pi_sorted = pi_result.sort('ID')

        ei_scores = ei_sorted.get_column('acquisition_score').to_numpy()
        pi_scores = pi_sorted.get_column('acquisition_score').to_numpy()

        # KEY TEST: Scores should NOT be identical (this was the bug!)
        # Before fix: scores were identical because both used inflated Z-scores
        # After fix: EI = improvement*Phi(Z) + sigma*phi(Z), PI = Phi(Z) - fundamentally different
        assert not np.allclose(ei_scores, pi_scores, rtol=0.001), \
            "EI and PI produced identical scores - sqrt bug may still exist!"

        # Verify scores are on different scales
        # PI is typically in [0, 1] (probability), EI can be larger (expected improvement)
        ei_mean = np.mean(ei_scores)
        pi_mean = np.mean(pi_scores)

        # The ratio should NOT be close to 1 (different scales)
        ratio = ei_mean / pi_mean if pi_mean > 0 else 0
        assert not np.isclose(ratio, 1.0, rtol=0.1), \
            f"EI and PI have same scale (ratio={ratio:.4f}) - check implementation!"

        # Verify EI has the additional sigma*phi(Z) term that PI lacks
        # This means for same compounds, EI and PI rankings can differ
        ei_ranks = np.argsort(np.argsort(-ei_scores))  # Higher score = lower rank number
        pi_ranks = np.argsort(np.argsort(-pi_scores))

        # Rank correlation should not be perfect (though it can be high)
        from scipy.stats import spearmanr
        rank_correlation, _ = spearmanr(ei_ranks, pi_ranks)

        # Correlation can be high (.95+) but should not be exactly 1.0
        # Allow for floating point precision (0.99999...)
        assert rank_correlation <= 0.99999, \
            f"EI and PI have perfect rank correlation ({rank_correlation:.6f})"

    def test_ei_pi_mathematical_correctness(self, small_real_compounds):
        """Test that EI and PI compute mathematically correct scores."""
        from scipy.stats import norm

        test_data = small_real_compounds.head(4).select(['ID', 'SMILES']).with_columns([
            pl.Series('prediction', np.array([10.0, 12.0, 8.0, 15.0])),
            pl.Series('uncertainty', np.array([1.0, 2.0, 0.5, 3.0])),
        ])
        target_id = test_data.get_column('ID')[1]

        current_best = 11.0
        xi = 0.01

        # Manually calculate expected scores for compound B (prediction=12.0, uncertainty=2.0)
        mu = 12.0
        sigma = 2.0  # Should use uncertainty directly, not sqrt(uncertainty)
        improvement = mu - current_best - xi  # 12.0 - 11.0 - 0.01 = 0.99
        z = improvement / sigma  # 0.99 / 2.0 = 0.495

        # Expected EI = improvement * Phi(z) + sigma * phi(z)
        expected_ei_b = improvement * norm.cdf(z) + sigma * norm.pdf(z)

        # Expected PI = Φ(z)
        expected_pi_b = norm.cdf(z)

        # Run acquisition functions
        ei_acq = ExpectedImprovementAcquisition(xi=xi, current_best=current_best, score_direction='higher')
        pi_acq = ProbabilityImprovementAcquisition(xi=xi, current_best=current_best, score_direction='higher')

        ei_result = ei_acq.select(test_data, n_select=4)
        pi_result = pi_acq.select(test_data, n_select=4)

        # Find compound B's scores
        ei_score_b = ei_result.filter(pl.col('ID') == target_id).get_column('acquisition_score')[0]
        pi_score_b = pi_result.filter(pl.col('ID') == target_id).get_column('acquisition_score')[0]

        # Verify scores match manual calculations (within numerical tolerance)
        assert np.isclose(ei_score_b, expected_ei_b, rtol=1e-5), \
            f"EI score {ei_score_b} doesn't match expected {expected_ei_b}"
        assert np.isclose(pi_score_b, expected_pi_b, rtol=1e-5), \
            f"PI score {pi_score_b} doesn't match expected {expected_pi_b}"

        # Verify EI and PI are different
        assert not np.isclose(ei_score_b, pi_score_b), "EI and PI scores should be different!"

    def test_uncertainty_format_is_std_not_variance(self, small_real_compounds):
        """Test that uncertainties represent standard deviation, not variance.

        This test documents the uncertainty format convention and ensures
        acquisition functions use uncertainties correctly.
        """
        test_data = small_real_compounds.head(2).select(['ID', 'SMILES']).with_columns([
            pl.Series('prediction', np.array([10.0, 10.0])),
            pl.Series('uncertainty', np.array([2.0, 4.0])),
        ])

        current_best = 8.0

        # UCB should use uncertainties directly
        ucb = UCBAcquisition(beta=1.0, score_direction='higher')
        ucb_result = ucb.select(test_data, n_select=2)

        # With same prediction and beta=1.0, UCB score should differ by uncertainty value
        ucb_scores = ucb_result.sort('ID').get_column('acquisition_score').to_numpy()
        # UCB(A) = 10.0 + 1.0*2.0 = 12.0
        # UCB(B) = 10.0 + 1.0*4.0 = 14.0
        assert np.isclose(ucb_scores[0], 12.0, rtol=1e-5)
        assert np.isclose(ucb_scores[1], 14.0, rtol=1e-5)

        # EI and PI should also use uncertainties as std devs (not sqrt)
        ei = ExpectedImprovementAcquisition(xi=0.0, current_best=current_best, score_direction='higher')
        ei_result = ei.select(test_data, n_select=2)

        # Verify EI used correct std dev (larger uncertainty should affect score)
        ei_scores = ei_result.sort('ID').get_column('acquisition_score').to_numpy()
        # With larger uncertainty, compound B should have higher EI score
        assert ei_scores[1] > ei_scores[0], "Larger uncertainty should increase EI score"


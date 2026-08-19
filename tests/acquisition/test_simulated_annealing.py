"""Tests for simulated annealing acquisition function.

This module tests the SimulatedAnnealingAcquisition class following LearnM8's
testing philosophy of using real molecular data and focusing on functionality,
integration, error handling, and data validation.
"""

import logging

import numpy as np
import polars as pl
import pytest

from learnm8.acquisition.simulated_annealing import (
    SA_BACKFILL_WARN,
    SA_CALIBRATION_SAMPLES,
    SA_CALIBRATION_TOL,
    SA_MAX_TOTAL_STEPS,
    SA_TARGET_UPHILL_ACCEPTANCE,
    SimulatedAnnealingAcquisition,
    _calibrate_temperature,
    _derive_step_budget,
    _pad_neighbor_map,
)


@pytest.mark.unit
class TestSimulatedAnnealingAcquisition:
    """Test SimulatedAnnealingAcquisition with real molecular data."""

    def test_simulated_annealing_selects_unique_compounds_with_finite_scores(self, small_real_compounds):
        """Test basic simulated annealing functionality with real pharmaceutical compounds."""
        compounds = small_real_compounds.clone()
        # Add realistic predictions based on docking scores
        np.random.seed(42)
        predictions = compounds.get_column('Activity').to_numpy() + np.random.normal(0, 0.5, len(compounds))
        compounds = compounds.with_columns(pl.Series('prediction', predictions))

        # Test basic selection
        acq = SimulatedAnnealingAcquisition(random_state=42, max_iterations=100)
        selected = acq.select(compounds, n_select=8)

        # Verify output structure
        assert len(selected) == 8
        assert 'acquisition_score' in selected.columns
        assert all(id in compounds.get_column('ID').to_numpy() for id in selected.get_column('ID').to_list())
        assert len(selected.get_column('ID').unique()) == len(selected)  # No duplicates

        # Verify acquisition scores are present and finite
        assert np.all(np.isfinite(selected.get_column('acquisition_score').to_numpy()))

    def test_parameter_validation(self):
        """Test parameter validation in constructor."""
        # Test invalid temperatures
        with pytest.raises(ValueError, match="initial_temp must be positive"):
            SimulatedAnnealingAcquisition(initial_temp=0)

        with pytest.raises(ValueError, match="final_temp must be positive"):
            SimulatedAnnealingAcquisition(final_temp=0)

        with pytest.raises(ValueError, match="final_temp must be less than initial_temp"):
            SimulatedAnnealingAcquisition(initial_temp=0.5, final_temp=1.0)

        # Test invalid iterations
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            SimulatedAnnealingAcquisition(max_iterations=0)

        # Test invalid cooling schedule
        with pytest.raises(ValueError, match="cooling_schedule must be"):
            SimulatedAnnealingAcquisition(cooling_schedule='invalid')

        # Test invalid score direction
        with pytest.raises(ValueError, match="score_direction must be"):
            SimulatedAnnealingAcquisition(score_direction='invalid')

        # Test invalid neighbor strategy
        with pytest.raises(ValueError, match="neighbor_strategy must be"):
            SimulatedAnnealingAcquisition(neighbor_strategy='invalid')

        # Test invalid n_neighbors
        with pytest.raises(ValueError, match="n_neighbors must be >= 1"):
            SimulatedAnnealingAcquisition(n_neighbors=0)

    def test_cooling_schedules(self, small_real_compounds):
        """Test different cooling schedules with real molecular data."""
        compounds = small_real_compounds.clone()
        np.random.seed(42)
        predictions = compounds.get_column('Activity').to_numpy() + np.random.normal(0, 0.5, len(compounds))
        compounds = compounds.with_columns(pl.Series('prediction', predictions))

        # Test exponential cooling
        acq_exp = SimulatedAnnealingAcquisition(
            cooling_schedule='exponential',
            random_state=42,
            max_iterations=50
        )
        selected_exp = acq_exp.select(compounds, n_select=5)
        assert len(selected_exp) == 5
        assert acq_exp.get_name() == "SimulatedAnnealing(exponential_higher)"

        # Test linear cooling
        acq_lin = SimulatedAnnealingAcquisition(
            cooling_schedule='linear',
            random_state=42,
            max_iterations=50
        )
        selected_lin = acq_lin.select(compounds, n_select=5)
        assert len(selected_lin) == 5
        assert acq_lin.get_name() == "SimulatedAnnealing(linear_higher)"

        # Results may differ due to different cooling schedules
        # but both should return valid selections
        assert np.all(np.isfinite(selected_exp.get_column('acquisition_score').to_numpy()))
        assert np.all(np.isfinite(selected_lin.get_column('acquisition_score').to_numpy()))

    def test_score_directions(self, small_real_compounds):
        """Test both score directions (higher and lower optimization)."""
        compounds = small_real_compounds.clone()
        np.random.seed(42)
        predictions = compounds.get_column('Activity').to_numpy() + np.random.normal(0, 0.5, len(compounds))
        compounds = compounds.with_columns(pl.Series('prediction', predictions))

        # Test higher is better (maximization)
        acq_higher = SimulatedAnnealingAcquisition(
            score_direction='higher',
            random_state=42,
            max_iterations=50
        )
        selected_higher = acq_higher.select(compounds, n_select=5)

        # Test lower is better (minimization)
        acq_lower = SimulatedAnnealingAcquisition(
            score_direction='lower',
            random_state=42,
            max_iterations=50
        )
        selected_lower = acq_lower.select(compounds, n_select=5)

        # Both should return valid selections
        assert len(selected_higher) == 5
        assert len(selected_lower) == 5
        assert acq_higher.get_name() == "SimulatedAnnealing(exponential_higher)"
        assert acq_lower.get_name() == "SimulatedAnnealing(exponential_lower)"

        # For the same predictions, different score directions may select different compounds
        # but all selections should be valid
        assert np.all(np.isfinite(selected_higher.get_column('acquisition_score').to_numpy()))
        assert np.all(np.isfinite(selected_lower.get_column('acquisition_score').to_numpy()))

    def test_reproducibility(self, small_real_compounds):
        """Test reproducible behavior with fixed random seed."""
        compounds = small_real_compounds.clone()
        np.random.seed(42)
        predictions = compounds.get_column('Activity').to_numpy() + np.random.normal(0, 0.5, len(compounds))
        compounds = compounds.with_columns(pl.Series('prediction', predictions))

        # Test same seed produces identical results
        acq1 = SimulatedAnnealingAcquisition(random_state=42, max_iterations=50)
        acq2 = SimulatedAnnealingAcquisition(random_state=42, max_iterations=50)

        selected1 = acq1.select(compounds, n_select=5)
        selected2 = acq2.select(compounds, n_select=5)

        # Should select identical compounds with same seed
        assert selected1.get_column('ID').to_list() == selected2.get_column('ID').to_list()
        assert np.allclose(selected1.get_column('acquisition_score').to_numpy(), selected2.get_column('acquisition_score').to_numpy())

        # Test different seed produces different results (with high probability)
        acq3 = SimulatedAnnealingAcquisition(random_state=123, max_iterations=50)
        selected3 = acq3.select(compounds, n_select=5)

        # Different seeds should likely produce different selections
        # Note: There's a small chance they could be the same, but very unlikely
        assert len(selected3) == 5

    def test_edge_cases(self, small_real_compounds):
        """Test edge cases and boundary conditions."""
        compounds = small_real_compounds.clone()
        np.random.seed(42)
        predictions = compounds.get_column('Activity').to_numpy() + np.random.normal(0, 0.5, len(compounds))
        compounds = compounds.with_columns(pl.Series('prediction', predictions))

        # Test selecting all compounds
        acq = SimulatedAnnealingAcquisition(random_state=42)
        selected_all = acq.select(compounds, n_select=len(compounds))
        assert len(selected_all) == len(compounds)
        assert set(selected_all.get_column('ID').to_list()) == set(compounds.get_column('ID').to_list())

        # Test selecting more than available (should return all)
        selected_more = acq.select(compounds, n_select=len(compounds) + 10)
        assert len(selected_more) == len(compounds)

        # Test selecting single compound
        selected_one = acq.select(compounds, n_select=1)
        assert len(selected_one) == 1
        assert selected_one.row(0, named=True)['ID'] in compounds.get_column('ID').to_numpy()

    def test_identical_prediction_scores_still_return_requested_unique_batch(self, small_real_compounds):
        """Test simulated annealing handles identical scores without duplicating compounds."""
        compounds = small_real_compounds.head(8).clone().with_columns(
            pl.Series('prediction', np.ones(8))
        )

        selected = SimulatedAnnealingAcquisition(random_state=42, max_iterations=50).select(
            compounds,
            n_select=4,
        )

        assert len(selected) == 4
        assert selected.get_column('ID').n_unique() == 4
        assert np.allclose(selected.get_column('acquisition_score').to_numpy(), 1.0)

    def test_with_uncertainty_estimates(self, compounds_with_uncertainty):
        """Test integration with uncertainty estimates from real molecular predictions."""
        compounds = compounds_with_uncertainty.clone()

        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")

        # Basic version doesn't use uncertainty, but should work with it present
        acq = SimulatedAnnealingAcquisition(random_state=42, max_iterations=50)
        selected = acq.select(compounds, n_select=5)

        assert len(selected) == 5
        assert 'uncertainty' in compounds.columns  # Verify uncertainty was present
        assert np.all(selected.get_column('uncertainty').to_numpy() > 0)  # Verify uncertainty values are valid

        # Should not require uncertainty
        assert not acq.requires_uncertainty()

    def test_error_handling(self, small_real_compounds):
        """Test error handling with real molecular data."""
        compounds = small_real_compounds.clone()
        # Add predictions for valid test
        np.random.seed(42)
        predictions = compounds.get_column('Activity').to_numpy() + np.random.normal(0, 0.5, len(compounds))
        compounds = compounds.with_columns(pl.Series('prediction', predictions))

        acq = SimulatedAnnealingAcquisition(random_state=42)

        # Test missing required columns
        compounds_no_smiles = compounds.drop('SMILES')
        with pytest.raises(ValueError, match="Missing required columns"):
            acq.select(compounds_no_smiles, n_select=5)

        # Test empty DataFrame
        empty_compounds = compounds.head(0)
        with pytest.raises(ValueError, match="compounds DataFrame is empty"):
            acq.select(empty_compounds, n_select=5)

        # Test invalid n_select
        with pytest.raises(ValueError, match="n_select must be positive"):
            acq.select(compounds, n_select=0)

        # Test NaN predictions
        compounds_nan = compounds.clone()
        pred_with_nan = compounds_nan.get_column('prediction').to_numpy().copy()
        pred_with_nan[0] = np.nan
        compounds_nan = compounds_nan.with_columns(pl.Series('prediction', pred_with_nan))
        # Feature 019: secondary-guard LearnerError at acquisition layer.
        from learnm8.exceptions import LearnerError
        with pytest.raises(LearnerError, match=r"\[secondary-guard\] Predictions contain NaN"):
            acq.select(compounds_nan, n_select=5)

    def test_temperature_calculation(self):
        """Test temperature calculation for different cooling schedules."""
        acq_exp = SimulatedAnnealingAcquisition(
            initial_temp=1.0,
            final_temp=0.1,
            max_iterations=100,
            cooling_schedule='exponential'
        )

        acq_lin = SimulatedAnnealingAcquisition(
            initial_temp=1.0,
            final_temp=0.1,
            max_iterations=100,
            cooling_schedule='linear'
        )

        # Test temperature at start
        assert acq_exp._get_temperature(0) == 1.0
        assert acq_lin._get_temperature(0) == 1.0

        # Test temperature at end
        assert abs(acq_exp._get_temperature(100) - 0.1) < 1e-10
        assert abs(acq_lin._get_temperature(100) - 0.1) < 1e-10

        # Test temperature decreases monotonically
        temp_exp_mid = acq_exp._get_temperature(50)
        temp_lin_mid = acq_lin._get_temperature(50)

        assert 0.1 < temp_exp_mid < 1.0
        assert 0.1 < temp_lin_mid < 1.0

        # Linear cooling should be exactly halfway at 50% progress
        assert abs(temp_lin_mid - 0.55) < 1e-10  # (1.0 + 0.1) / 2 = 0.55

    def test_energy_calculation(self):
        """Test energy calculation for different score directions."""
        acq_higher = SimulatedAnnealingAcquisition(score_direction='higher')
        acq_lower = SimulatedAnnealingAcquisition(score_direction='lower')

        prediction = 5.0

        # For maximization: higher predictions should have lower energy
        energy_higher = acq_higher._calculate_energy(prediction)
        assert energy_higher == -5.0

        # For minimization: prediction value is the energy
        energy_lower = acq_lower._calculate_energy(prediction)
        assert energy_lower == 5.0

    def test_metropolis_acceptance(self):
        """Test Metropolis acceptance criterion."""
        acq = SimulatedAnnealingAcquisition(random_state=42)

        # Always accept better (lower energy) candidates
        assert acq._metropolis_accept(5.0, 3.0, 1.0)
        assert acq._metropolis_accept(3.0, 3.0, 1.0)  # Equal energy

        # Test temperature effect on worse candidates
        # At high temperature, should have higher probability of accepting worse candidates
        # At low temperature, should have lower probability

        # We can't test exact probabilities due to randomness, but we can test logic
        current_energy = 3.0
        worse_energy = 5.0

        # At zero temperature, never accept worse candidates
        assert not acq._metropolis_accept(current_energy, worse_energy, 0.0)

        # At very high temperature, acceptance probability approaches 1
        # exp(-2/1000) ≈ 0.998, so very likely to accept
        np.random.seed(42)
        high_temp_accept = acq._metropolis_accept(current_energy, worse_energy, 1000.0)
        # Don't assert the exact result due to randomness, just ensure no errors
        assert isinstance(high_temp_accept, bool)

    def test_integration_with_diverse_molecular_data(self, diverse_real_compounds):
        """Test with structurally diverse compounds across multiple targets."""
        compounds = diverse_real_compounds.clone()

        if len(compounds) == 0:
            pytest.skip("No diverse molecular data available")

        # Add realistic predictions
        np.random.seed(42)
        predictions = compounds.get_column('Activity').to_numpy() + np.random.normal(0, 1.0, len(compounds))
        compounds = compounds.with_columns(pl.Series('prediction', predictions))

        # Feature 027: the schedule is derived from the predictions. The
        # hand-set initial_temp=2.0/max_iterations=200 this test used to pass
        # bear no relation to this pool's energy scale and leave 60% of the
        # batch to greedy backfill, which now raises.
        acq = SimulatedAnnealingAcquisition(
            cooling_schedule='exponential',
            random_state=42
        )

        selected = acq.select(compounds, n_select=15)

        assert len(selected) == 15
        assert len(selected.get_column('ID').unique()) == 15  # No duplicates

        # Check that we get diversity across targets if Target column exists
        if 'Target' in compounds.columns:
            # Should ideally get compounds from multiple targets
            unique_targets_in_selection = selected.get_column('Target').n_unique()
            unique_targets_total = compounds.get_column('Target').n_unique()

            # At minimum, should have valid target assignments
            assert unique_targets_in_selection >= 1
            assert unique_targets_in_selection <= unique_targets_total

    def test_knn_neighbor_strategy_selects_compounds(self, small_real_compounds, small_real_morgan_features):
        """Test kNN-based neighbor generation produces valid selections."""
        compounds = small_real_compounds.clone()
        np.random.seed(42)
        predictions = compounds.get_column('Activity').to_numpy() + np.random.normal(0, 0.5, len(compounds))
        compounds = compounds.with_columns(pl.Series('prediction', predictions))

        from learnm8.features import create_featurizer
        featurizer_obj = create_featurizer('morgan')

        acq = SimulatedAnnealingAcquisition(
            neighbor_strategy='knn_features',
            n_neighbors=5,
            max_iterations=100,
            random_state=42,
            featurizer_obj=featurizer_obj,
        )
        selected = acq.select(compounds, n_select=8)

        assert len(selected) == 8
        assert len(selected.get_column('ID').unique()) == 8
        assert 'acquisition_score' in selected.columns
        assert np.all(np.isfinite(selected.get_column('acquisition_score').to_numpy()))

    def test_knn_falls_back_without_featurizer(self, small_real_compounds):
        """Test kNN strategy gracefully falls back to random when featurizer_obj is missing."""
        compounds = small_real_compounds.clone()
        np.random.seed(42)
        predictions = compounds.get_column('Activity').to_numpy() + np.random.normal(0, 0.5, len(compounds))
        compounds = compounds.with_columns(pl.Series('prediction', predictions))

        acq = SimulatedAnnealingAcquisition(
            neighbor_strategy='knn_features',
            n_neighbors=5,
            max_iterations=100,
            random_state=42,
        )
        selected = acq.select(compounds, n_select=8)

        assert len(selected) == 8
        assert len(selected.get_column('ID').unique()) == 8

    def test_knn_neighbor_strategy_reproducible(self, small_real_compounds, small_real_morgan_features):
        """Test kNN neighbor strategy produces reproducible results with fixed seed."""
        compounds = small_real_compounds.clone()
        np.random.seed(42)
        predictions = compounds.get_column('Activity').to_numpy() + np.random.normal(0, 0.5, len(compounds))
        compounds = compounds.with_columns(pl.Series('prediction', predictions))

        from learnm8.features import create_featurizer
        featurizer_obj = create_featurizer('morgan')

        acq1 = SimulatedAnnealingAcquisition(
            neighbor_strategy='knn_features',
            n_neighbors=5,
            max_iterations=100,
            random_state=42,
            featurizer_obj=featurizer_obj,
        )
        acq2 = SimulatedAnnealingAcquisition(
            neighbor_strategy='knn_features',
            n_neighbors=5,
            max_iterations=100,
            random_state=42,
            featurizer_obj=featurizer_obj,
        )

        selected1 = acq1.select(compounds, n_select=5)
        selected2 = acq2.select(compounds, n_select=5)

        assert selected1.get_column('ID').to_list() == selected2.get_column('ID').to_list()

    def test_knn_get_name_includes_neighbor_strategy(self):
        """Test get_name includes neighbor_strategy when not default."""
        acq_default = SimulatedAnnealingAcquisition(
            neighbor_strategy='random',
        )
        assert acq_default.get_name() == "SimulatedAnnealing(exponential_higher)"

        acq_knn = SimulatedAnnealingAcquisition(
            neighbor_strategy='knn_features',
        )
        assert "knn_features" in acq_knn.get_name()

    def test_knn_fallback_does_not_mutate_instance(self, small_real_compounds):
        """The no-featurizer fallback must not permanently downgrade the instance."""
        compounds = small_real_compounds.clone()
        np.random.seed(42)
        predictions = compounds.get_column('Activity').to_numpy() + np.random.normal(
            0, 0.5, len(compounds)
        )
        compounds = compounds.with_columns(pl.Series('prediction', predictions))

        acq = SimulatedAnnealingAcquisition(
            neighbor_strategy='knn_features', max_iterations=50, random_state=42
        )
        acq.select(compounds, n_select=4)
        # Configured strategy is unchanged despite the runtime fallback.
        assert acq.neighbor_strategy == 'knn_features'


@pytest.mark.unit
class TestScoreBandStrategy:
    """Tests for the 'score_band' neighbour strategy (Item 16)."""

    def _pool(self, small_real_compounds):
        compounds = small_real_compounds.clone()
        rng = np.random.default_rng(0)
        predictions = rng.normal(0, 1, len(compounds))
        return compounds.with_columns(pl.Series('prediction', predictions))

    def test_score_band_selects_requested_unique_batch(self, small_real_compounds):
        compounds = self._pool(small_real_compounds)
        acq = SimulatedAnnealingAcquisition(
            neighbor_strategy='score_band', band_width=5, max_iterations=200,
            random_state=42,
        )
        selected = acq.select(compounds, n_select=6)
        assert len(selected) == 6
        assert len(selected.get_column('ID').unique()) == 6
        assert np.all(np.isfinite(selected.get_column('acquisition_score').to_numpy()))

    def test_score_band_is_reproducible(self, small_real_compounds):
        compounds = self._pool(small_real_compounds)
        a = SimulatedAnnealingAcquisition(
            neighbor_strategy='score_band', band_width=5, max_iterations=200,
            random_state=7,
        ).select(compounds, n_select=5)
        b = SimulatedAnnealingAcquisition(
            neighbor_strategy='score_band', band_width=5, max_iterations=200,
            random_state=7,
        ).select(compounds, n_select=5)
        assert a.get_column('ID').to_list() == b.get_column('ID').to_list()

    def test_score_band_needs_no_featurizer(self, small_real_compounds):
        # score_band works purely from the prediction column — no featurizer_obj.
        compounds = self._pool(small_real_compounds)
        acq = SimulatedAnnealingAcquisition(
            neighbor_strategy='score_band', max_iterations=50, random_state=1
        )
        selected = acq.select(compounds, n_select=3)
        assert len(selected) == 3

    def test_proposals_respect_the_rank_window(self, small_real_compounds):
        """A band proposal stays inside +/- band_width ranks and is never self."""
        compounds = self._pool(small_real_compounds)
        predictions = compounds.get_column('prediction').to_numpy()
        acq = SimulatedAnnealingAcquisition(
            neighbor_strategy='score_band', band_width=3, random_state=0
        )
        band_index = acq._build_score_band_index(predictions)
        rank_of_idx = band_index[1]
        origins = np.arange(len(predictions))

        for _ in range(20):
            candidates = acq._propose(
                origins, len(predictions), score_band_index=band_index
            )
            assert np.all(np.abs(rank_of_idx[candidates] - rank_of_idx[origins]) <= 3)
            assert not np.any(candidates == origins)
            assert candidates.min() >= 0
            assert candidates.max() < len(predictions)

    def test_band_width_must_be_positive(self):
        with pytest.raises(ValueError, match='band_width must be >= 1'):
            SimulatedAnnealingAcquisition(neighbor_strategy='score_band', band_width=0)


@pytest.mark.unit
class TestAnnealingCalibration:
    """Calibration reads the proposal neighbourhood, not the global sigma.

    Feature 027: the original defect was a fixed ``initial_temp=1.0`` bearing no
    relation to the target's energy scale. Calibration must therefore measure
    the uphill gap through the *same* proposal path ``select()`` will walk,
    because the gap differs by orders of magnitude between neighbourhoods.
    """

    def _pool(self, medium_real_compounds):
        """200 real compounds with a right-skewed prediction column."""
        rng = np.random.default_rng(0)
        predictions = rng.gamma(shape=2.0, scale=3.0, size=len(medium_real_compounds))
        return medium_real_compounds.clone().with_columns(
            pl.Series('prediction', predictions)
        )

    def test_sample_uphill_deltas_random_is_all_positive(self, medium_real_compounds):
        compounds = self._pool(medium_real_compounds)
        predictions = compounds.get_column('prediction').to_numpy()
        acq = SimulatedAnnealingAcquisition(neighbor_strategy='random', random_state=42)

        deltas = acq._sample_uphill_deltas(predictions)

        assert isinstance(deltas, np.ndarray)
        assert deltas.size > 0
        assert deltas.size <= SA_CALIBRATION_SAMPLES
        assert np.all(deltas > 0)
        assert np.all(np.isfinite(deltas))

    def test_sample_uphill_deltas_score_band_is_all_positive(self, medium_real_compounds):
        compounds = self._pool(medium_real_compounds)
        predictions = compounds.get_column('prediction').to_numpy()
        acq = SimulatedAnnealingAcquisition(
            neighbor_strategy='score_band', band_width=5, random_state=42
        )
        band_index = acq._build_score_band_index(predictions)

        deltas = acq._sample_uphill_deltas(predictions, score_band_index=band_index)

        assert deltas.size > 0
        assert deltas.size <= SA_CALIBRATION_SAMPLES
        assert np.all(deltas > 0)

    def test_sample_uphill_deltas_knn_is_all_positive(self, small_real_compounds):
        """kNN builds a whole-pool NN index, so use the 20-compound fixture."""
        rng = np.random.default_rng(0)
        compounds = small_real_compounds.clone().with_columns(
            pl.Series('prediction', rng.gamma(2.0, 3.0, len(small_real_compounds)))
        )
        predictions = compounds.get_column('prediction').to_numpy()

        from learnm8.features import create_featurizer

        acq = SimulatedAnnealingAcquisition(
            neighbor_strategy='knn_features',
            n_neighbors=5,
            random_state=42,
            featurizer_obj=create_featurizer('morgan'),
        )
        neighbor_map = acq._build_neighbor_map(compounds)
        assert neighbor_map is not None
        neighbor_pad = _pad_neighbor_map(neighbor_map)

        deltas = acq._sample_uphill_deltas(predictions, neighbor_pad=neighbor_pad)

        assert deltas.size > 0
        assert deltas.size <= SA_CALIBRATION_SAMPLES
        assert np.all(deltas > 0)

    def test_score_band_deltas_are_smaller_than_random_deltas(self, medium_real_compounds):
        """The property proving calibration reads the proposal distribution.

        A +/-5 rank window is a genuinely local move, so its mean uphill gap
        must be strictly smaller than a uniform whole-pool draw on the SAME
        predictions. If calibration used the global sigma this would not hold,
        and a single temperature could not serve both neighbourhoods (spec C2).
        """
        compounds = self._pool(medium_real_compounds)
        predictions = compounds.get_column('prediction').to_numpy()

        acq_random = SimulatedAnnealingAcquisition(
            neighbor_strategy='random', random_state=42
        )
        acq_band = SimulatedAnnealingAcquisition(
            neighbor_strategy='score_band', band_width=5, random_state=42
        )
        band_index = acq_band._build_score_band_index(predictions)

        random_deltas = acq_random._sample_uphill_deltas(predictions)
        band_deltas = acq_band._sample_uphill_deltas(
            predictions, score_band_index=band_index
        )

        assert band_deltas.mean() < random_deltas.mean()

    def test_calibrate_temperature_hits_the_target_ratio(self):
        """Ben-Ameur fixed point must land on the REALIZED mean acceptance.

        The closed form ``T = E[dE]/-ln(chi)`` seeds only: ``exp`` is convex, so
        by Jensen ``E[exp(-dE/T)] > exp(-E[dE]/T)`` and the closed form
        overshoots chi. This test pins the realized value, not the seed.
        """
        rng = np.random.default_rng(0)
        # Right-skewed gaps with mean ~6.0, matching the measured AmpC
        # E[dE+] quoted in the spec, where the closed form's overshoot is
        # larger than the tolerance.
        dE_plus = rng.exponential(scale=6.0, size=5000)

        temperature = _calibrate_temperature(dE_plus, SA_TARGET_UPHILL_ACCEPTANCE)

        realized = float(np.exp(-dE_plus / temperature).mean())
        assert abs(realized - SA_TARGET_UPHILL_ACCEPTANCE) < SA_CALIBRATION_TOL

    def test_calibrate_temperature_beats_the_closed_form_seed(self):
        """The seed alone misses the tolerance; the fixed point must be run.

        On an AmpC-shaped dE+ the closed-form seed realizes chi ~ 0.818 against
        a 0.8 target -- an overshoot of ~0.018, nearly twice
        ``SA_CALIBRATION_TOL``. Pinning both halves is what stops a future
        'simplification' from dropping the iteration and keeping only the seed.
        """
        rng = np.random.default_rng(1)
        dE_plus = rng.exponential(scale=6.0, size=5000)

        seed_temp = float(dE_plus.mean() / -np.log(SA_TARGET_UPHILL_ACCEPTANCE))
        seed_realized = float(np.exp(-dE_plus / seed_temp).mean())
        calibrated_realized = float(
            np.exp(
                -dE_plus / _calibrate_temperature(dE_plus, SA_TARGET_UPHILL_ACCEPTANCE)
            ).mean()
        )

        # Jensen: the convex exp makes the seed overshoot, and by more than the
        # tolerance the feature is specified against.
        assert seed_realized > SA_TARGET_UPHILL_ACCEPTANCE + SA_CALIBRATION_TOL
        # The fixed point recovers it.
        assert abs(calibrated_realized - SA_TARGET_UPHILL_ACCEPTANCE) < SA_CALIBRATION_TOL

    def test_calibrate_temperature_handles_the_degenerate_empty_case(self):
        """REQ-5: no uphill move sampled must not divide by zero or raise."""
        temperature = _calibrate_temperature(
            np.array([], dtype=np.float64), SA_TARGET_UPHILL_ACCEPTANCE
        )

        assert np.isfinite(temperature)
        assert temperature > 0

    def test_calibrate_temperature_is_scale_equivariant(self):
        """T has the units of energy, so scaling dE must scale T identically.

        This is the dimensional property whose absence allowed the original
        bug: a fixed ``initial_temp=1.0`` cannot be right for two targets whose
        energy scales differ by 100x.
        """
        rng = np.random.default_rng(2)
        dE_plus = rng.gamma(shape=2.0, scale=3.0, size=5000)

        base = _calibrate_temperature(dE_plus, SA_TARGET_UPHILL_ACCEPTANCE)
        scaled = _calibrate_temperature(100.0 * dE_plus, SA_TARGET_UPHILL_ACCEPTANCE)

        assert scaled == pytest.approx(100.0 * base, rel=1e-6)


PRODUCTION_POOL_SIZE = 100_000


@pytest.fixture(scope='module')
def production_scale_pool(medium_real_compounds) -> pl.DataFrame:
    """A >=100k-row pool for the production-shape regression tests (spec C4).

    Constitution §3.4 red line 1 forbids generating synthetic *molecular data*.
    Nothing here fabricates a structure: the SMILES column cycles the 200 real
    molecules in ``tests/data/medium_molecules.csv``, and the tests that use
    this fixture run ``'random'`` and ``'score_band'``, which never read SMILES
    at all. Only the ``prediction`` column is generated, from a seeded
    right-skewed distribution matched to AmpC's shape (``E[dE+] ~ 6``). This is
    a scale harness for a numerical routine, not a chemical claim.
    """
    smiles = medium_real_compounds.get_column('SMILES').to_list()
    repeats = -(-PRODUCTION_POOL_SIZE // len(smiles))  # ceil division

    rng = np.random.default_rng(20270819)
    return pl.DataFrame(
        {
            'ID': [f'SCALE_{i:07d}' for i in range(PRODUCTION_POOL_SIZE)],
            'SMILES': (smiles * repeats)[:PRODUCTION_POOL_SIZE],
            'prediction': rng.exponential(scale=6.0, size=PRODUCTION_POOL_SIZE),
        }
    )


@pytest.mark.slow
@pytest.mark.unit
class TestProductionShape:
    """Regressions at the scale the framework actually runs at.

    Every test here is one the pre-027 implementation would have failed. Their
    absence is why an arm labelled 'simulated_annealing' shipped reproducing
    greedy's top-k at 98.6-99.8% per cycle.
    """

    @staticmethod
    def _acq(strategy='random', **kwargs):
        params = {'band_width': 50} if strategy == 'score_band' else {}
        params.update(kwargs)
        return SimulatedAnnealingAcquisition(
            neighbor_strategy=strategy, random_state=42, **params
        )

    @pytest.mark.parametrize('strategy', ['random', 'score_band'])
    def test_measured_acceptance_at_calibrated_temperature(
        self, production_scale_pool, strategy
    ):
        """The realized uphill acceptance AT T_0 must hit the target ratio.

        This measures the quantity REQ-3 calibrates: acceptance at the
        calibrated initial temperature, on proposals drawn from the chain's
        starting distribution. It deliberately does NOT average over a window
        of the cooling schedule -- across such a window the temperature decays
        and the chain relocates, so the realized rate is a different quantity
        that no choice of T_0 can pin to 0.8.

        Against the pre-027 fixed initial_temp=1.0 the same measurement on this
        pool gives 0.162 versus 0.823 at the derived T_0 of ~24.9, so this is
        the assertion that would have caught the frozen chain.
        """
        predictions = production_scale_pool.get_column('prediction').to_numpy()
        acq = self._acq(strategy)
        band_index = (
            acq._build_score_band_index(predictions)
            if strategy == 'score_band'
            else None
        )

        schedule = acq._derive_schedule(
            predictions, n_select=1000, score_band_index=band_index
        )

        deltas = acq._sample_deltas(predictions, score_band_index=band_index)
        uphill = deltas[deltas > 0]
        accepted = acq._acceptance_mask(uphill, schedule.initial_temp)

        assert float(accepted.mean()) == pytest.approx(
            SA_TARGET_UPHILL_ACCEPTANCE, abs=0.05
        )

    def test_measured_acceptance_knn_on_a_small_pool(self, small_real_compounds):
        """kNN builds a whole-pool NN index, so it gets the 20-compound pool."""
        from learnm8.features import create_featurizer

        rng = np.random.default_rng(0)
        compounds = small_real_compounds.clone().with_columns(
            pl.Series(
                'prediction', rng.exponential(6.0, len(small_real_compounds))
            )
        )
        predictions = compounds.get_column('prediction').to_numpy()

        acq = SimulatedAnnealingAcquisition(
            neighbor_strategy='knn_features',
            n_neighbors=5,
            random_state=42,
            featurizer_obj=create_featurizer('morgan'),
        )
        neighbor_pad = _pad_neighbor_map(acq._build_neighbor_map(compounds))

        schedule = acq._derive_schedule(
            predictions, n_select=5, neighbor_pad=neighbor_pad
        )
        deltas = acq._sample_deltas(predictions, neighbor_pad=neighbor_pad)
        uphill = deltas[deltas > 0]

        accepted = acq._acceptance_mask(uphill, schedule.initial_temp)
        assert float(accepted.mean()) == pytest.approx(
            SA_TARGET_UPHILL_ACCEPTANCE, abs=0.05
        )

    def test_scale_invariance(self, production_scale_pool):
        """The property whose absence allowed the bug.

        A fixed initial_temp=1.0 is meaningful only for a target whose energy
        scale happens to be ~1. Multiplying the target by 100, shifting it by
        1000, or negating it with a matching score_direction flip must not
        change which compounds are selected.
        """
        base = production_scale_pool
        predictions = base.get_column('prediction').to_numpy()

        def ids_for(new_predictions, score_direction='higher'):
            pool = base.with_columns(pl.Series('prediction', new_predictions))
            acq = SimulatedAnnealingAcquisition(
                random_state=42, score_direction=score_direction
            )
            return acq.select(pool, n_select=1000).get_column('ID').to_list()

        reference = ids_for(predictions)

        assert ids_for(100.0 * predictions) == reference, 'not invariant to scaling'
        assert ids_for(predictions + 1000.0) == reference, 'not invariant to shifting'
        assert ids_for(-predictions, score_direction='lower') == reference, (
            'not invariant to negation with a matching score_direction'
        )

    @pytest.mark.parametrize('n_select', [1000, 10000])
    def test_production_backfill_below_ten_percent(
        self, production_scale_pool, n_select
    ):
        """REQ-12 threshold must not be tripped by a normal production run."""
        acq = self._acq()

        acq.select(production_scale_pool, n_select=n_select)

        assert acq.last_backfill_fraction < SA_BACKFILL_WARN

    def test_differs_from_greedy(self, production_scale_pool):
        """The regression whose absence let the defect ship.

        The archived A-08 arm reproduced greedy's top-k at 98.6-99.8% every
        cycle and no test noticed, because no test compared the two.
        """
        n_select = 1000
        predictions = production_scale_pool.get_column('prediction').to_numpy()

        selected = self._acq().select(production_scale_pool, n_select=n_select)
        sa_ids = set(selected.get_column('ID').to_list())

        greedy_indices = np.argsort(-predictions, kind='stable')[:n_select]
        greedy_ids = set(
            production_scale_pool.get_column('ID').to_numpy()[greedy_indices].tolist()
        )

        overlap = len(sa_ids & greedy_ids) / n_select
        assert overlap < 0.90, (
            f'SA reproduced greedy at {overlap:.1%} overlap -- it is not '
            f'exploring, it is doing greedy top-k under another name'
        )

    def test_determinism_at_production_shape(self, production_scale_pool):
        first = self._acq().select(production_scale_pool, n_select=1000)
        second = self._acq().select(production_scale_pool, n_select=1000)

        assert first.get_column('ID').to_list() == second.get_column('ID').to_list()
        assert np.allclose(
            first.get_column('acquisition_score').to_numpy(),
            second.get_column('acquisition_score').to_numpy(),
        )

        different_seed = SimulatedAnnealingAcquisition(random_state=123).select(
            production_scale_pool, n_select=1000
        )
        assert (
            different_seed.get_column('ID').to_list()
            != first.get_column('ID').to_list()
        ), 'a different seed must explore a different trajectory'


@pytest.mark.unit
class TestBackfillPolicy:
    """Residual greedy backfill must be impossible to produce silently.

    The original defect filled 98% of every batch with ``argsort(prediction)``
    and said nothing, so an arm labelled 'simulated_annealing' reproduced
    greedy's top-k. These tests pin the reporting contract (decision D3).
    """

    @staticmethod
    def _pool(n: int = 5000) -> pl.DataFrame:
        rng = np.random.default_rng(0)
        return pl.DataFrame(
            {
                'ID': [f'C{i}' for i in range(n)],
                'SMILES': ['CCO'] * n,
                'prediction': rng.exponential(scale=6.0, size=n),
            }
        )

    def test_fraction_is_none_before_select_and_zero_on_a_healthy_run(self):
        acq = SimulatedAnnealingAcquisition(
            random_state=42, neighbor_strategy='score_band', band_width=50
        )
        # Always present, so core/cycle.py can read it unconditionally.
        assert acq.last_backfill_fraction is None

        acq.select(self._pool(), n_select=200)

        assert acq.last_backfill_fraction == 0.0

    def test_moderate_backfill_logs_a_warning_naming_the_fraction(self, caplog):
        acq = SimulatedAnnealingAcquisition(
            random_state=42, neighbor_strategy='score_band', band_width=50
        )

        with caplog.at_level(
            logging.WARNING, logger='learnm8.acquisition.simulated_annealing'
        ):
            acq.select(self._pool(), n_select=500)

        assert 0.10 < acq.last_backfill_fraction < 0.50
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, 'a backfill above SA_BACKFILL_WARN must log'
        assert 'backfill' in caplog.text.lower()
        assert '500' in caplog.text  # names n_select

    def test_severe_backfill_raises_acquisition_error(self):
        from learnm8.exceptions import AcquisitionError

        acq = SimulatedAnnealingAcquisition(random_state=42, max_iterations=10)

        with pytest.raises(AcquisitionError, match='backfill'):
            acq.select(self._pool(), n_select=50)

    def test_full_pool_path_reports_zero_not_one(self):
        """REQ-14: no annealing attempted means nothing was degraded."""
        pool = self._pool(n=40)
        acq = SimulatedAnnealingAcquisition(random_state=42)

        selected = acq.select(pool, n_select=len(pool))

        assert len(selected) == len(pool)
        assert acq.last_backfill_fraction == 0.0

    def test_full_pool_path_is_also_best_first(self):
        """REQ-15 covers select() as a whole, not just the annealed path."""
        pool = self._pool(n=40)
        selected = SimulatedAnnealingAcquisition(random_state=42).select(
            pool, n_select=len(pool)
        )

        scores = selected.get_column('acquisition_score').to_numpy()
        assert np.all(np.diff(scores) <= 0)
        # The score must still track the right row after reordering.
        assert np.allclose(scores, selected.get_column('prediction').to_numpy())

    def test_over_request_also_reports_zero(self):
        pool = self._pool(n=40)
        acq = SimulatedAnnealingAcquisition(random_state=42)

        acq.select(pool, n_select=len(pool) + 10)

        assert acq.last_backfill_fraction == 0.0


@pytest.mark.unit
class TestAcceptanceMask:
    """The Metropolis rule has ONE implementation (REQ-18).

    ``_metropolis_accept`` is a thin scalar wrapper over the vectorized
    ``_acceptance_mask``, so the two can never drift apart.
    """

    def test_metropolis_accept_keeps_its_scalar_contract(self):
        acq = SimulatedAnnealingAcquisition(random_state=42)

        # Downhill is always accepted.
        assert acq._metropolis_accept(5.0, 3.0, 1.0) is True
        # Equal energy is not uphill, so also accepted.
        assert acq._metropolis_accept(3.0, 3.0, 1.0) is True
        # Uphill at zero temperature is never accepted.
        assert acq._metropolis_accept(3.0, 5.0, 0.0) is False

    def test_acceptance_mask_returns_bool_array_of_matching_shape(self):
        acq = SimulatedAnnealingAcquisition(random_state=42)
        deltas = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

        mask = acq._acceptance_mask(deltas, 1.0)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_
        assert mask.shape == deltas.shape

    def test_acceptance_mask_is_all_true_for_non_positive_deltas(self):
        acq = SimulatedAnnealingAcquisition(random_state=42)
        deltas = np.array([-5.0, -1.0, 0.0, -0.001])

        assert np.all(acq._acceptance_mask(deltas, 0.5))
        # Even at zero temperature: downhill is unconditional.
        assert np.all(acq._acceptance_mask(deltas, 0.0))

    def test_acceptance_mask_rejects_all_uphill_at_zero_temperature(self):
        acq = SimulatedAnnealingAcquisition(random_state=42)
        deltas = np.array([0.5, 1.0, 10.0])

        assert not np.any(acq._acceptance_mask(deltas, 0.0))

    def test_acceptance_mask_matches_the_target_rate_on_uphill_moves(self):
        """The mask must realize exp(-dE/T), not merely be monotone in it."""
        acq = SimulatedAnnealingAcquisition(random_state=42)
        delta = 2.0
        temperature = 3.0
        deltas = np.full(200_000, delta)

        rate = float(acq._acceptance_mask(deltas, temperature).mean())

        assert rate == pytest.approx(np.exp(-delta / temperature), abs=0.005)

    def test_empty_input_yields_empty_mask(self):
        acq = SimulatedAnnealingAcquisition(random_state=42)
        mask = acq._acceptance_mask(np.array([]), 1.0)
        assert mask.shape == (0,)
        assert mask.dtype == np.bool_


@pytest.mark.unit
class TestStepBudgetDerivation:
    """The step budget is derived, never a hard-coded multiplier (REQ-4)."""

    def test_step_budget_grows_monotonically_with_n_select(self):
        budgets = [
            _derive_step_budget(n_select=n, pool_size=100_000, mean_acceptance=0.2)
            for n in (100, 1_000, 10_000, 50_000)
        ]

        assert budgets == sorted(budgets)
        assert len(set(budgets)) == len(budgets)  # strictly increasing
        assert all(b > 0 for b in budgets)

    def test_budget_grows_as_measured_acceptance_falls(self):
        """A chain that rejects more proposals needs proportionally more steps.

        This is the relation that was broken: chi-bar was estimated from
        uniformly-drawn pairs at ~0.66 while the walk actually ran at ~0.21,
        under-budgeting 'random' by ~3x and dumping the shortfall into greedy
        backfill.
        """
        budgets = [
            _derive_step_budget(n_select=1000, pool_size=100_000, mean_acceptance=chi)
            for chi in (0.8, 0.4, 0.2, 0.1)
        ]

        assert budgets == sorted(budgets)
        # Halving acceptance roughly doubles the budget.
        assert budgets[2] == pytest.approx(2 * budgets[1], rel=0.01)

    def test_unique_visit_inversion_exceeds_the_naive_ratio(self):
        """Near exhaustion, coupon-collector cost dominates the naive U/chi-bar.

        A walk revisits compounds it has already seen, so ``n_select`` unique
        compounds costs strictly more than ``n_select`` accepted moves. The
        naive ratio ignores that and under-budgets exactly where backfill would
        then paper over the shortfall.
        """
        pool_size = 100_000
        n_select = 90_000  # 90% exhaustion

        budget = _derive_step_budget(
            n_select=n_select, pool_size=pool_size, mean_acceptance=1.0
        )

        # At a perfect acceptance rate the naive answer would be exactly
        # n_select; the inversion must still demand substantially more.
        assert budget > n_select
        # Coupon-collector at 90% exhaustion costs ~ln(10) = 2.3x the naive count.
        assert budget > 2.0 * n_select

    def test_budget_clamps_at_ceiling_and_warns(self, caplog):
        """REQ-9: clamp rather than allocate an unbounded visit matrix."""
        with caplog.at_level(
            logging.WARNING, logger='learnm8.acquisition.simulated_annealing'
        ):
            budget = _derive_step_budget(
                n_select=999_999, pool_size=1_000_000, mean_acceptance=0.2
            )

        assert budget == SA_MAX_TOTAL_STEPS
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)
        assert 'SA_MAX_TOTAL_STEPS' in caplog.text or 'clamp' in caplog.text.lower()

    def test_budget_is_at_least_one_step(self):
        """A single-compound request must still walk."""
        assert (
            _derive_step_budget(n_select=1, pool_size=1000, mean_acceptance=0.5) >= 1
        )

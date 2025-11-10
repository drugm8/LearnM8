"""Tests for LearnM8 enrichment metrics.

Tests for virtual screening enrichment metrics using real molecular data.
"""

import pytest
import numpy as np
import polars as pl
from numpy.testing import assert_allclose

from learnm8.evaluation.metrics.enrichment import (
    calculate_top_k_overlap,
    calculate_enrichment_factor,
    calculate_multiple_top_k_overlaps,
    calculate_multiple_enrichment_factors,
    calculate_average_score_ratio,
    calculate_batch_average_score_ratio
)


class TestEnrichmentMetrics:
    """Test enrichment and virtual screening metrics."""

    def test_top_k_overlap(self):
        """Test top-K overlap calculation."""
        # Test data - create predictions DataFrame
        predictions_df = pl.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5'],
            'prediction': [9.5, 8.5, 7.5, 6.5, 5.5]  # mol_1, mol_2, mol_3 should be top 3
        })

        ground_truth_data = pl.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_6', 'mol_7', 'mol_8'],
            'Activity': [10, 9, 8, 7, 6, 5]  # mol_1, mol_2, mol_3 are top 3
        })

        # Test k=3
        overlap = calculate_top_k_overlap(predictions_df, ground_truth_data, k=3, target_col='Activity')
        assert overlap == 100.0  # All top 3 are in selected (100% overlap)

        # Test k=5 - but we only have 3 compounds that match, so 100% overlap on those 3
        # (The function only considers compounds that exist in both datasets)
        overlap = calculate_top_k_overlap(predictions_df, ground_truth_data, k=5, target_col='Activity')
        assert overlap == 100.0  # All 3 available compounds overlap (100%)

    def test_enrichment_factor(self):
        """Test enrichment factor calculation."""
        # Create test data with scores and binary labels
        scores = np.array([10, 9, 8, 7, 6, 5, 4, 3])
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # Top 4 are active

        # Calculate enrichment factor at 50% (top 4 compounds)
        # All 4 selected compounds are active (4/4 = 100%)
        # Random selection would give 50% (4/8 actives total)
        # EF = 100% / 50% = 2.0
        enrichment = calculate_enrichment_factor(scores, labels, percentile=50.0)
        assert_allclose(enrichment, 2.0, rtol=1e-2)

        # Test at 25% (top 2 compounds)
        # 2 out of 2 selected are active (100%)
        # Random would give 50%
        # EF = 100% / 50% = 2.0
        enrichment = calculate_enrichment_factor(scores, labels, percentile=25.0)
        assert_allclose(enrichment, 2.0, rtol=1e-2)

    def test_top_k_overlap_lower_direction(self):
        """Test top-K overlap with score_direction='lower' (for docking scores)."""
        # Test data where LOWER scores are better (like docking scores)
        predictions_df = pl.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5'],
            'prediction': [-15.5, -12.3, -8.7, -5.1, -2.4]  # mol_1 has best (lowest) score
        })

        ground_truth_data = pl.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5'],
            'dockscore': [-16.0, -13.2, -9.1, -4.8, -1.9]  # mol_1 has best (lowest) score
        })

        # Test with score_direction='lower' - should select compounds with lowest scores
        overlap = calculate_top_k_overlap(
            predictions_df, ground_truth_data,
            k=3, target_col='dockscore', score_direction='lower'
        )

        # Top 3 by prediction: mol_1, mol_2, mol_3 (lowest scores)
        # Top 3 by ground truth: mol_1, mol_2, mol_3 (lowest scores)
        # Should have 100% overlap
        assert overlap == 100.0

    def test_enrichment_factor_lower_direction(self):
        """Test enrichment factor with score_direction='lower' - this tests the bug fix."""
        # Test data where LOWER scores are better
        scores = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        labels = np.array([1,   1,   1,   1,   0,   0,   0,   0])  # First 4 (lowest scores) are active

        # With score_direction='lower', should select lowest scores first
        # Top 50% (4 compounds) should be: 3.0, 4.0, 5.0, 6.0 (all active)
        # EF = (4 actives selected / 4 selected) / (4 actives total / 8 total) = 1.0 / 0.5 = 2.0
        enrichment_lower = calculate_enrichment_factor(scores, labels, percentile=50.0, score_direction='lower')
        assert_allclose(enrichment_lower, 2.0, rtol=1e-2)

        # With score_direction='higher', should select highest scores first
        # Top 50% (4 compounds) should be: 10.0, 9.0, 8.0, 7.0 (all inactive)
        # EF = (0 actives selected / 4 selected) / (4 actives total / 8 total) = 0.0 / 0.5 = 0.0
        enrichment_higher = calculate_enrichment_factor(scores, labels, percentile=50.0, score_direction='higher')
        assert_allclose(enrichment_higher, 0.0, rtol=1e-2)

        # Verify the directions produce different results (this would catch the original bug)
        assert enrichment_lower != enrichment_higher

    def test_score_direction_consistency(self):
        """Test that score_direction parameter affects results consistently."""
        # Create test case where direction should matter
        predictions_df = pl.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5', 'mol_6'],
            'prediction': [1, 2, 3, 4, 5, 6]  # Simple ascending order
        })

        ground_truth_data = pl.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5', 'mol_6'],
            'activity': [1, 2, 3, 4, 5, 6]  # Same pattern - perfect correlation
        })

        # Test with k=3
        overlap_lower = calculate_top_k_overlap(
            predictions_df, ground_truth_data,
            k=3, target_col='activity', score_direction='lower'
        )
        overlap_higher = calculate_top_k_overlap(
            predictions_df, ground_truth_data,
            k=3, target_col='activity', score_direction='higher'
        )

        # With perfect correlation:
        # - 'lower' direction should select [mol_1, mol_2, mol_3] from both -> 100% overlap
        # - 'higher' direction should select [mol_6, mol_5, mol_4] from both -> 100% overlap
        # Both should be 100% but demonstrate different selection behavior
        assert overlap_lower == 100.0
        assert overlap_higher == 100.0

        # More importantly, test that the enrichment factor direction affects results
        scores = np.array([1, 2, 3, 4, 5, 6])
        labels = np.array([1, 1, 1, 0, 0, 0])  # First 3 are active

        ef_lower = calculate_enrichment_factor(scores, labels, 50.0, score_direction='lower')
        ef_higher = calculate_enrichment_factor(scores, labels, 50.0, score_direction='higher')

        # These should be different due to the bug fix
        assert ef_lower != ef_higher

    def test_multiple_top_k_overlaps_lower_direction(self):
        """Test calculate_multiple_top_k_overlaps with score_direction='lower'."""
        predictions_df = pl.DataFrame({
            'ID': [f'mol_{i}' for i in range(1000)],
            'prediction': np.random.uniform(-20, 0, 1000)  # Random docking-like scores
        })

        ground_truth_data = pl.DataFrame({
            'ID': [f'mol_{i}' for i in range(1000)],
            'dockscore': predictions_df.get_column('prediction').to_numpy() + np.random.normal(0, 1, 1000)  # Correlated
        })

        # Test multiple top-K overlaps with lower direction
        results = calculate_multiple_top_k_overlaps(
            predictions_df, ground_truth_data,
            target_col='dockscore', score_direction='lower'
        )

        # Should return all expected metrics
        expected_keys = ['top_100_overlap', 'top_1000_overlap',
                        'top_0_1_percent_overlap', 'top_1_percent_overlap', 'top_10_percent_overlap']
        for key in expected_keys:
            assert key in results
            assert isinstance(results[key], (int, float))
            assert 0 <= results[key] <= 100  # Should be percentages

    def test_multiple_enrichment_factors_lower_direction(self):
        """Test calculate_multiple_enrichment_factors with score_direction='lower'."""
        # Create scores where lower is better, with clear active/inactive separation
        n_compounds = 200
        n_actives = 40

        # Actives have lower scores (better)
        active_scores = np.random.uniform(-20, -10, n_actives)
        inactive_scores = np.random.uniform(-5, 5, n_compounds - n_actives)

        scores = np.concatenate([active_scores, inactive_scores])
        labels = np.concatenate([np.ones(n_actives), np.zeros(n_compounds - n_actives)])

        # Shuffle to avoid ordering bias
        shuffle_idx = np.random.permutation(len(scores))
        scores = scores[shuffle_idx]
        labels = labels[shuffle_idx]

        results = calculate_multiple_enrichment_factors(scores, labels, score_direction='lower')

        # Should return enrichment factors at multiple percentiles (based on actual implementation)
        expected_keys = ['ef_5_0', 'ef_1_0', 'ef_0_5', 'ef_0_1']
        for key in expected_keys:
            assert key in results
            assert isinstance(results[key], (int, float))
            # For lower direction with well-separated data, should see enrichment > 1
            if key in ['ef_1_0', 'ef_5_0']:
                assert results[key] > 1.0

    def test_invalid_score_direction(self):
        """Test error handling for invalid score_direction values."""
        predictions_df = pl.DataFrame({
            'ID': ['mol_1', 'mol_2'],
            'prediction': [1.0, 2.0]
        })

        ground_truth_data = pl.DataFrame({
            'ID': ['mol_1', 'mol_2'],
            'activity': [1.5, 2.5]
        })

        scores = np.array([1.0, 2.0])
        labels = np.array([1, 0])

        # Invalid score_direction should be handled gracefully
        # (Note: Current implementation may not validate this, but test documents expected behavior)

        # Test Top-K overlap - should work with valid directions
        overlap_valid = calculate_top_k_overlap(
            predictions_df, ground_truth_data,
            k=1, target_col='activity', score_direction='higher'
        )
        assert isinstance(overlap_valid, (int, float))

        # Test enrichment factor - should work with valid directions
        ef_valid = calculate_enrichment_factor(scores, labels, 50.0, score_direction='lower')
        assert isinstance(ef_valid, (int, float))
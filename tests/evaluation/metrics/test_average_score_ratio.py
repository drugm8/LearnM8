"""Tests for average score ratio metrics with various score types.

Tests the calculate_average_score_ratio and calculate_batch_average_score_ratio
functions with positive scores, negative scores, and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from learnm8.evaluation.metrics.enrichment import (
    calculate_average_score_ratio,
    calculate_batch_average_score_ratio
)


class TestAverageScoreRatio:
    """Test average score ratio calculation with various score types."""

    def test_negative_scores_lower_direction_better_selection(self):
        """Test with negative docking scores where lower is better - better selection."""
        ground_truth = pd.DataFrame({
            'ID': ['A', 'B', 'C', 'D', 'E', 'F'],
            'dockscore': [-50.0, -45.0, -40.0, -30.0, -25.0, -20.0]
        })

        selected_ids = {'A', 'B', 'C'}

        ratio = calculate_average_score_ratio(
            selected_ids, ground_truth, 'dockscore', 'lower'
        )

        selected_mean = -45.0
        population_mean = -35.0
        expected_ratio = abs(selected_mean) / abs(population_mean)
        assert ratio > 1.0
        assert_allclose(ratio, expected_ratio, rtol=0.01)

    def test_negative_scores_lower_direction_worse_selection(self):
        """Test with negative docking scores where lower is better - worse selection."""
        ground_truth = pd.DataFrame({
            'ID': ['A', 'B', 'C', 'D', 'E', 'F'],
            'dockscore': [-50.0, -45.0, -40.0, -30.0, -25.0, -20.0]
        })

        selected_ids = {'D', 'E', 'F'}

        ratio = calculate_average_score_ratio(
            selected_ids, ground_truth, 'dockscore', 'lower'
        )

        expected_ratio = abs(-25.0) / abs(-35.0)
        assert ratio < 1.0
        assert_allclose(ratio, expected_ratio, rtol=0.01)

    def test_negative_scores_lower_direction_baseline(self):
        """Test with negative scores - baseline selection similar to mean."""
        ground_truth = pd.DataFrame({
            'ID': ['A', 'B', 'C', 'D', 'E', 'F'],
            'dockscore': [-50.0, -40.0, -35.0, -30.0, -25.0, -20.0]
        })

        selected_ids = {'B', 'C', 'D'}

        ratio = calculate_average_score_ratio(
            selected_ids, ground_truth, 'dockscore', 'lower'
        )

        assert 0.95 < ratio < 1.10

    def test_positive_scores_higher_direction_better_selection(self):
        """Test with positive activity scores where higher is better - better selection."""
        ground_truth = pd.DataFrame({
            'ID': ['A', 'B', 'C', 'D', 'E', 'F'],
            'activity': [0.9, 0.8, 0.7, 0.5, 0.4, 0.3]
        })

        selected_ids = {'A', 'B', 'C'}

        ratio = calculate_average_score_ratio(
            selected_ids, ground_truth, 'activity', 'higher'
        )

        expected_ratio = 0.8 / 0.6
        assert ratio > 1.0
        assert_allclose(ratio, expected_ratio, rtol=0.01)

    def test_positive_scores_higher_direction_worse_selection(self):
        """Test with positive activity scores where higher is better - worse selection."""
        ground_truth = pd.DataFrame({
            'ID': ['A', 'B', 'C', 'D', 'E', 'F'],
            'activity': [0.9, 0.8, 0.7, 0.5, 0.4, 0.3]
        })

        selected_ids = {'D', 'E', 'F'}

        ratio = calculate_average_score_ratio(
            selected_ids, ground_truth, 'activity', 'higher'
        )

        expected_ratio = 0.4 / 0.6
        assert ratio < 1.0
        assert_allclose(ratio, expected_ratio, rtol=0.01)

    def test_positive_scores_lower_direction(self):
        """Test with positive scores but lower is better (e.g., RMSD)."""
        ground_truth = pd.DataFrame({
            'ID': ['A', 'B', 'C', 'D', 'E', 'F'],
            'rmsd': [1.2, 1.5, 2.0, 2.5, 3.0, 3.5]
        })

        selected_ids = {'A', 'B', 'C'}

        ratio = calculate_average_score_ratio(
            selected_ids, ground_truth, 'rmsd', 'lower'
        )

        expected_ratio = abs(1.567) / abs(2.283)
        assert ratio < 1.0
        assert_allclose(ratio, expected_ratio, rtol=0.05)

    def test_batch_vs_cumulative_consistency(self):
        """Test that batch and cumulative functions behave consistently."""
        ground_truth = pd.DataFrame({
            'ID': ['A', 'B', 'C', 'D', 'E', 'F'],
            'dockscore': [-50.0, -45.0, -40.0, -30.0, -25.0, -20.0]
        })

        selected_ids = {'A', 'B', 'C'}
        selected_df = ground_truth[ground_truth['ID'].isin(selected_ids)]

        cumulative_ratio = calculate_average_score_ratio(
            selected_ids, ground_truth, 'dockscore', 'lower'
        )

        batch_ratio = calculate_batch_average_score_ratio(
            selected_df, ground_truth, 'dockscore', 'lower'
        )

        assert_allclose(cumulative_ratio, batch_ratio, rtol=0.01)

    def test_empty_selection(self):
        """Test with empty selection returns 1.0."""
        ground_truth = pd.DataFrame({
            'ID': ['A', 'B', 'C'],
            'dockscore': [-50.0, -40.0, -30.0]
        })

        ratio = calculate_average_score_ratio(
            set(), ground_truth, 'dockscore', 'lower'
        )
        assert ratio == 1.0

        empty_df = pd.DataFrame({'ID': [], 'dockscore': []})
        batch_ratio = calculate_batch_average_score_ratio(
            empty_df, ground_truth, 'dockscore', 'lower'
        )
        assert batch_ratio == 1.0

    def test_real_world_docking_example(self):
        """Test with real-world docking score scenario from validation data."""
        ground_truth = pd.DataFrame({
            'ID': [f'compound_{i}' for i in range(100)],
            'dockscore': np.random.uniform(-80, -20, 100)
        })

        population_mean = ground_truth['dockscore'].mean()

        better_compounds = ground_truth.nsmallest(10, 'dockscore')
        selected_ids = set(better_compounds['ID'])

        ratio = calculate_average_score_ratio(
            selected_ids, ground_truth, 'dockscore', 'lower'
        )

        assert ratio > 1.0

        selected_mean = better_compounds['dockscore'].mean()
        expected_ratio = abs(selected_mean) / abs(population_mean)
        assert_allclose(ratio, expected_ratio, rtol=0.01)

    def test_near_zero_scores(self):
        """Test with scores near zero (edge case)."""
        ground_truth = pd.DataFrame({
            'ID': ['A', 'B', 'C', 'D'],
            'score': [-0.1, -0.05, 0.05, 0.1]
        })

        selected_ids = {'A', 'B'}

        ratio = calculate_average_score_ratio(
            selected_ids, ground_truth, 'score', 'lower'
        )

        assert isinstance(ratio, float)
        assert ratio > 0

        ratio_higher = calculate_average_score_ratio(
            selected_ids, ground_truth, 'score', 'higher'
        )

        assert isinstance(ratio_higher, float)

    def test_mixed_sign_scores_lower_direction(self):
        """Test with mixed positive and negative scores where lower is better."""
        ground_truth = pd.DataFrame({
            'ID': ['A', 'B', 'C', 'D', 'E'],
            'score': [-10.0, -5.0, 0.0, 5.0, 10.0]
        })

        selected_ids = {'A', 'B'}

        ratio = calculate_average_score_ratio(
            selected_ids, ground_truth, 'score', 'lower'
        )

        selected_mean = -7.5
        population_mean = 0.0

        assert ratio > 1.0

    def test_batch_ratio_with_negative_scores(self):
        """Test batch ratio function specifically with negative docking scores."""
        ground_truth = pd.DataFrame({
            'ID': ['A', 'B', 'C', 'D', 'E', 'F'],
            'dockscore': [-60.0, -50.0, -40.0, -30.0, -20.0, -10.0]
        })

        batch_df = ground_truth[ground_truth['ID'].isin(['A', 'B', 'C'])].copy()

        ratio = calculate_batch_average_score_ratio(
            batch_df, ground_truth, 'dockscore', 'lower'
        )

        batch_mean = -50.0
        population_mean = -35.0
        expected_ratio = abs(batch_mean) / abs(population_mean)

        assert ratio > 1.0
        assert_allclose(ratio, expected_ratio, rtol=0.01)

    def test_score_direction_parameter_effect(self):
        """Test that score_direction parameter produces different results."""
        ground_truth = pd.DataFrame({
            'ID': ['A', 'B', 'C', 'D'],
            'score': [-10.0, -5.0, 5.0, 10.0]
        })

        selected_ids = {'A', 'B'}

        ratio_lower = calculate_average_score_ratio(
            selected_ids, ground_truth, 'score', 'lower'
        )

        ratio_higher = calculate_average_score_ratio(
            selected_ids, ground_truth, 'score', 'higher'
        )

        assert ratio_lower != ratio_higher

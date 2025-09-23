"""Tests for LearnM8 performance metrics.

Tests for ML model performance metrics using real molecular data.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from learnm8.evaluation.metrics.performance import (
    calculate_spearman_correlation,
    calculate_average_score,
    calculate_mape
)


class TestPerformanceMetrics:
    """Test ML performance metrics."""

    def test_spearman_correlation_basic(self):
        """Test Spearman correlation with real data patterns."""
        # Perfect positive correlation
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = calculate_spearman_correlation(x, y)
        assert_allclose(result, 1.0, rtol=1e-10)

        # Perfect negative correlation
        y_neg = np.array([10, 8, 6, 4, 2])
        result = calculate_spearman_correlation(x, y_neg)
        assert_allclose(result, -1.0, rtol=1e-10)

        # No correlation (random)
        np.random.seed(42)
        x_rand = np.random.random(100)
        y_rand = np.random.random(100)
        result = calculate_spearman_correlation(x_rand, y_rand)
        assert -1 <= result <= 1

    def test_spearman_correlation_edge_cases(self):
        """Test Spearman correlation edge cases."""
        # Constant values
        x_const = np.array([5, 5, 5, 5])
        y_var = np.array([1, 2, 3, 4])
        result = calculate_spearman_correlation(x_const, y_var)
        assert result == 0.0  # Function returns 0.0 for undefined correlation

        # Empty arrays - function handles gracefully and returns 0.0
        result = calculate_spearman_correlation(np.array([]), np.array([]))
        assert result == 0.0

        # Single value - function handles gracefully and returns 0.0
        result = calculate_spearman_correlation(np.array([1]), np.array([2]))
        assert result == 0.0

    def test_average_score_calculation(self):
        """Test average score calculation."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_average_score(scores)
        assert_allclose(result, 3.0)

        # With NaN values
        scores_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = calculate_average_score(scores_with_nan)
        assert_allclose(result, 3.0)  # Should ignore NaN

    def test_mape_calculation(self):
        """Test Mean Absolute Percentage Error calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])

        mape = calculate_mape(y_true, y_pred)

        # Manual calculation: |1.1-1|/1 + |1.9-2|/2 + |3.2-3|/3 + |3.8-4|/4
        # = 0.1 + 0.05 + 0.067 + 0.05 = 0.267 => 26.7%
        expected = (0.1 + 0.05 + 0.2/3 + 0.05) * 25  # Convert to percentage
        assert_allclose(mape, expected, rtol=1e-2)

    def test_mape_with_zero_values(self):
        """Test MAPE with zero values in ground truth."""
        y_true = np.array([0.0, 2.0, 3.0])
        y_pred = np.array([0.1, 1.9, 3.2])

        # Should handle division by zero gracefully
        mape = calculate_mape(y_true, y_pred)
        assert np.isfinite(mape) or np.isnan(mape)
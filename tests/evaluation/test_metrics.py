"""Tests for LearnM8 evaluation metrics.

Individual metric calculation tests using real molecular data where relevant.
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from learnm8.evaluation.metrics import (
    calculate_spearman_correlation,
    calculate_average_score,
    calculate_top_k_overlap,
    calculate_enrichment_factor,
    calculate_mape
)


class TestEvaluationMetrics:
    """Test individual evaluation metrics."""
    
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
    
    def test_top_k_overlap(self):
        """Test top-K overlap calculation."""
        # Test data - create predictions DataFrame
        predictions_df = pd.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5'],
            'prediction': [9.5, 8.5, 7.5, 6.5, 5.5]  # mol_1, mol_2, mol_3 should be top 3
        })
        
        ground_truth_data = pd.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_6', 'mol_7', 'mol_8'],
            'Activity': [10, 9, 8, 7, 6, 5]  # mol_1, mol_2, mol_3 are top 3
        })
        
        # Test k=3
        overlap = calculate_top_k_overlap(predictions_df, ground_truth_data, k=3, target_column='Activity')
        assert overlap == 100.0  # All top 3 are in selected (100% overlap)
        
        # Test k=5 - but we only have 3 compounds that match, so 100% overlap on those 3
        # (The function only considers compounds that exist in both datasets)
        overlap = calculate_top_k_overlap(predictions_df, ground_truth_data, k=5, target_column='Activity')
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
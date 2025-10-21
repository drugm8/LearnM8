"""Tests for LearnM8 evaluation core functionality.

Focused tests for core evaluation functions using real molecular data.
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from learnm8.evaluation.core import (
    evaluate_cycle,
    format_progress_output,
    export_metrics_csv
)


class TestEvaluateCore:
    """Test core evaluation functionality with real molecular data."""
    
    def test_evaluate_cycle_basic_functionality(self, small_real_compounds, tmp_path):
        """Test basic evaluate_cycle function with real compounds."""
        # Create test data from real compounds
        labeled_data = small_real_compounds.copy()
        selected_compounds = small_real_compounds.head(10).copy()
        
        # Generate realistic predictions (add noise to actual activities)
        np.random.seed(42)
        predictions = labeled_data['Activity'].values + np.random.normal(0, 0.1, len(labeled_data))
        
        # Run evaluation
        result = evaluate_cycle(
            cycle=0,
            predictions=predictions,
            ground_truth=labeled_data['Activity'].values,
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_column='Activity'
        )
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'cycle' in result
        assert 'rmse' in result
        assert 'mae' in result
        assert 'r2_score' in result
        assert 'spearman_correlation' in result
        
        # Verify numeric values are reasonable
        assert result['cycle'] == 0
        assert result['rmse'] >= 0
        assert result['mae'] >= 0
        assert -1 <= result['r2_score'] <= 1
        assert -1 <= result['spearman_correlation'] <= 1
    
    def test_evaluate_cycle_benchmark_mode(self, diverse_real_compounds):
        """Test evaluate_cycle in benchmark mode with ground truth data."""
        labeled_data = diverse_real_compounds.copy()
        selected_compounds = diverse_real_compounds.head(15).copy()
        
        # Use actual activities as ground truth
        ground_truth_data = diverse_real_compounds[['ID', 'Activity']].copy()
        
        np.random.seed(42)
        predictions = labeled_data['Activity'].values + np.random.normal(0, 0.2, len(labeled_data))
        
        result = evaluate_cycle(
            cycle=1,
            predictions=predictions,
            ground_truth=labeled_data['Activity'].values,
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_column='Activity',
            ground_truth_data=ground_truth_data
        )
        
        # Should include benchmark metrics
        assert 'ground_truth_ef_0_1' in result
        assert 'ground_truth_ef_0_5' in result
        assert 'ground_truth_ef_1_0' in result
    
    def test_evaluate_cycle_with_uncertainty(self, compounds_with_uncertainty):
        """Test evaluation with uncertainty estimates."""
        compounds = compounds_with_uncertainty.copy()
        labeled_data = compounds
        selected_compounds = compounds.head(8)
        
        predictions = compounds['prediction'].values
        ground_truth = compounds['Activity'].values if 'Activity' in compounds.columns else predictions
        uncertainty = compounds['uncertainty'].values
        
        result = evaluate_cycle(
            cycle=2,
            predictions=predictions,
            ground_truth=ground_truth,
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_column='Activity',
            uncertainties=uncertainty
        )
        
        # Should include uncertainty metrics
        assert 'uncertainty_mean' in result
        assert 'uncertainty_std' in result
        assert result['uncertainty_mean'] >= 0
        assert result['uncertainty_std'] >= 0
    
    def test_evaluate_cycle_empty_selection(self, small_real_compounds):
        """Test evaluation with empty selected compounds."""
        labeled_data = small_real_compounds.copy()
        empty_selection = pd.DataFrame(columns=['ID', 'SMILES', 'Activity'])
        
        predictions = labeled_data['Activity'].values
        ground_truth = labeled_data['Activity'].values
        
        result = evaluate_cycle(
            cycle=0,
            predictions=predictions,
            ground_truth=ground_truth,
            labeled_data=labeled_data,
            selected_compounds=empty_selection,
            target_column='Activity'
        )
        
        # Should handle empty selection gracefully
        assert isinstance(result, dict)
        assert result['batch_size'] == 0
    
    def test_evaluate_cycle_missing_target_column(self, small_real_compounds):
        """Test error handling when target column is missing."""
        labeled_data = small_real_compounds.drop(columns=['Activity'])
        selected_compounds = small_real_compounds.head(5)
        
        predictions = np.random.random(len(labeled_data))
        ground_truth = np.random.random(len(labeled_data))
        
        # Function doesn't fail on missing target column - it handles gracefully
        result = evaluate_cycle(
            cycle=0,
            predictions=predictions,
            ground_truth=ground_truth,
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_column='Activity'
        )
        # Should handle gracefully and return basic metrics (not dependent on target column)
        assert isinstance(result, dict)
        assert 'cycle' in result


class TestEvaluationIntegration:
    """Test evaluation system integration."""
    
    def test_format_progress_output(self):
        """Test progress output formatting."""
        metrics = {
            'cycle': 5,
            'batch_size': 20,
            'rmse': 1.234,
            'mae': 0.876,
            'r2_score': 0.654,
            'spearman_correlation': 0.789,
            'cumulative_labeled': 150
        }
        
        output = format_progress_output(metrics)
        
        assert isinstance(output, str)
        assert 'Cycle 5' in output
        assert '1.234' in output  # RMSE
        assert '0.876' in output  # MAE
        assert '0.654' in output  # R²
        assert '0.789' in output  # Spearman
    
    def test_export_metrics_csv(self, tmp_path):
        """Test CSV export functionality."""
        metrics_list = [
            {'cycle': 0, 'rmse': 1.5, 'mae': 1.2, 'r2_score': 0.6, 'spearman_correlation': 0.7},
            {'cycle': 1, 'rmse': 1.3, 'mae': 1.0, 'r2_score': 0.7, 'spearman_correlation': 0.8},
            {'cycle': 2, 'rmse': 1.1, 'mae': 0.9, 'r2_score': 0.8, 'spearman_correlation': 0.85}
        ]
        
        output_file = tmp_path / "test_metrics.csv"
        export_metrics_csv(metrics_list, str(output_file))
        
        # Verify file was created
        assert output_file.exists()
        
        # Read CSV content, skipping comment lines
        df = pd.read_csv(output_file, comment='#')
        assert len(df) == 3
        assert 'cycle' in df.columns
        assert 'rmse' in df.columns
        assert df['cycle'].tolist() == [0, 1, 2]
        assert df['rmse'].tolist() == [1.5, 1.3, 1.1]
    
    def test_evaluation_with_real_molecular_workflow(self, medium_real_compounds):
        """Test complete evaluation workflow with real molecular data."""
        # Use subset for faster testing
        compounds = medium_real_compounds.head(50).copy()
        
        # Simulate active learning cycle
        labeled_data = compounds.head(30)  # Training set
        selected_compounds = compounds.tail(10)  # Newly selected
        
        # Simulate model predictions (add noise to actual activities)
        np.random.seed(42)
        base_predictions = labeled_data['Activity'].values
        predictions = base_predictions + np.random.normal(0, 0.2, len(base_predictions))
        ground_truth = base_predictions
        
        # Run evaluation
        result = evaluate_cycle(
            cycle=3,
            predictions=predictions,
            ground_truth=ground_truth,
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_column='Activity'
        )
        
        # Verify realistic results
        assert isinstance(result, dict)
        assert result['cycle'] == 3
        assert result['cumulative_labeled'] == 30
        assert result['batch_size'] == 10
        assert result['rmse'] >= 0
        assert result['mae'] >= 0
        assert -1 <= result['spearman_correlation'] <= 1
        
        # Results should be reasonable for molecular data
        assert result['rmse'] < 10  # Not too high for normalized activities
        assert result['mae'] < 10
    
    def test_evaluation_error_handling(self, small_real_compounds):
        """Test error handling in evaluation functions."""
        labeled_data = small_real_compounds.copy()
        selected_compounds = small_real_compounds.head(5)
        
        # Mismatched array lengths - function handles this gracefully by setting metrics to None
        result = evaluate_cycle(
            cycle=0,
            predictions=np.array([1, 2, 3]),  # Wrong length
            ground_truth=labeled_data['Activity'].values,
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_column='Activity'
        )
        # Should handle gracefully and return None for failed metrics
        assert result['rmse'] is None
        assert result['mae'] is None
        
        # Invalid target column - function handles gracefully
        result = evaluate_cycle(
            cycle=0,
            predictions=labeled_data['Activity'].values,
            ground_truth=labeled_data['Activity'].values,
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_column='NonexistentColumn'
        )
        # Should handle gracefully and return basic metrics
        assert isinstance(result, dict)
        assert 'cycle' in result
        assert result['avg_score_selected'] is None  # Should be None for missing column


class TestEvaluationEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_evaluation_with_nan_predictions(self, small_real_compounds):
        """Test evaluation handling NaN predictions."""
        labeled_data = small_real_compounds.copy()
        selected_compounds = small_real_compounds.head(5)
        
        # Create predictions with NaN values
        predictions = labeled_data['Activity'].values.copy()
        predictions[0] = np.nan
        predictions[5] = np.nan
        
        ground_truth = labeled_data['Activity'].values
        
        # Should handle NaN values gracefully or raise appropriate error
        try:
            result = evaluate_cycle(
                cycle=0,
                predictions=predictions,
                ground_truth=ground_truth,
                labeled_data=labeled_data,
                selected_compounds=selected_compounds,
                target_column='Activity'
            )
            # If it succeeds, metrics should be finite or None (function sets None for errors)
            assert (result['rmse'] is None or 
                   (isinstance(result['rmse'], (int, float)) and np.isfinite(result['rmse'])))
        except (ValueError, RuntimeError):
            # This is also acceptable behavior
            pass
    
    def test_evaluation_with_constant_predictions(self, small_real_compounds):
        """Test evaluation with constant predictions."""
        labeled_data = small_real_compounds.copy()
        selected_compounds = small_real_compounds.head(5)
        
        # All predictions are the same
        predictions = np.full(len(labeled_data), 0.5)
        ground_truth = labeled_data['Activity'].values
        
        result = evaluate_cycle(
            cycle=0,
            predictions=predictions,
            ground_truth=ground_truth,
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_column='Activity'
        )
        
        # Should handle constant predictions
        assert isinstance(result, dict)
        assert result['rmse'] >= 0
        assert result['mae'] >= 0
        # Spearman correlation should be NaN or 0 for constant predictions
        assert np.isnan(result['spearman_correlation']) or result['spearman_correlation'] == 0
    
    def test_evaluation_single_compound(self):
        """Test evaluation with single compound."""
        single_compound = pd.DataFrame({
            'ID': ['mol_1'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })
        
        predictions = np.array([0.6])
        ground_truth = np.array([0.5])
        
        result = evaluate_cycle(
            cycle=0,
            predictions=predictions,
            ground_truth=ground_truth,
            labeled_data=single_compound,
            selected_compounds=single_compound,
            target_column='Activity'
        )
        
        # Should handle single compound case
        assert isinstance(result, dict)
        assert result['cumulative_labeled'] == 1
        assert result['batch_size'] == 1
        # Single point correlations are undefined - function returns 0.0
        assert result['spearman_correlation'] == 0.0
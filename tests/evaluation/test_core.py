"""Tests for LearnM8 evaluation core functionality.

Focused tests for core evaluation functions using real molecular data.
"""

import logging
from unittest.mock import patch

import pytest
import numpy as np
import pandas as pd
import polars as pl
from numpy.testing import assert_allclose

from learnm8.evaluation.core import (
    evaluate_cycle,
    format_progress_output,
    export_metrics_csv
)


@pytest.mark.unit
class TestEvaluateCore:
    """Test core evaluation functionality with real molecular data."""
    
    def test_evaluate_cycle_basic_functionality(self, small_real_compounds, tmp_path):
        """Test basic evaluate_cycle function with real compounds."""
        # Create test data from real compounds
        labeled_data = small_real_compounds.clone()
        selected_compounds = small_real_compounds.head(10).clone()
        
        # Generate realistic predictions (add noise to actual activities)
        np.random.seed(42)
        predictions = labeled_data['Activity'].to_list() + np.random.normal(0, 0.1, len(labeled_data))
        
        # Run evaluation
        result = evaluate_cycle(
            cycle=0,
            predictions=predictions,
            ground_truth=labeled_data['Activity'].to_list(),
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_col='Activity'
        )
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'cycle' in result
        assert 'batch_size' in result
        assert 'cumulative_labeled' in result
        assert 'avg_score_selected' in result

        # Verify numeric values are reasonable
        assert result['cycle'] == 0
        assert result['batch_size'] >= 0
        assert result['cumulative_labeled'] >= 0
    
    def test_evaluate_cycle_benchmark_mode(self, diverse_real_compounds):
        """Test evaluate_cycle in benchmark mode with ground truth data."""
        labeled_data = diverse_real_compounds.clone()
        selected_compounds = diverse_real_compounds.head(15).clone()
        
        # Use actual activities as ground truth
        ground_truth_data = diverse_real_compounds[['ID', 'Activity']].clone()
        
        np.random.seed(42)
        predictions = labeled_data['Activity'].to_list() + np.random.normal(0, 0.2, len(labeled_data))
        
        result = evaluate_cycle(
            cycle=1,
            predictions=predictions,
            ground_truth=labeled_data['Activity'].to_list(),
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_col='Activity',
            ground_truth_data=ground_truth_data
        )
        
        # Should include benchmark metrics
        assert 'ground_truth_ef_0_1' in result
        assert 'ground_truth_ef_0_5' in result
        assert 'ground_truth_ef_1_0' in result
    
    def test_evaluate_cycle_with_uncertainty(self, compounds_with_uncertainty):
        """Test evaluation with uncertainty estimates."""
        compounds = compounds_with_uncertainty.clone()
        labeled_data = compounds
        selected_compounds = compounds.head(8)
        
        predictions = compounds['prediction'].to_list()
        ground_truth = compounds['Activity'].to_list() if 'Activity' in compounds.columns else predictions
        uncertainty = compounds['uncertainty'].to_list()
        
        result = evaluate_cycle(
            cycle=2,
            predictions=predictions,
            ground_truth=ground_truth,
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_col='Activity',
            uncertainties=uncertainty
        )
        
        # Should include uncertainty metrics
        assert 'uncertainty_mean' in result
        assert 'uncertainty_std' in result
        assert result['uncertainty_mean'] >= 0
        assert result['uncertainty_std'] >= 0
    
    def test_evaluate_cycle_empty_selection(self, small_real_compounds):
        """Test evaluation with empty selected compounds."""
        labeled_data = small_real_compounds.clone()
        empty_selection = pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8, 'Activity': pl.Float64})
        
        predictions = labeled_data['Activity'].to_list()
        ground_truth = labeled_data['Activity'].to_list()
        
        result = evaluate_cycle(
            cycle=0,
            predictions=predictions,
            ground_truth=ground_truth,
            labeled_data=labeled_data,
            selected_compounds=empty_selection,
            target_col='Activity'
        )
        
        # Should handle empty selection gracefully
        assert isinstance(result, dict)
        assert result['batch_size'] == 0
    
    def test_evaluate_cycle_missing_target_col(self, small_real_compounds):
        """Test error handling when target column is missing."""
        labeled_data = small_real_compounds.drop('Activity')
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
            target_col='Activity'
        )
        # Should handle gracefully and return basic metrics (not dependent on target column)
        assert isinstance(result, dict)
        assert 'cycle' in result


@pytest.mark.unit
class TestEvaluationIntegration:
    """Test evaluation system integration."""
    
    def test_format_progress_output(self):
        """Test progress output formatting."""
        metrics = {
            'cycle': 5,
            'batch_size': 20,
            'avg_score_selected': 1.234,
            'ground_truth_avg_score': 0.876,
            'cumulative_labeled': 150,
            'uncertainty_mean': 0.321
        }

        output = format_progress_output(metrics)

        assert isinstance(output, str)
        assert 'Cycle 5' in output
        assert '20' in output  # batch_size
        assert '150' in output  # cumulative_labeled
    
    def test_export_metrics_csv(self, tmp_path):
        """Test CSV export functionality."""
        metrics_list = [
            {'cycle': 0, 'batch_size': 10, 'cumulative_labeled': 10, 'avg_score_selected': 1.5},
            {'cycle': 1, 'batch_size': 10, 'cumulative_labeled': 20, 'avg_score_selected': 1.3},
            {'cycle': 2, 'batch_size': 10, 'cumulative_labeled': 30, 'avg_score_selected': 1.1}
        ]
        
        output_file = tmp_path / "test_metrics.csv"
        export_metrics_csv(metrics_list, str(output_file))
        
        # Verify file was created
        assert output_file.exists()
        
        # Read CSV content, skipping comment lines
        df = pd.read_csv(output_file, comment='#')
        assert len(df) == 3
        assert 'cycle' in df.columns
        assert 'batch_size' in df.columns
        assert df['cycle'].to_list() == [0, 1, 2]
        assert df['batch_size'].to_list() == [10, 10, 10]
    
    def test_evaluation_with_real_molecular_workflow(self, medium_real_compounds):
        """Test complete evaluation workflow with real molecular data."""
        # Use subset for faster testing
        compounds = medium_real_compounds.head(50).clone()
        
        # Simulate active learning cycle
        labeled_data = compounds.head(30)  # Training set
        selected_compounds = compounds.tail(10)  # Newly selected
        
        # Simulate model predictions (add noise to actual activities)
        np.random.seed(42)
        base_predictions = labeled_data['Activity'].to_list()
        predictions = base_predictions + np.random.normal(0, 0.2, len(base_predictions))
        ground_truth = base_predictions
        
        # Run evaluation
        result = evaluate_cycle(
            cycle=3,
            predictions=predictions,
            ground_truth=ground_truth,
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_col='Activity'
        )
        
        # Verify realistic results
        assert isinstance(result, dict)
        assert result['cycle'] == 3
        assert result['cumulative_labeled'] == 30
        assert result['batch_size'] == 10

        # Check basic metrics are present
        assert 'avg_score_selected' in result or result['avg_score_selected'] is None
    
    def test_evaluation_error_handling(self, small_real_compounds):
        """Test error handling in evaluation functions."""
        labeled_data = small_real_compounds.clone()
        selected_compounds = small_real_compounds.head(5)
        
        # Mismatched array lengths - function handles this gracefully by setting metrics to None
        result = evaluate_cycle(
            cycle=0,
            predictions=np.array([1, 2, 3]),  # Wrong length
            ground_truth=labeled_data['Activity'].to_list(),
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_col='Activity'
        )
        # Should handle gracefully and return basic metrics
        assert isinstance(result, dict)
        assert 'cycle' in result
        
        # Invalid target column - function handles gracefully
        result = evaluate_cycle(
            cycle=0,
            predictions=labeled_data['Activity'].to_list(),
            ground_truth=labeled_data['Activity'].to_list(),
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_col='NonexistentColumn'
        )
        # Should handle gracefully and return basic metrics
        assert isinstance(result, dict)
        assert 'cycle' in result
        assert result['avg_score_selected'] is None  # Should be None for missing column


@pytest.mark.unit
class TestEvaluationEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_evaluation_with_nan_predictions(self, small_real_compounds):
        """Test evaluation handling NaN predictions."""
        labeled_data = small_real_compounds.clone()
        selected_compounds = small_real_compounds.head(5)
        
        # Create predictions with NaN values
        predictions = list(labeled_data['Activity'].to_list())
        predictions[0] = np.nan
        predictions[5] = np.nan

        ground_truth = list(labeled_data['Activity'].to_list())
        
        # Should handle NaN values gracefully or raise appropriate error
        try:
            result = evaluate_cycle(
                cycle=0,
                predictions=predictions,
                ground_truth=ground_truth,
                labeled_data=labeled_data,
                selected_compounds=selected_compounds,
                target_col='Activity'
            )
            # If it succeeds, should return valid structure
            assert isinstance(result, dict)
            assert 'cycle' in result
        except (ValueError, RuntimeError):
            # This is also acceptable behavior
            pass
    
    def test_evaluation_with_constant_predictions(self, small_real_compounds):
        """Test evaluation with constant predictions."""
        labeled_data = small_real_compounds.clone()
        selected_compounds = small_real_compounds.head(5)
        
        # All predictions are the same
        predictions = np.full(len(labeled_data), 0.5)
        ground_truth = labeled_data['Activity'].to_list()
        
        result = evaluate_cycle(
            cycle=0,
            predictions=predictions,
            ground_truth=ground_truth,
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_col='Activity'
        )
        
        # Should handle constant predictions
        assert isinstance(result, dict)
        assert 'cycle' in result
        assert 'batch_size' in result
    
    def test_evaluation_single_compound(self):
        """Test evaluation with single compound."""
        single_compound = pl.DataFrame({
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
            target_col='Activity'
        )
        
        # Should handle single compound case
        assert isinstance(result, dict)
        assert result['cumulative_labeled'] == 1
        assert result['batch_size'] == 1


@pytest.mark.unit
class TestSpecificExceptionHandling:
    """Test that evaluation uses specific exception types, not bare except Exception."""

    def _make_base_args(self):
        labeled = pl.DataFrame({
            'ID': [f'mol_{i}' for i in range(20)],
            'SMILES': ['CCO'] * 20,
            'Activity': np.random.default_rng(42).random(20).tolist(),
        })
        selected = labeled.head(5)
        predictions = np.random.default_rng(42).random(20)
        ground_truth = labeled['Activity'].to_numpy()
        return labeled, selected, predictions, ground_truth

    def test_ground_truth_avg_score_catches_value_error(self, caplog):
        labeled, selected, predictions, ground_truth = self._make_base_args()
        gt_data = labeled.clone()

        call_count = [0]
        original_fn = __import__('learnm8.evaluation.metrics.performance', fromlist=['calculate_average_score']).calculate_average_score

        def fail_on_second_call(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise ValueError('test value error')
            return original_fn(*args, **kwargs)

        with patch(
            'learnm8.evaluation.core.calculate_average_score',
            side_effect=fail_on_second_call,
        ):
            with caplog.at_level(logging.WARNING):
                result = evaluate_cycle(
                    cycle=1, predictions=predictions, ground_truth=ground_truth,
                    labeled_data=labeled, selected_compounds=selected,
                    target_col='Activity', ground_truth_data=gt_data,
                )
            assert result['ground_truth_avg_score'] is None
            assert 'ground_truth_avg_score' in caplog.text

    def test_ground_truth_avg_score_catches_type_error(self, caplog):
        labeled, selected, predictions, ground_truth = self._make_base_args()
        gt_data = labeled.clone()

        call_count = [0]
        original_fn = __import__('learnm8.evaluation.metrics.performance', fromlist=['calculate_average_score']).calculate_average_score

        def fail_on_second_call(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise TypeError('bad type')
            return original_fn(*args, **kwargs)

        with patch(
            'learnm8.evaluation.core.calculate_average_score',
            side_effect=fail_on_second_call,
        ):
            with caplog.at_level(logging.WARNING):
                result = evaluate_cycle(
                    cycle=1, predictions=predictions, ground_truth=ground_truth,
                    labeled_data=labeled, selected_compounds=selected,
                    target_col='Activity', ground_truth_data=gt_data,
                )
            assert result['ground_truth_avg_score'] is None

    def test_uncertainty_catches_value_error(self, caplog):
        labeled, selected, predictions, ground_truth = self._make_base_args()
        bad_uncertainties = np.array(['not', 'a', 'number', 'list', 'x'])

        with caplog.at_level(logging.WARNING):
            result = evaluate_cycle(
                cycle=1, predictions=predictions, ground_truth=ground_truth,
                labeled_data=labeled, selected_compounds=selected,
                target_col='Activity', uncertainties=bad_uncertainties,
            )
        assert result['uncertainty_mean'] is None
        assert result['uncertainty_std'] is None

    def test_molecular_similarity_catches_runtime_error(self, caplog):
        labeled, selected, predictions, ground_truth = self._make_base_args()

        with patch(
            'learnm8.evaluation.core.calculate_molecular_similarity_metrics',
            side_effect=RuntimeError('RDKit internal error'),
        ):
            with caplog.at_level(logging.WARNING):
                result = evaluate_cycle(
                    cycle=1, predictions=predictions, ground_truth=ground_truth,
                    labeled_data=labeled, selected_compounds=selected,
                    target_col='Activity',
                )
            assert result['intra_batch_diversity'] is None
            assert 'molecular similarity' in caplog.text.lower()

    def test_discovery_metrics_catches_zero_division(self, caplog):
        labeled, selected, predictions, ground_truth = self._make_base_args()
        gt_data = labeled.clone()
        pool_df = pl.DataFrame({
            'ID': [f'mol_{i}' for i in range(20)],
            'prediction': np.random.default_rng(42).random(20).tolist(),
        })
        cum_ids = {f'mol_{i}' for i in range(5)}

        with patch(
            'learnm8.evaluation.core.calculate_multiple_top_k_discovery_rates',
            side_effect=ZeroDivisionError('division by zero'),
        ):
            with caplog.at_level(logging.WARNING):
                result = evaluate_cycle(
                    cycle=1, predictions=predictions, ground_truth=ground_truth,
                    labeled_data=labeled, selected_compounds=selected,
                    target_col='Activity', oracle_type='benchmark',
                    ground_truth_data=gt_data, pool_df=pool_df,
                    cumulative_selected_ids=cum_ids,
                )
            assert result.get('top_10_discovery') is None
            assert 'discovery' in caplog.text.lower()

    def test_unlabeled_ranking_catches_value_error(self, caplog):
        labeled, selected, predictions, ground_truth = self._make_base_args()
        gt_data = labeled.clone()
        pool_df = pl.DataFrame({
            'ID': [f'mol_{i}' for i in range(20)],
            'prediction': np.random.default_rng(42).random(20).tolist(),
        })
        cum_ids = {f'mol_{i}' for i in range(5)}

        with patch(
            'learnm8.evaluation.core.calculate_multiple_unlabeled_top_k_overlaps',
            side_effect=ValueError('bad value'),
        ):
            with caplog.at_level(logging.WARNING):
                result = evaluate_cycle(
                    cycle=1, predictions=predictions, ground_truth=ground_truth,
                    labeled_data=labeled, selected_compounds=selected,
                    target_col='Activity', oracle_type='benchmark',
                    ground_truth_data=gt_data, pool_df=pool_df,
                    cumulative_selected_ids=cum_ids,
                )
            assert result.get('unlabeled_spearman_correlation') is None
            assert 'unlabeled' in caplog.text.lower() or 'ranking' in caplog.text.lower()

    def test_ground_truth_ef_catches_type_error(self, caplog):
        labeled, selected, predictions, ground_truth = self._make_base_args()
        gt_data = labeled.clone()

        with patch(
            'learnm8.evaluation.core.calculate_ground_truth_enrichment_factors',
            side_effect=TypeError('bad ef input'),
        ):
            with caplog.at_level(logging.WARNING):
                result = evaluate_cycle(
                    cycle=1, predictions=predictions, ground_truth=ground_truth,
                    labeled_data=labeled, selected_compounds=selected,
                    target_col='Activity', ground_truth_data=gt_data,
                )
            assert result.get('ground_truth_ef_5_0') is None

    def test_warning_messages_include_metric_context(self, caplog):
        labeled, selected, predictions, ground_truth = self._make_base_args()
        gt_data = labeled.clone()

        call_count = [0]
        original_fn = __import__('learnm8.evaluation.metrics.performance', fromlist=['calculate_average_score']).calculate_average_score

        def fail_on_second_call(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise ValueError('test')
            return original_fn(*args, **kwargs)

        with patch(
            'learnm8.evaluation.core.calculate_average_score',
            side_effect=fail_on_second_call,
        ):
            with caplog.at_level(logging.WARNING):
                evaluate_cycle(
                    cycle=1, predictions=predictions, ground_truth=ground_truth,
                    labeled_data=labeled, selected_compounds=selected,
                    target_col='Activity', ground_truth_data=gt_data,
                )
            assert 'ground_truth_avg_score' in caplog.text
            assert 'Check input data quality' in caplog.text


@pytest.mark.unit
class TestEvaluateCycleRegression:
    """Exact-value regression fixtures for evaluate_cycle.

    These tests capture the baseline metric values BEFORE any refactoring.
    All numeric assertions use atol=1e-8 so the rounded outputs must be
    bit-for-bit identical after refactoring.
    """

    @pytest.fixture(scope='class')
    def fd(self):
        rng = np.random.default_rng(0)
        n_total = 10_000
        n_actives = 1_000
        ids = [f'cmp_{i:05d}' for i in range(n_total)]
        activity = np.array([1.0 if i < n_actives else 0.0 for i in range(n_total)])
        ground_truth_data = pl.DataFrame({'ID': ids, 'Activity': activity.tolist()})
        cumulative_selected_ids = {f'cmp_{i:05d}' for i in range(500)}
        labeled_data = ground_truth_data.head(500)
        selected_compounds = labeled_data.tail(50)
        predictions = activity[:500] + rng.normal(0, 0.4, 500)
        ground_truth_arr = activity[:500]
        pool_size = n_total - 500
        pool_ids = [f'cmp_{i:05d}' for i in range(500, n_total)]
        pool_activity = activity[500:]
        sw = 0.6
        pool_preds = sw * pool_activity + np.sqrt(1 - sw**2) * rng.normal(0, 1, pool_size)
        pool_df = pl.DataFrame({'ID': pool_ids, 'prediction': pool_preds.tolist()})
        return dict(
            ground_truth_data=ground_truth_data,
            cumulative_selected_ids=cumulative_selected_ids,
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            predictions=predictions,
            ground_truth_arr=ground_truth_arr,
            pool_ids=pool_ids,
            pool_df=pool_df,
        )

    def _call(self, fd, **overrides):
        kwargs = dict(
            cycle=5,
            predictions=fd['predictions'],
            ground_truth=fd['ground_truth_arr'],
            labeled_data=fd['labeled_data'],
            selected_compounds=fd['selected_compounds'],
            target_col='Activity',
            oracle_type='benchmark',
            ground_truth_data=fd['ground_truth_data'],
            pool_df=fd['pool_df'],
            cumulative_selected_ids=fd['cumulative_selected_ids'],
            score_direction='higher',
            disable_molecular_similarity=True,
        )
        kwargs.update(overrides)
        return evaluate_cycle(**kwargs)

    def test_higher_direction_all_metrics(self, fd):
        r = self._call(fd, score_direction='higher')
        assert r['cycle'] == 5
        assert r['batch_size'] == 50
        assert r['cumulative_labeled'] == 500
        assert_allclose(r['avg_score_selected'], 1.0, atol=1e-8)
        assert_allclose(r['ground_truth_avg_score'], 0.1, atol=1e-8)
        assert_allclose(r['ground_truth_ef_0_1'], 10.0, atol=1e-8)
        assert_allclose(r['ground_truth_ef_0_5'], 10.0, atol=1e-8)
        assert_allclose(r['ground_truth_ef_1_0'], 10.0, atol=1e-8)
        assert_allclose(r['ground_truth_ef_5_0'], 10.0, atol=1e-8)
        assert_allclose(r['top_10_discovery'], 100.0, atol=1e-8)
        assert_allclose(r['top_100_discovery'], 100.0, atol=1e-8)
        assert_allclose(r['top_1000_discovery'], 50.0, atol=1e-8)
        assert_allclose(r['top_0_1_pct_discovery'], 100.0, atol=1e-8)
        assert_allclose(r['top_1_pct_discovery'], 100.0, atol=1e-8)
        assert_allclose(r['top_10_pct_discovery'], 50.0, atol=1e-8)
        assert_allclose(r['cumulative_ef'], 10.0, atol=1e-8)
        assert_allclose(r['cumulative_avg_score_ratio'], 10.0, atol=1e-8)
        assert_allclose(r['batch_ef'], 10.0, atol=1e-8)
        assert_allclose(r['batch_hit_rate'], 1.0, atol=1e-8)
        assert_allclose(r['batch_avg_score_ratio'], 10.0, atol=1e-8)
        assert_allclose(r['unlabeled_top_100_overlap'], 4.0, atol=1e-8)
        assert_allclose(r['unlabeled_top_1000_overlap'], 17.5, atol=1e-8)
        assert_allclose(r['unlabeled_ef_1_0'], 4.2, atol=1e-8)
        assert_allclose(r['unlabeled_ef_5_0'], 2.68, atol=1e-8)
        assert_allclose(r['unlabeled_spearman_correlation'], 0.1441, atol=1e-4)

    def test_lower_direction_all_metrics(self, fd):
        r = self._call(fd, score_direction='lower')
        assert_allclose(r['ground_truth_ef_0_1'], 0.0, atol=1e-8)
        assert_allclose(r['ground_truth_ef_0_5'], 0.0, atol=1e-8)
        assert_allclose(r['ground_truth_ef_1_0'], 0.0, atol=1e-8)
        assert_allclose(r['ground_truth_ef_5_0'], 0.0, atol=1e-8)
        assert_allclose(r['top_10_discovery'], 0.0, atol=1e-8)
        assert_allclose(r['top_100_discovery'], 0.0, atol=1e-8)
        assert_allclose(r['top_1000_discovery'], 0.0, atol=1e-8)
        assert_allclose(r['top_0_1_pct_discovery'], 0.0, atol=1e-8)
        assert_allclose(r['top_1_pct_discovery'], 0.0, atol=1e-8)
        assert_allclose(r['top_10_pct_discovery'], 0.0, atol=1e-8)
        assert_allclose(r['unlabeled_ef_1_0'], 0.0, atol=1e-8)
        assert_allclose(r['unlabeled_ef_5_0'], 0.24, atol=1e-8)
        assert_allclose(r['unlabeled_spearman_correlation'], 0.1441, atol=1e-4)
        assert_allclose(r['unlabeled_top_100_overlap'], 2.0, atol=1e-8)
        assert_allclose(r['unlabeled_top_1000_overlap'], 12.7, atol=1e-8)
        assert_allclose(r['batch_hit_rate'], 1.0, atol=1e-8)
        assert_allclose(r['cumulative_ef'], 10.0, atol=1e-8)

    def test_empty_cumulative_selected_ids(self, fd):
        r = self._call(fd, cumulative_selected_ids=set(), cycle=0)
        assert r['cycle'] == 0
        assert_allclose(r['top_10_discovery'], 0.0, atol=1e-8)
        assert_allclose(r['top_100_discovery'], 0.0, atol=1e-8)
        assert_allclose(r['top_1000_discovery'], 0.0, atol=1e-8)
        assert_allclose(r['top_0_1_pct_discovery'], 0.0, atol=1e-8)
        assert_allclose(r['top_1_pct_discovery'], 0.0, atol=1e-8)
        assert_allclose(r['top_10_pct_discovery'], 0.0, atol=1e-8)
        assert r['cumulative_ef'] is None
        assert_allclose(r['cumulative_avg_score_ratio'], 1.0, atol=1e-8)
        assert_allclose(r['unlabeled_spearman_correlation'], 0.1441, atol=1e-4)

    def test_no_activity_column_ef_metrics_are_none(self, fd):
        rng = np.random.default_rng(0)
        n_total = 10_000
        ids = [f'cmp_{i:05d}' for i in range(n_total)]
        score_vals = np.arange(n_total, dtype=float)
        gt_no_act = pl.DataFrame({'ID': ids, 'Score': score_vals.tolist()})
        lab = gt_no_act.head(500)
        sel = lab.tail(50)
        pool_size = n_total - 500
        pool_ids = [f'cmp_{i:05d}' for i in range(500, n_total)]
        pool_scores = score_vals[500:]
        sw = 0.6
        pool_pred = sw * pool_scores + np.sqrt(1 - sw**2) * rng.normal(0, pool_scores.std(), pool_size)
        preds = score_vals[:500] + rng.normal(0, score_vals[:500].std() * 0.3, 500)
        pool_df = pl.DataFrame({'ID': pool_ids, 'prediction': pool_pred.tolist()})

        r = evaluate_cycle(
            cycle=5, predictions=preds, ground_truth=score_vals[:500],
            labeled_data=lab, selected_compounds=sel,
            target_col='Score', oracle_type='benchmark',
            ground_truth_data=gt_no_act,
            pool_df=pool_df,
            cumulative_selected_ids=fd['cumulative_selected_ids'],
            score_direction='higher', disable_molecular_similarity=True,
        )
        assert r['batch_ef'] is None
        assert r['batch_hit_rate'] is None
        assert r['cumulative_ef'] is None
        assert r['unlabeled_ef_1_0'] is None
        assert r['unlabeled_ef_5_0'] is None
        assert r['ground_truth_ef_0_1'] is None
        assert r['ground_truth_ef_0_5'] is None
        assert r['ground_truth_ef_1_0'] is None
        assert r['ground_truth_ef_5_0'] is None
        assert_allclose(r['unlabeled_top_100_overlap'], 7.0, atol=1e-8)
        assert_allclose(r['unlabeled_top_1000_overlap'], 34.4, atol=1e-8)
        assert_allclose(r['unlabeled_spearman_correlation'], 0.614, atol=1e-4)
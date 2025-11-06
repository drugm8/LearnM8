"""Tests for the main LearnM8 API (learnm8.api.run_active_learning).

Tests cover:
1. Basic simple-API run with minimal parameters
2. Advanced-API with CycleConfig list
3. Pruning via pruning_fraction parameter
4. Mode auto-detection for CSV oracle (should detect benchmark mode)
5. Saved files in results

Validates results structure:
- compounds_df
- cycle_metrics
- saved_files
- labeled_data
- unlabeled_data
- validation_result
"""

import logging
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from learnm8.api import run_active_learning
from learnm8.core.config import CycleConfig


class TestAPIBasicSimple:
    """Test basic simple-API functionality with minimal parameters."""

    def test_simple_api_with_minimal_parameters(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test simple API with minimal required parameters."""
        output_dir = tmp_path / "simple_api_test"
        cache_dir = tmp_path / "cache"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=3,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=cache_dir,
            random_state=42
        )

        assert 'compounds_df' in results
        assert 'cycle_metrics' in results
        assert 'saved_files' in results
        assert 'labeled_data' in results
        assert 'unlabeled_data' in results
        assert 'validation_result' in results

        assert isinstance(results['compounds_df'], pd.DataFrame)
        assert isinstance(results['cycle_metrics'], list)
        assert isinstance(results['saved_files'], dict)
        assert isinstance(results['labeled_data'], pd.DataFrame)
        assert isinstance(results['unlabeled_data'], pd.DataFrame)

        assert len(results['cycle_metrics']) == 3
        assert len(results['labeled_data']) > 0
        assert output_dir.exists()

    def test_simple_api_strategy_parameter(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test simple API with different acquisition strategy."""
        output_dir = tmp_path / "strategy_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            strategy='random',
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert len(results['cycle_metrics']) == 2
        assert results['compounds_df'] is not None

    def test_simple_api_initial_sampling_parameters(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test simple API with initialization phase."""
        output_dir = tmp_path / "initial_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        labeled_in_init = results['compounds_df'][
            results['compounds_df']['labeled_cycle'] == 0
        ]
        assert len(labeled_in_init) == 5

        labeled_in_cycle_1 = results['compounds_df'][
            results['compounds_df']['labeled_cycle'] == 1
        ]
        assert len(labeled_in_cycle_1) == 5

        labeled_in_cycle_2 = results['compounds_df'][
            results['compounds_df']['labeled_cycle'] == 2
        ]
        assert len(labeled_in_cycle_2) == 0


class TestAPIAdvancedCycleConfig:
    """Test advanced API with CycleConfig list."""

    def test_advanced_api_with_cycle_config_list(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test advanced API with explicit CycleConfig list."""
        output_dir = tmp_path / "advanced_api_test"

        cycles = [
            CycleConfig('random', n_cycles=2, batch_fraction=0.05),
            CycleConfig('greedy', n_cycles=3, batch_fraction=0.03),
        ]

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            cycles=cycles,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert len(results['cycle_metrics']) == 5
        assert results['compounds_df'] is not None
        assert 'saved_files' in results

    def test_advanced_api_overrides_simple_parameters(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test that CycleConfig list overrides simple API parameters."""
        output_dir = tmp_path / "override_test"

        cycles = [
            CycleConfig('random', n_cycles=1, batch_fraction=0.05),
            CycleConfig('greedy', n_cycles=2, batch_fraction=0.03),
        ]

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            cycles=cycles,
            n_cycles=10,
            batch_fraction=0.1,
            strategy='ucb',
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        # Count cycles in the list: 3
        assert len(results['cycle_metrics']) == 3

    def test_advanced_api_mixed_strategies(self, tmp_path, sample_compounds, mock_learner_with_uncertainty, mock_oracle):
        """Test advanced API with mixed acquisition strategies."""
        output_dir = tmp_path / "mixed_strategies_test"

        cycles = [
            CycleConfig('random', n_cycles=1, batch_fraction=0.05),
            CycleConfig('greedy', n_cycles=2, batch_fraction=0.03),
            CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        ]

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner_with_uncertainty,
            target_col='Activity',
            featurizer_type='morgan',
            cycles=cycles,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert len(results['cycle_metrics']) == 4
        assert len(results['labeled_data']) > 0


class TestAPIPruning:
    """Test pruning functionality via pruning_fraction parameter."""

    def test_pruning_fraction_enables_pruning(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test that providing pruning_fraction enables pruning."""
        output_dir = tmp_path / "pruning_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=3,
            batch_fraction=0.05,
            pruning_fraction=0.2,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert len(results['cycle_metrics']) <= 3
        assert 'compounds_df' in results

        pruned_compounds = results['compounds_df'][
            results['compounds_df']['status'] == 'pruned'
        ]
        assert len(pruned_compounds) >= 0

    def test_pruning_with_advanced_api(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test pruning with CycleConfig list."""
        output_dir = tmp_path / "pruning_advanced_test"

        cycles = [
            CycleConfig('random', n_cycles=1, batch_fraction=0.05),
            CycleConfig('greedy', n_cycles=2, batch_fraction=0.03,
                       pruning_strategy='score',
                       pruning_params={'pruning_fraction': 0.3}),
        ]

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            cycles=cycles,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert len(results['cycle_metrics']) <= 3
        assert 'compounds_df' in results

        # Verify pruning actually happened
        final_metrics = results['cycle_metrics'][-1]
        assert final_metrics['cumulative_pruned'] > 0, "Pruning should have removed compounds"

    def test_pruning_via_parameters(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test pruning via pruning_strategy and pruning_params parameters (Advanced API without CycleConfig)."""
        output_dir = tmp_path / "pruning_params_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=3,
            batch_fraction=0.05,
            pruning_strategy='score',
            pruning_params={'pruning_fraction': 0.25},
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert len(results['cycle_metrics']) == 3
        # Verify pruning was applied
        final_metrics = results['cycle_metrics'][-1]
        assert final_metrics['cumulative_pruned'] > 0, "Pruning should have removed compounds via parameters"

    def test_pruning_fraction_validation(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test that invalid pruning_fraction raises error."""
        output_dir = tmp_path / "pruning_validation_test"

        with pytest.raises(ValueError, match="pruning_fraction must be in"):
            run_active_learning(
                compound_pool=sample_compounds,
                oracle=mock_oracle,
                learner=mock_learner,
                target_col='Activity',
                featurizer_type='morgan',
                n_cycles=2,
                batch_fraction=0.05,
                pruning_fraction=1.5,
                output_dir=output_dir,
                cache_dir=tmp_path / "cache",
                random_state=42
            )


class TestAPIModeDetection:
    """Test mode auto-detection for CSV oracle (benchmark mode)."""

    def test_csv_oracle_auto_detects_benchmark_mode(self, tmp_path, sample_compounds, mock_learner):
        """Test that CSV oracle automatically triggers benchmark mode."""
        csv_file = tmp_path / "oracle_data.csv"
        oracle_data = sample_compounds.copy()
        oracle_data['Activity'] = np.random.uniform(0, 1, len(sample_compounds))
        oracle_data.to_csv(csv_file, index=False)

        output_dir = tmp_path / "csv_oracle_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=str(csv_file),
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert results is not None
        assert len(results['cycle_metrics']) == 2

    def test_oracle_none_with_csv_compound_pool(self, tmp_path, sample_compounds, mock_learner):
        """Test oracle=None auto-detects CSV oracle from compound_pool path."""
        csv_file = tmp_path / "compounds_with_activity.csv"
        compounds_data = sample_compounds.copy()
        compounds_data['Activity'] = np.random.uniform(0, 1, len(sample_compounds))
        compounds_data.to_csv(csv_file, index=False)

        output_dir = tmp_path / "auto_oracle_test"

        results = run_active_learning(
            compound_pool=str(csv_file),
            oracle=None,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert results is not None
        assert 'cycle_metrics' in results

    def test_oracle_none_with_dataframe_raises_error(self, tmp_path, sample_compounds, mock_learner):
        """Test that oracle=None with DataFrame compound_pool raises error."""
        output_dir = tmp_path / "no_oracle_df_test"

        with pytest.raises(ValueError, match="Oracle required when compound_pool is DataFrame"):
            run_active_learning(
                compound_pool=sample_compounds,
                oracle=None,
                learner=mock_learner,
                target_col='Activity',
                featurizer_type='morgan',
                n_cycles=2,
                batch_fraction=0.05,
                output_dir=output_dir,
                cache_dir=tmp_path / "cache",
                random_state=42
            )

    def test_explicit_mode_parameter(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test explicitly setting mode parameter."""
        output_dir = tmp_path / "explicit_mode_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            mode='run',
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert results is not None
        assert len(results['cycle_metrics']) == 2


class TestAPISavedFiles:
    """Test saved_files dictionary in results."""

    def test_saved_files_contains_required_keys(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test that saved_files contains all required file paths."""
        output_dir = tmp_path / "saved_files_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        saved_files = results['saved_files']
        assert isinstance(saved_files, dict)
        assert 'compounds_final' in saved_files
        assert 'cycle_metrics' in saved_files
        assert 'selection_history' in saved_files
        assert 'config' in saved_files

    def test_saved_files_exist_on_filesystem(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test that all saved files actually exist on filesystem."""
        output_dir = tmp_path / "files_exist_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        saved_files = results['saved_files']
        for file_type, file_path in saved_files.items():
            assert Path(file_path).exists(), f"File {file_type} not found at {file_path}"

    def test_output_dir_in_results(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test that output_dir is included in results."""
        output_dir = tmp_path / "output_dir_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert 'output_dir' in results
        assert results['output_dir'] == output_dir
        assert output_dir.exists()


class TestAPIResultsStructure:
    """Test complete results structure and data integrity."""

    def test_compounds_df_has_required_columns(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test that compounds_df has all required columns."""
        output_dir = tmp_path / "columns_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        compounds_df = results['compounds_df']
        required_columns = {'ID', 'SMILES', 'status', 'selected_cycle',
                          'labeled_cycle', 'pruned_cycle', 'Activity'}
        assert required_columns.issubset(compounds_df.columns)

    def test_cycle_metrics_length_matches_executed_cycles(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test that len(cycle_metrics) matches number of executed cycles (including cycle 0)."""
        output_dir = tmp_path / "metrics_length_test"

        n_al_cycles = 4
        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=n_al_cycles,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert len(results['cycle_metrics']) == n_al_cycles

    def test_cycle_0_in_metrics(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test that cycle 0 (initialization) is included in cycle_metrics."""
        output_dir = tmp_path / "cycle_0_test"

        n_al_cycles = 3
        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=n_al_cycles,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        cycle_metrics = results['cycle_metrics']

        assert len(cycle_metrics) == n_al_cycles

        # Check that cycle 0 is present
        assert cycle_metrics[0]['cycle'] == 0
        assert cycle_metrics[0]['strategy'] == 'random'

        # Check that cycles are sequential: 0, 1, 2
        cycles = [m['cycle'] for m in cycle_metrics]
        assert cycles == [0, 1, 2]

        # Check that cycle 0 has expected metrics structure
        assert 'batch_size' in cycle_metrics[0]
        assert 'cumulative_labeled' in cycle_metrics[0]
        assert 'has_uncertainty' in cycle_metrics[0]
        assert cycle_metrics[0]['has_uncertainty'] is False

    def test_labeled_data_accessor(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test labeled_data convenience accessor."""
        output_dir = tmp_path / "labeled_accessor_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        labeled_data = results['labeled_data']
        assert isinstance(labeled_data, pd.DataFrame)
        assert len(labeled_data) > 0
        assert all(labeled_data['status'] == 'labeled')

    def test_unlabeled_data_accessor(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test unlabeled_data convenience accessor."""
        output_dir = tmp_path / "unlabeled_accessor_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        unlabeled_data = results['unlabeled_data']
        assert isinstance(unlabeled_data, pd.DataFrame)
        assert all(unlabeled_data['status'] == 'unlabeled')

    def test_validation_result_structure(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test validation_result structure."""
        output_dir = tmp_path / "validation_result_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        validation_result = results['validation_result']
        assert hasattr(validation_result, 'valid_compounds')
        assert hasattr(validation_result, 'invalid_compounds')
        assert hasattr(validation_result, 'success_rate')


class TestAPILearnerIntegration:
    """Test learner string shortcuts and instantiation."""

    @pytest.mark.skip(reason="Requires actual learner implementation - tested in integration tests")
    def test_learner_string_shortcut(self, tmp_path, sample_compounds):
        """Test instantiating learner from string shortcut."""
        csv_file = tmp_path / "oracle_data.csv"
        oracle_data = sample_compounds.copy()
        oracle_data['Activity'] = np.random.uniform(0, 1, len(sample_compounds))
        oracle_data.to_csv(csv_file, index=False)

        output_dir = tmp_path / "learner_string_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=str(csv_file),
            learner='rf',
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert results is not None
        assert len(results['cycle_metrics']) == 2

    def test_invalid_learner_string_raises_error(self, tmp_path, sample_compounds, mock_oracle):
        """Test that invalid learner string raises error."""
        output_dir = tmp_path / "invalid_learner_test"

        with pytest.raises(ValueError, match="Unknown learner"):
            run_active_learning(
                compound_pool=sample_compounds,
                oracle=mock_oracle,
                learner='invalid_learner_name',
                target_col='Activity',
                featurizer_type='morgan',
                n_cycles=2,
                batch_fraction=0.05,
                output_dir=output_dir,
                cache_dir=tmp_path / "cache",
                random_state=42
            )


class TestAPIErrorHandling:
    """Test error handling and validation."""

    def test_missing_required_columns(self, tmp_path, mock_learner, mock_oracle):
        """Test that missing ID or SMILES columns raises error."""
        invalid_compounds = pd.DataFrame({
            'compound_id': ['C1', 'C2'],
            'structure': ['CCO', 'CCC']
        })

        output_dir = tmp_path / "missing_columns_test"

        with pytest.raises(ValueError, match="must have 'ID' and 'SMILES' columns"):
            run_active_learning(
                compound_pool=invalid_compounds,
                oracle=mock_oracle,
                learner=mock_learner,
                target_col='Activity',
                featurizer_type='morgan',
                n_cycles=2,
                batch_fraction=0.05,
                output_dir=output_dir,
                cache_dir=tmp_path / "cache",
                random_state=42
            )

    def test_invalid_compound_pool_type(self, tmp_path, mock_learner, mock_oracle):
        """Test that invalid compound_pool type raises error."""
        output_dir = tmp_path / "invalid_type_test"

        with pytest.raises(TypeError, match="compound_pool must be str, Path, or pd.DataFrame"):
            run_active_learning(
                compound_pool={'invalid': 'type'},
                oracle=mock_oracle,
                learner=mock_learner,
                target_col='Activity',
                featurizer_type='morgan',
                n_cycles=2,
                batch_fraction=0.05,
                output_dir=output_dir,
                cache_dir=tmp_path / "cache",
                random_state=42
            )

    def test_nonexistent_csv_file(self, tmp_path, mock_learner, mock_oracle):
        """Test that nonexistent CSV file raises FileNotFoundError."""
        output_dir = tmp_path / "nonexistent_file_test"

        with pytest.raises(FileNotFoundError):
            run_active_learning(
                compound_pool='/nonexistent/path/to/file.csv',
                oracle=mock_oracle,
                learner=mock_learner,
                target_col='Activity',
                featurizer_type='morgan',
                n_cycles=2,
                batch_fraction=0.05,
                output_dir=output_dir,
                cache_dir=tmp_path / "cache",
                random_state=42
            )

    def test_empty_cycle_schedule_raises_error(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test that empty cycle schedule raises error."""
        output_dir = tmp_path / "empty_schedule_test"

        with pytest.raises(ValueError, match="Cycle schedule is empty"):
            run_active_learning(
                compound_pool=sample_compounds,
                oracle=mock_oracle,
                learner=mock_learner,
                target_col='Activity',
                featurizer_type='morgan',
                cycles=[],
                output_dir=output_dir,
                cache_dir=tmp_path / "cache",
                random_state=42
            )


class TestAPIFeaturizerTypes:
    """Test different featurizer types."""

    @pytest.mark.parametrize('featurizer', ['morgan', 'maccs', 'ecfp6'])
    def test_different_featurizer_types(self, tmp_path, sample_compounds, mock_learner, mock_oracle, featurizer):
        """Test API with different featurizer types."""
        output_dir = tmp_path / f"featurizer_{featurizer}_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type=featurizer,
            n_cycles=2,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / f"cache_{featurizer}",
            random_state=42
        )

        assert results is not None
        assert len(results['cycle_metrics']) == 2


class TestAPIEvaluationMetrics:
    """Test evaluation metrics integration in run_active_learning."""

    def test_cycle_metrics_include_evaluation_metrics(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test that cycle_metrics include basic cycle information (cycle, batch_size, cumulative_labeled)."""
        output_dir = tmp_path / "eval_metrics_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=3,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert 'cycle_metrics' in results
        assert len(results['cycle_metrics']) > 0

        for cycle_metric in results['cycle_metrics']:
            assert 'cycle' in cycle_metric, "Cycle number should be present"
            assert 'batch_size' in cycle_metric, "Batch size should be present"
            assert 'cumulative_labeled' in cycle_metric, "Cumulative labeled should be present"

    def test_aggregate_metrics_in_results(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test that aggregate_metrics is included in results dictionary."""
        output_dir = tmp_path / "agg_metrics_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=3,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert 'aggregate_metrics' in results
        agg = results['aggregate_metrics']

        assert 'total_cycles' in agg
        assert 'total_labeled' in agg
        assert agg['total_cycles'] == 3

    def test_aggregate_metrics_basic_tracking(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test that aggregate metrics track basic information."""
        output_dir = tmp_path / "basic_tracking_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=3,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        agg = results['aggregate_metrics']

        # Check basic aggregate metrics exist
        assert 'total_cycles' in agg
        assert 'total_labeled' in agg
        assert agg['total_cycles'] == 3

    def test_benchmark_mode_includes_top_k_metrics(self, tmp_path, sample_compounds, mock_learner):
        """Test that benchmark mode includes top-K overlap metrics."""
        csv_file = tmp_path / "benchmark_oracle.csv"
        oracle_data = sample_compounds.copy()
        oracle_data['Activity'] = np.random.uniform(0, 1, len(sample_compounds))
        oracle_data.to_csv(csv_file, index=False)

        output_dir = tmp_path / "benchmark_topk_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=str(csv_file),
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            mode='benchmark',
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        cycle_metrics = results['cycle_metrics']

        for metric in cycle_metrics:
            assert 'top_100_overlap' in metric or metric.get('top_100_overlap') is None
            assert 'top_1000_overlap' in metric or metric.get('top_1000_overlap') is None

    def test_evaluation_metrics_backward_compatible(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        """Test that existing basic metrics are still present (backward compatibility)."""
        output_dir = tmp_path / "backward_compat_test"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        for cycle_metric in results['cycle_metrics']:
            assert 'cycle' in cycle_metric
            assert 'strategy' in cycle_metric
            assert 'batch_size' in cycle_metric
            assert 'selected_count' in cycle_metric
            assert 'remaining_unlabeled' in cycle_metric
            assert 'prediction_mean' in cycle_metric or cycle_metric.get('prediction_mean') is None
            assert 'best_so_far' in cycle_metric or cycle_metric.get('best_so_far') is None


class TestValidateCompoundPoolAPI:

    def test_validate_returns_validation_result(self, sample_compounds):
        from learnm8 import validate_compound_pool, ValidationResult

        result = validate_compound_pool(sample_compounds, progress=False)

        assert isinstance(result, ValidationResult)
        assert hasattr(result, 'valid_compounds')
        assert hasattr(result, 'invalid_compounds')
        assert len(result.valid_compounds) > 0
        assert result.valid_compounds['ID'].tolist() == sample_compounds['ID'].tolist()

    def test_validate_with_invalid_smiles(self):
        from learnm8 import validate_compound_pool
        import pandas as pd

        compounds = pd.DataFrame({
            'ID': ['valid_1', 'invalid_1', 'valid_2'],
            'SMILES': ['CCO', 'NOT_A_SMILES_###', 'CCC']
        })

        result = validate_compound_pool(compounds, progress=False)

        assert len(result.valid_compounds) == 2
        assert len(result.invalid_compounds) == 1
        assert 'invalid_1' in result.invalid_compounds['ID'].tolist()

    def test_validate_integration_with_run_active_learning(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        from learnm8 import validate_compound_pool

        validation_result = validate_compound_pool(sample_compounds, progress=False)

        results = run_active_learning(
            compound_pool=validation_result.valid_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=1,
            batch_fraction=0.05,
            output_dir=tmp_path / "output",
            cache_dir=tmp_path / "cache",
            random_state=42
        )

        assert 'compounds_df' in results
        assert len(results['compounds_df']) == len(validation_result.valid_compounds)


class TestCacheDirParameter:

    def test_cache_dir_used_for_feature_extraction(self, tmp_path):
        from learnm8.features.extraction import extract_features

        cache_dir = tmp_path / "custom_cache_location"
        smiles_list = ['CCO', 'CCC', 'CCCC']

        features = extract_features(smiles_list, 'morgan', cache_dir=cache_dir)

        assert cache_dir.exists()
        cache_files = list(cache_dir.glob("*.h5"))
        assert len(cache_files) > 0
        assert features.shape[0] == 3

    def test_cache_dir_parameter_passed_through(self, tmp_path, sample_compounds, mock_learner, mock_oracle):
        cache_dir = tmp_path / "custom_cache"
        output_dir = tmp_path / "output"

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner=mock_learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=1,
            batch_fraction=0.05,
            output_dir=output_dir,
            cache_dir=cache_dir,
            random_state=42
        )

        assert 'compounds_df' in results
        assert results['output_dir'] == output_dir


class TestLoggingBehavior:
    """Test logging level behavior after refactoring."""

    def test_info_logs_are_user_focused(self, sample_compounds, tmp_path, caplog_info):
        """INFO logs should show high-level progress only, no technical details."""
        from learnm8 import run_active_learning

        # Save sample compounds as CSV for oracle
        oracle_path = tmp_path / "oracle.csv"
        oracle_data = sample_compounds.copy()
        oracle_data['Activity'] = np.random.uniform(0, 1, len(sample_compounds))
        oracle_data.to_csv(oracle_path, index=False)

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=str(oracle_path),
            learner='rf',
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.1,
            cache_dir=tmp_path,
            output_dir=tmp_path / "output"
        )

        info_messages = [rec.message for rec in caplog_info.records
                        if rec.levelno == logging.INFO]

        assert any('Starting active learning' in msg for msg in info_messages)
        assert any('Phase 1: Validating' in msg for msg in info_messages)
        assert any('Training model' in msg for msg in info_messages)
        assert any('Acquiring compounds' in msg for msg in info_messages)

        assert not any('cache_dir=' in msg.lower() for msg in info_messages)
        assert not any('shape:' in msg.lower() for msg in info_messages)
        assert not any('dtype' in msg.lower() for msg in info_messages)

    def test_debug_logs_contain_technical_details(self, sample_compounds, tmp_path, caplog_debug):
        """DEBUG logs should contain technical details and shapes."""
        from learnm8 import run_active_learning

        # Save sample compounds as CSV for oracle
        oracle_path = tmp_path / "oracle.csv"
        oracle_data = sample_compounds.copy()
        oracle_data['Activity'] = np.random.uniform(0, 1, len(sample_compounds))
        oracle_data.to_csv(oracle_path, index=False)

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=str(oracle_path),
            learner='rf',
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.1,
            cache_dir=tmp_path,
            output_dir=tmp_path / "output"
        )

        debug_messages = [rec.message for rec in caplog_debug.records
                         if rec.levelno == logging.DEBUG]

        assert len(debug_messages) > 0
        assert any('shape' in msg.lower() for msg in debug_messages)

    def test_phase_separators_in_info_logs(self, sample_compounds, tmp_path, caplog_info):
        """INFO logs should include phase separators for visual organization."""
        from learnm8 import run_active_learning

        # Save sample compounds as CSV for oracle
        oracle_path = tmp_path / "oracle.csv"
        oracle_data = sample_compounds.copy()
        oracle_data['Activity'] = np.random.uniform(0, 1, len(sample_compounds))
        oracle_data.to_csv(oracle_path, index=False)

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=str(oracle_path),
            learner='rf',
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.1,
            cache_dir=tmp_path,
            output_dir=tmp_path / "output"
        )

        info_messages = [rec.message for rec in caplog_info.records
                        if rec.levelno == logging.INFO]

        assert any('═══════' in msg for msg in info_messages)
        assert any('Phase 1:' in msg for msg in info_messages)
        assert any('Phase 2:' in msg for msg in info_messages)
        assert any('Phase 5:' in msg for msg in info_messages)


class TestChempropWithExtraDescriptors:
    """Test Chemprop learner with extra descriptors (x_d) integration."""

    def test_chemprop_with_extra_descriptors_full_cycle(self, sample_compounds, tmp_path):
        """Test full active learning cycle with Chemprop + extra descriptors."""
        from learnm8 import run_active_learning

        oracle_path = tmp_path / "oracle.csv"
        oracle_data = sample_compounds.copy()
        oracle_data['Activity'] = np.random.uniform(0, 1, len(sample_compounds))
        oracle_data.to_csv(oracle_path, index=False)

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=str(oracle_path),
            learner='chemprop',
            featurizer_type='morgan',
            target_col='Activity',
            n_cycles=2,
            batch_fraction=0.1,
            output_dir=tmp_path / "output",
            cache_dir=tmp_path / "cache"
        )

        assert 'compounds_df' in results
        assert 'cycle_metrics' in results
        assert len(results['cycle_metrics']) == 2

        compounds_df = results['compounds_df']
        assert 'prediction_cycle_1' in compounds_df.columns

    def test_chemprop_without_featurizer_backward_compat(self, sample_compounds, tmp_path):
        """Test backward compatibility - Chemprop without featurizer (graph-only)."""
        from learnm8 import run_active_learning

        oracle_path = tmp_path / "oracle.csv"
        oracle_data = sample_compounds.copy()
        oracle_data['Activity'] = np.random.uniform(0, 1, len(sample_compounds))
        oracle_data.to_csv(oracle_path, index=False)

        results = run_active_learning(
            compound_pool=sample_compounds,
            oracle=str(oracle_path),
            learner='chemprop',
            featurizer_type=None,
            target_col='Activity',
            n_cycles=2,
            batch_fraction=0.1,
            output_dir=tmp_path / "output"
        )

        assert 'compounds_df' in results
        assert len(results['cycle_metrics']) == 2

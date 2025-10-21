"""
Integration tests for core LearnM8 active learning functionality.

Tests the main run_active_learning function with real molecular data,
focusing on integration, functionality, error handling, and data validation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from learnm8 import run_active_learning
from learnm8.core.data_structures import validate_master_dataframe
from learnm8.core.config import CycleConfig
from tests.conftest import MockOracle, MockLearner


class TestRunActiveLearning:
    """Test core run_active_learning function integration."""

    def test_basic_active_learning_integration(self, small_real_compounds, tmp_path, mock_oracle, mock_learner_with_uncertainty):
        """Test basic active learning workflow with real molecular data."""
        compounds = small_real_compounds.copy()

        # Add activity column for testing
        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))

        oracle = mock_oracle
        learner = mock_learner_with_uncertainty
        
        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=3,
            batch_fraction=0.1,
            n_initial=5,
            output_dir=str(tmp_path),
            random_state=42
        )

        # Validate results structure
        assert 'labeled_data' in results
        assert 'unlabeled_data' in results
        assert 'cycle_metrics' in results

        # Validate compounds_df presence and schema
        assert 'compounds_df' in results
        master_df = results['compounds_df']
        assert validate_master_dataframe(master_df) is True
        
        # Validate data integrity
        labeled_data = results['labeled_data']
        unlabeled_data = results['unlabeled_data']
        
        assert len(labeled_data) > 5  # Should have initial + selected compounds
        assert len(unlabeled_data) + len(labeled_data) == len(compounds)
        assert 'Activity' in labeled_data.columns

        # Validate cycle metrics
        cycle_metrics = results['cycle_metrics']
        assert len(cycle_metrics) == 3
        assert len(results['cycle_metrics']) == 3

        for i, metrics in enumerate(cycle_metrics):
            assert metrics['cycle'] == i
            assert metrics['selected_count'] > 0
            assert 'cumulative_labeled' in metrics

        # Backward compatibility: views should match master DataFrame filters
        assert 'ID' in labeled_data.columns
        assert 'SMILES' in labeled_data.columns
        assert 'Activity' in labeled_data.columns
        assert len(labeled_data) + len(unlabeled_data) == len(compounds)
    
    def test_single_cycle_active_learning(self, small_real_compounds, tmp_path, mock_oracle, mock_learner):
        """Test single cycle active learning."""
        compounds = small_real_compounds.copy()

        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))

        oracle = mock_oracle
        learner = mock_learner

        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=1,
            initial_batch_fraction=0.2,  # Use initial_batch_fraction for single cycle
            n_initial=3,
            output_dir=str(tmp_path),
            random_state=42
        )

        assert len(results['cycle_metrics']) == 1
        
        # Check that initial compounds + selected compounds are labeled
        expected_labeled = 3 + max(1, int(len(compounds) * 0.2))
        assert len(results['labeled_data']) >= expected_labeled
    
    def test_advanced_cycles_specification(self, small_real_compounds, tmp_path, mock_oracle_low_noise, mock_learner_with_uncertainty):
        """Test advanced cycles specification with different strategies."""
        compounds = small_real_compounds.copy()

        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))

        oracle = mock_oracle_low_noise
        learner = mock_learner_with_uncertainty

        # Use advanced cycles specification
        cycles = [
            CycleConfig('random', n_cycles=1, batch_fraction=0.05),
            CycleConfig('greedy', n_cycles=1, batch_fraction=0.1),
            CycleConfig('random', n_cycles=1, batch_fraction=0.05)
        ]

        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_col='Activity',
            featurizer_type='morgan',
            cycles=cycles,
            n_initial=3,
            output_dir=str(tmp_path),
            random_state=42
        )

        assert len(results['cycle_metrics']) == 3

        # Verify strategies were used correctly
        expected_strategies = ['random', 'greedy', 'random']
        for i, expected_strategy in enumerate(expected_strategies):
            assert results['cycle_metrics'][i]['strategy'] == expected_strategy
    
    def test_string_learner_creation(self, small_real_compounds, tmp_path, mock_oracle_low_noise):
        """Test creation of learner from string specification."""
        compounds = small_real_compounds.copy()

        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))
        oracle = mock_oracle_low_noise

        # Test with string learner specification
        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner='rf',  # String specification
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.15,
            n_initial=3,
            output_dir=str(tmp_path),
            random_state=42
        )

        assert len(results['cycle_metrics']) == 2
        assert len(results['labeled_data']) > 3
    
    def test_score_direction_handling(self, small_real_compounds, tmp_path, mock_oracle_low_noise, mock_learner):
        """Test both score directions (higher/lower is better)."""
        compounds = small_real_compounds.copy()

        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))
        oracle = mock_oracle_low_noise
        learner = mock_learner
        
        # Test 'higher' direction (default)
        results_higher = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_col='Activity',
            featurizer_type='morgan',
            score_direction='higher',
            n_cycles=1,
            batch_fraction=0.1,
            n_initial=3,
            output_dir=str(tmp_path / 'higher'),
            random_state=42
        )

        # Test 'lower' direction
        results_lower = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_col='Activity',
            featurizer_type='morgan',
            score_direction='lower',
            n_cycles=1,
            batch_fraction=0.1,
            n_initial=3,
            output_dir=str(tmp_path / 'lower'),
            random_state=42
        )

        assert len(results_higher['cycle_metrics']) == 1
        assert len(results_lower['cycle_metrics']) == 1
    
    def test_empty_compound_pool_handling(self, tmp_path):
        """Test error handling with empty compound pool."""
        empty_compounds = pd.DataFrame(columns=['ID', 'SMILES'])
        oracle = Mock(spec=['measure'])
        oracle.measure.return_value = pd.DataFrame(columns=['ID', 'Activity'])
        learner = Mock(spec=['train', 'predict', 'supports_uncertainty'])
        learner.supports_uncertainty.return_value = False
        
        with pytest.raises((ValueError, IndexError)):
            run_active_learning(
                compound_pool=empty_compounds,
                oracle=oracle,
                learner=learner,
                target_col='Activity',
                featurizer_type='morgan',
                n_cycles=1,
                output_dir=str(tmp_path)
            )
    
    def test_invalid_compound_pool_columns(self, tmp_path):
        """Test error handling with missing required columns."""
        # Missing SMILES column
        invalid_compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004', 'COMP_005',
                   'COMP_006', 'COMP_007', 'COMP_008', 'COMP_009', 'COMP_010', 'COMP_011'],
            'structure': ['CCO', 'CCC', 'CCN', 'CO', 'CN', 'C', 'CC', 'CNC', 'COC', 'CCO', 'CCCO']  # Wrong column name
        })

        oracle = Mock(spec=['measure'])
        oracle.measure.return_value = pd.DataFrame(columns=['ID', 'Activity'])
        learner = Mock(spec=['train', 'predict', 'supports_uncertainty'])
        learner.supports_uncertainty.return_value = False
        
        with pytest.raises(KeyError):
            run_active_learning(
                compound_pool=invalid_compounds,
                oracle=oracle,
                learner=learner,
                target_col='Activity',
                featurizer_type='morgan',
                n_cycles=1,
                n_initial=2,  # Small initial size to avoid sampling error
                output_dir=str(tmp_path)
            )
    
    def test_invalid_parameters_validation(self, small_real_compounds, tmp_path):
        """Test validation of invalid parameters."""
        compounds = small_real_compounds.copy()

        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002'],
                'SMILES': ['CCO', 'CCC']
            })

        oracle = Mock(spec=['measure'])
        oracle.measure.return_value = compounds[['ID']].copy()
        oracle.measure.return_value['Activity'] = np.random.uniform(0, 1, len(compounds))
        learner = Mock(spec=['train', 'predict', 'supports_uncertainty'])
        learner.supports_uncertainty.return_value = False
        
        # Test invalid score_direction
        with pytest.raises(ValueError, match="score_direction must be one of"):
            run_active_learning(
                compound_pool=compounds,
                oracle=oracle,
                learner=learner,
                target_col='Activity',
                featurizer_type='morgan',
                score_direction='invalid',
                output_dir=str(tmp_path)
            )

        # Test invalid n_cycles
        with pytest.raises(ValueError, match="n_cycles must be at least 1"):
            run_active_learning(
                compound_pool=compounds,
                oracle=oracle,
                learner=learner,
                target_col='Activity',
                featurizer_type='morgan',
                n_cycles=0,
                output_dir=str(tmp_path)
            )

        # Test invalid batch_fraction
        with pytest.raises(ValueError, match="batch_fraction must be in"):
            run_active_learning(
                compound_pool=compounds,
                oracle=oracle,
                learner=learner,
                target_col='Activity',
                featurizer_type='morgan',
                batch_fraction=1.5,  # > 1
                output_dir=str(tmp_path)
            )

        with pytest.raises(ValueError, match="batch_fraction must be in"):
            run_active_learning(
                compound_pool=compounds,
                oracle=oracle,
                learner=learner,
                target_col='Activity',
                featurizer_type='morgan',
                batch_fraction=0,  # <= 0
                output_dir=str(tmp_path)
            )
    
    def test_oracle_failure_handling(self, small_real_compounds, tmp_path, mock_oracle, mock_learner):
        """Test handling of oracle measurement failures."""
        compounds = small_real_compounds.copy()

        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002'],
                'SMILES': ['CCO', 'CCC']
            })

        oracle = mock_oracle
        learner = mock_learner

        with patch.object(oracle, 'measure', side_effect=RuntimeError("Oracle measurement failed")):
            with pytest.raises(RuntimeError, match="Oracle measurement failed"):
                run_active_learning(
                    compound_pool=compounds,
                    oracle=oracle,
                    learner=learner,
                    target_col='Activity',
                    featurizer_type='morgan',
                    n_cycles=1,
                    n_initial=1,
                    output_dir=str(tmp_path)
                )
    
    def test_learner_failure_handling(self, small_real_compounds, tmp_path, mock_oracle_low_noise):
        """Test handling of learner training and prediction failures."""
        compounds = small_real_compounds.copy()

        # Use a small subset for predictable testing
        if len(compounds) >= 3:
            compounds = compounds.iloc[:3].copy()
        else:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
                'SMILES': ['CCO', 'CCC', 'CCN']
            })

        compounds['Activity'] = [0.1, 0.5, 0.9]

        # Test training failure
        oracle = mock_oracle_low_noise
        failing_learner = MockLearner(fail_training=True)

        with pytest.raises(RuntimeError, match="Training failed"):
            run_active_learning(
                compound_pool=compounds,
                oracle=oracle,
                learner=failing_learner,
                target_col='Activity',
                featurizer_type='morgan',
                n_cycles=1,
                n_initial=1,
                output_dir=str(tmp_path)
            )
    
    def test_csv_export_functionality(self, small_real_compounds, tmp_path, mock_oracle_low_noise, mock_learner_with_uncertainty):
        """Test CSV export functionality."""
        compounds = small_real_compounds.copy()

        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
                'SMILES': ['CCO', 'CCC', 'CCN']
            })

        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))

        oracle = mock_oracle_low_noise
        learner = mock_learner_with_uncertainty

        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.3,
            n_initial=1,
            output_dir=str(tmp_path),
            random_state=42
        )

        # Check that CSV files are mentioned in results
        assert 'saved_files' in results

        # Check that output directory exists
        assert Path(tmp_path).exists()

        # Verify compounds_final.csv exists and is in saved_files
        assert 'compounds_final' in results['saved_files']
        compounds_final_csv = Path(results['saved_files']['compounds_final'])
        assert compounds_final_csv.exists()

        # Read and validate CSV format
        pred_df = pd.read_csv(compounds_final_csv)
        assert 'prediction_cycle_0' in pred_df.columns
        assert 'prediction_cycle_1' in pred_df.columns
        assert 'final_oracle_value' in pred_df.columns or 'Activity' in pred_df.columns

        # Check for NaN values for initially labeled compounds (sparse storage)
        assert pred_df['prediction_cycle_0'].isna().any()
    
    def test_temporary_output_directory(self, small_real_compounds, mock_oracle_low_noise, mock_learner):
        """Test automatic creation of temporary output directory."""
        compounds = small_real_compounds.copy()

        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002'],
                'SMILES': ['CCO', 'CCC']
            })

        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))

        oracle = mock_oracle_low_noise
        learner = mock_learner

        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=1,
            batch_fraction=0.5,
            n_initial=1,
            output_dir=None,  # Should create temporary directory
            random_state=42
        )
        
        # Verify temporary directory was created and is in results
        assert 'output_dir' in results
        assert results['output_dir'] is not None
        assert Path(results['output_dir']).exists()
    
    def test_early_termination_empty_pool(self, tmp_path, mock_oracle_low_noise, mock_learner):
        """Test early termination when pool is exhausted."""
        # Very small compound pool
        compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002'],
            'SMILES': ['CCO', 'CCC'],
            'Activity': [0.1, 0.9]
        })

        oracle = mock_oracle_low_noise
        learner = mock_learner

        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=10,  # More cycles than compounds
            batch_fraction=0.5,
            n_initial=1,
            output_dir=str(tmp_path),
            random_state=42
        )

        # Should terminate early
        assert len(results['cycle_metrics']) < 10
        assert len(results['unlabeled_data']) == 0  # Pool exhausted
    
    def test_different_acquisition_strategies(self, small_real_compounds, tmp_path, mock_oracle_low_noise, mock_learner_with_uncertainty):
        """Test different acquisition strategies work correctly."""
        compounds = small_real_compounds.copy()

        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(10)],
                'SMILES': ['CCO'] * 10  # Simple molecules
            })

        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))

        oracle = mock_oracle_low_noise
        learner = mock_learner_with_uncertainty
        
        strategies_to_test = ['random', 'greedy']
        
        for strategy in strategies_to_test:
            results = run_active_learning(
                compound_pool=compounds,
                oracle=oracle,
                learner=learner,
                target_col='Activity',
                featurizer_type='morgan',
                strategy=strategy,
                n_cycles=2,
                batch_fraction=0.2,
                n_initial=2,
                output_dir=str(tmp_path / strategy),
                random_state=42
            )

            assert len(results['cycle_metrics']) == 2

            # Verify strategies were used correctly
            # Cycle 0 uses initial_strategy (default: 'random')
            # Cycle 1 uses the specified strategy
            assert results['cycle_metrics'][0]['strategy'] == 'random'  # initial_strategy default
            assert results['cycle_metrics'][1]['strategy'] == strategy  # specified strategy


class TestMasterDataFrameIntegration:
    """Test master DataFrame structure and backward compatibility."""

    def test_backward_compatibility_views(self, small_real_compounds, tmp_path, mock_oracle_low_noise, mock_learner_with_uncertainty):
        """Test that returned views match master DataFrame filters."""
        compounds = small_real_compounds.copy()

        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(15)],
                'SMILES': ['CCO'] * 15
            })

        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))

        oracle = mock_oracle_low_noise
        learner = mock_learner_with_uncertainty

        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=3,
            batch_fraction=0.1,
            n_initial=3,
            output_dir=str(tmp_path),
            random_state=42
        )

        labeled_data = results['labeled_data']
        unlabeled_data = results['unlabeled_data']

        # Verify labeled_data contains Activity column (renamed from target_value)
        assert 'Activity' in labeled_data.columns
        assert all(labeled_data['Activity'].notna())

        # Verify unlabeled_data only has ID and SMILES columns
        assert set(unlabeled_data.columns) == {'ID', 'SMILES'}

        # Confirm no overlap between labeled and unlabeled IDs
        labeled_ids = set(labeled_data['ID'])
        unlabeled_ids = set(unlabeled_data['ID'])
        assert len(labeled_ids & unlabeled_ids) == 0

        # Validate sum of lengths equals original pool size
        assert len(labeled_data) + len(unlabeled_data) == len(compounds)

    def test_prediction_history_columns(self, small_real_compounds, tmp_path, mock_oracle_low_noise, mock_learner_with_uncertainty):
        """Test prediction history columns are created correctly."""
        compounds = small_real_compounds.copy()

        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(12)],
                'SMILES': ['CCO'] * 12
            })

        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))

        oracle = mock_oracle_low_noise
        learner = mock_learner_with_uncertainty

        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=3,
            batch_fraction=0.15,
            n_initial=2,
            output_dir=str(tmp_path),
            random_state=42
        )

        # Read compounds_final.csv from saved_files
        assert 'compounds_final' in results['saved_files']
        compounds_final_csv = Path(results['saved_files']['compounds_final'])
        assert compounds_final_csv.exists()

        pred_df = pd.read_csv(compounds_final_csv)

        # Verify prediction columns
        assert 'prediction_cycle_0' in pred_df.columns
        assert 'prediction_cycle_1' in pred_df.columns
        assert 'prediction_cycle_2' in pred_df.columns

        # Verify uncertainty columns
        assert 'uncertainty_cycle_0' in pred_df.columns
        assert 'uncertainty_cycle_1' in pred_df.columns
        assert 'uncertainty_cycle_2' in pred_df.columns

        # Check that labeled compounds have final_oracle_value
        assert 'final_oracle_value' in pred_df.columns
        labeled_compounds = pred_df[pred_df['final_oracle_value'].notna()]
        assert len(labeled_compounds) > 0

        # Validate NaN values for compounds not predicted in certain cycles (sparse storage)
        assert pred_df['prediction_cycle_0'].isna().any()

    def test_status_transitions(self, small_real_compounds, tmp_path, mock_oracle_low_noise, mock_learner):
        """Test status transitions from unlabeled to labeled."""
        compounds = small_real_compounds.copy()

        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(10)],
                'SMILES': ['CCO'] * 10
            })

        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))

        oracle = mock_oracle_low_noise
        learner = mock_learner

        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.2,
            n_initial=2,
            output_dir=str(tmp_path),
            random_state=42
        )

        # Read selection_history.csv from saved_files
        assert 'selection_history' in results['saved_files']
        selection_csv = Path(results['saved_files']['selection_history'])
        assert selection_csv.exists()

        selection_df = pd.read_csv(selection_csv)

        # Verify selected compounds have selected_cycle recorded
        assert 'selected_cycle' in selection_df.columns
        assert all(selection_df['selected_cycle'].notna())

        # Check that selection counts match
        total_selected = len(selection_df)
        initial_size = 2
        cycle_0_batch = max(1, int(len(compounds) * 0.2))
        cycle_1_batch = max(1, int(len(compounds) * 0.2))

        assert total_selected >= cycle_0_batch + cycle_1_batch

    def test_sparse_storage_nan_values(self, small_real_compounds, tmp_path, mock_oracle_low_noise, mock_learner_with_uncertainty):
        """Test sparse storage pattern with NaN values."""
        compounds = small_real_compounds.copy()

        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(8)],
                'SMILES': ['CCO'] * 8
            })

        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))

        oracle = mock_oracle_low_noise
        learner = mock_learner_with_uncertainty

        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=2,
            batch_fraction=0.25,
            n_initial=2,
            output_dir=str(tmp_path),
            random_state=42
        )

        # Read compounds_final.csv and selection_history.csv from saved_files
        assert 'compounds_final' in results['saved_files']
        assert 'selection_history' in results['saved_files']
        pred_df = pd.read_csv(results['saved_files']['compounds_final'])
        sel_df = pd.read_csv(results['saved_files']['selection_history'])

        # Identify initially labeled compounds (first 2)
        initial_ids = compounds['ID'].iloc[:2].tolist()
        initial_rows = pred_df[pred_df['ID'].isin(initial_ids)]

        # Verify these compounds have NaN in prediction_cycle_0 column
        assert initial_rows['prediction_cycle_0'].isna().all()

        # Verify sparse storage pattern across cycles
        assert pred_df['prediction_cycle_0'].isna().sum() >= 2
        assert pred_df['prediction_cycle_1'].isna().sum() >= 2

        # Verify compounds selected in cycle 0 lack later predictions
        cycle0_ids = sel_df[sel_df['selected_cycle'] == 0]['ID']
        assert pred_df[pred_df['ID'].isin(cycle0_ids)]['prediction_cycle_1'].isna().all()

    def test_cycle_metrics_with_master_df(self, small_real_compounds, tmp_path, mock_oracle_low_noise, mock_learner):
        """Test cycle metrics align with labeled/unlabeled counts."""
        compounds = small_real_compounds.copy()

        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(20)],
                'SMILES': ['CCO'] * 20
            })

        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))

        oracle = mock_oracle_low_noise
        learner = mock_learner

        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=3,
            batch_fraction=0.1,
            n_initial=5,
            output_dir=str(tmp_path),
            random_state=42
        )

        cycle_metrics = results['cycle_metrics']
        labeled_data = results['labeled_data']
        unlabeled_data = results['unlabeled_data']

        # Validate cumulative_labeled increases each cycle
        prev_labeled = 0
        for metrics in cycle_metrics:
            assert metrics['cumulative_labeled'] > prev_labeled
            prev_labeled = metrics['cumulative_labeled']

        # Verify final counts match
        assert len(labeled_data) == cycle_metrics[-1]['cumulative_labeled']
        assert len(unlabeled_data) == cycle_metrics[-1]['remaining_unlabeled']
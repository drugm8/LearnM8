"""
Tests for cycle execution functionality in learnm8.py.

Tests execute_cycle function with real molecular data, focusing on integration and data flow.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from unittest.mock import Mock

from learnm8.core.cycle import execute_cycle
from learnm8.core.config import CycleConfig


def create_test_master_df(compounds, initial_labeled_count=3):
    """Helper to create master DataFrame for testing."""
    from tests.fixtures.master_dataframe import create_initialized_master_df as initialize_master_dataframe

    initial_compounds = compounds.slice(0, initial_labeled_count)
    initial_ids = initial_compounds['ID'].to_list()
    initial_values = pd.Series(
        np.random.uniform(0.1, 0.9, initial_labeled_count),
        index=initial_ids
    )

    return initialize_master_dataframe(
        valid_compounds=compounds,
        target_col='Activity'
    ,
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values
    )


@pytest.mark.integration
class TestCycleExecution:
    """Test cycle execution functions."""

    def test_execute_single_cycle_basic(self, small_real_compounds, tmp_path, mock_oracle, mock_learner_with_uncertainty):
        """Test basic single cycle execution with master DataFrame."""
        compounds = small_real_compounds.clone()

        if len(compounds) == 0:
            compounds = pl.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(10)],
                'SMILES': ['CCO'] * 10
            })

        # Create master DataFrame
        master_df = create_test_master_df(compounds, initial_labeled_count=3)

        oracle = mock_oracle
        learner = mock_learner_with_uncertainty

        # Create CycleConfig
        config = CycleConfig(strategy='greedy', batch_fraction=0.2)

        # Execute single cycle with master DataFrame
        updated_master_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='run',
            output_dir=tmp_path,
        )

        # Validate master DataFrame updated
        assert len(updated_master_df) == len(compounds)
        # Master DF stays narrow; predictions are in the cycle parquet.
        assert 'prediction_cycle_0' not in updated_master_df.columns
        assert metrics.get('parquet_path') is not None
        assert (tmp_path / 'prediction_cycle_0.parquet').exists()
        assert 'status' in updated_master_df.columns

        # Validate status changes
        labeled_count = (updated_master_df['status'] == 'labeled').sum()
        assert labeled_count > 3  # Should have added compounds

        # Validate metrics
        assert metrics['cycle'] == 0
        assert metrics['strategy'] == 'greedy'
        assert metrics['selected_count'] > 0
    
    def test_execute_run_mode_cycle(self, small_real_compounds, tmp_path, mock_oracle, mock_learner):
        """Test run mode cycle execution with master DataFrame."""
        compounds = small_real_compounds.clone()

        if len(compounds) == 0:
            compounds = pl.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(8)],
                'SMILES': ['CCO'] * 8
            })

        # Create master DataFrame
        master_df = create_test_master_df(compounds, initial_labeled_count=2)

        oracle = mock_oracle
        learner = mock_learner

        config = CycleConfig(strategy='random', batch_fraction=0.25)
        updated_master_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=1,
            config=config,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='run',
            output_dir=tmp_path,
        )

        # Master DF stays narrow.
        assert 'prediction_cycle_1' not in updated_master_df.columns
        assert (tmp_path / 'prediction_cycle_1.parquet').exists()
        assert (updated_master_df['status'] == 'labeled').sum() > 2

        # Validate metrics
        assert metrics['strategy'] == 'random'
        assert metrics['cycle'] == 1
        # Batch size should be reasonable (around 25% of original pool size)
        assert metrics['selected_count'] > 0
        assert metrics['selected_count'] <= len(compounds) * 0.3  # Allow some flexibility
    
    def test_execute_benchmark_mode_cycle(self, small_real_compounds, tmp_path, mock_oracle, mock_learner_with_uncertainty):
        """Test benchmark mode cycle execution with master DataFrame."""
        compounds = small_real_compounds.clone()

        if len(compounds) == 0:
            compounds = pl.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(6)],
                'SMILES': ['CCC'] * 6
            })

        # Add ground truth
        compounds = compounds.with_columns(
            pl.Series('Activity', np.random.uniform(0, 1, len(compounds)))
        )

        # Create master DataFrame
        master_df = create_test_master_df(compounds, initial_labeled_count=2)

        oracle = mock_oracle
        learner = mock_learner_with_uncertainty

        config = CycleConfig(strategy='greedy', batch_fraction=0.33)
        updated_master_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='benchmark',
            original_pool=compounds,
            output_dir=tmp_path,
        )

        # Predictions are persisted to parquet, not the master DF.
        assert 'prediction_cycle_0' not in updated_master_df.columns
        cycle_predictions = pl.read_parquet(metrics['parquet_path'])
        assert cycle_predictions.height > 0

        assert metrics['strategy'] == 'greedy'
    
    def test_empty_unlabeled_pool_handling(self, tmp_path, mock_oracle, mock_learner):
        """Test handling of empty unlabeled pool with master DataFrame."""
        compounds = pl.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO']
        })

        # Create master DataFrame with all compounds labeled
        master_df = create_test_master_df(compounds, initial_labeled_count=1)

        oracle = mock_oracle
        learner = mock_learner

        config = CycleConfig(strategy='greedy', batch_fraction=0.5)
        updated_master_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=1,
            score_direction='higher',
            mode='run'
        )

        # Validate returned master_df unchanged
        assert len(updated_master_df) == len(master_df)

        # Check metrics show 0 selected_count
        assert metrics['selected_count'] == 0
        assert metrics['remaining_unlabeled'] == 0
    
    def test_batch_size_calculation(self, tmp_path, mock_oracle, mock_learner):
        """Test correct batch size calculation from batch_fraction with master DataFrame."""
        compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(20)],
            'SMILES': ['CCO'] * 20
        })

        # Create master DataFrame
        master_df = create_test_master_df(compounds, initial_labeled_count=5)

        oracle = mock_oracle
        learner = mock_learner

        # Test different batch fractions
        test_cases = [
            (0.1, 2),   # 10% of 20 = 2
            (0.25, 5),  # 25% of 20 = 5
            (0.05, 1),  # 5% of 20 = 1 (minimum 1)
        ]

        for batch_fraction, expected_selected in test_cases:
            config = CycleConfig(strategy='random', batch_fraction=batch_fraction)
            updated_master_df, metrics = execute_cycle(
                compounds_df=master_df.clone(),
                cycle=0,
                config=config,
                learner=learner,
                oracle=oracle,
                target_col='Activity',
                featurizer='morgan',
                cache_dir=tmp_path,
                original_pool_size=len(compounds),
                score_direction='higher',
                mode='run'
            )

            assert metrics['selected_count'] == expected_selected
    
    def test_score_direction_lower(self, tmp_path, mock_oracle, mock_learner):
        """Test cycle execution with 'lower' score direction."""
        from tests.fixtures.master_dataframe import create_initialized_master_df as initialize_master_dataframe

        compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(8)],
            'SMILES': ['CCN'] * 8
        })

        initial_ids = compounds['ID'].head(2).to_list()
        initial_values = pd.Series([0.9, 0.1], index=initial_ids)
        master_df = initialize_master_dataframe(
        valid_compounds=compounds,
        target_col='Activity'
        ,
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values
    )

        oracle = mock_oracle
        learner = mock_learner

        config = CycleConfig(strategy='greedy', batch_fraction=0.25)
        updated_master_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='lower',
            mode='run'
        )

        assert metrics['selected_count'] == 2
        labeled_count = (updated_master_df['status'] == 'labeled').sum()
        assert labeled_count == 4
    
    def test_csv_export_mode(self, tmp_path, mock_oracle, mock_learner_with_uncertainty):
        """Test cycle execution with CSV export enabled."""
        from tests.fixtures.master_dataframe import create_initialized_master_df as initialize_master_dataframe

        compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(6)],
            'SMILES': ['CCO'] * 6
        })

        initial_ids = compounds['ID'].head(2).to_list()
        initial_values = pd.Series([0.3, 0.7], index=initial_ids)
        master_df = initialize_master_dataframe(
        valid_compounds=compounds,
        target_col='Activity'
        ,
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values
    )

        oracle = mock_oracle
        learner = mock_learner_with_uncertainty

        config = CycleConfig(strategy='random', batch_fraction=0.5)
        updated_master_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='run',
            output_dir=tmp_path,
        )

        assert 'prediction_cycle_0' not in updated_master_df.columns
        assert metrics.get('parquet_path') is not None
        assert metrics['selected_count'] > 0
        assert metrics['strategy'] == 'random'

    def test_learner_training_failure(self, tmp_path, mock_oracle, mock_learner):
        """Test handling of cycle with no labeled compounds raises error."""
        from tests.fixtures.master_dataframe import create_initialized_master_df as initialize_master_dataframe

        compounds = pl.DataFrame({
            'ID': ['COMP_001', 'COMP_002'],
            'SMILES': ['CCO', 'CCC']
        })

        master_df = initialize_master_dataframe(
        valid_compounds=compounds,
        target_col='Activity'
        ,
        initial_labeled_ids=[],
        initial_measurements=pd.Series(dtype='float64')
    )

        oracle = mock_oracle
        learner = mock_learner

        config = CycleConfig(strategy='greedy', batch_fraction=0.5)

        # Cycle 0 should raise error if no labeled compounds (this validates initialization check)
        with pytest.raises(RuntimeError, match="No labeled compounds available"):
            execute_cycle(
                compounds_df=master_df,
                cycle=0,
                config=config,
                learner=learner,
                oracle=oracle,
                target_col='Activity',
                featurizer='morgan',
                cache_dir=tmp_path,
                original_pool_size=len(compounds),
                score_direction='higher',
                mode='run'
            )
    
    def test_prediction_statistics_calculation(self, tmp_path, mock_oracle, mock_learner_with_uncertainty):
        """Test calculation of prediction statistics in cycle metrics."""
        from tests.fixtures.master_dataframe import create_initialized_master_df as initialize_master_dataframe

        compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(10)],
            'SMILES': ['CCO'] * 10
        })

        initial_ids = compounds['ID'].head(3).to_list()
        initial_values = pd.Series([0.1, 0.5, 0.9], index=initial_ids)
        master_df = initialize_master_dataframe(
        valid_compounds=compounds,
        target_col='Activity'
        ,
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values
    )

        oracle = mock_oracle
        learner = mock_learner_with_uncertainty

        config = CycleConfig(strategy='random', batch_fraction=0.2)
        updated_master_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='run'
        )

        assert 'prediction_mean' in metrics
        assert 'prediction_std' in metrics
        assert 'uncertainty_mean' in metrics
        assert 'uncertainty_std' in metrics

        assert 0 <= metrics['prediction_mean'] <= 1
        assert metrics['prediction_std'] >= 0

    def test_feature_extraction_time_metric(self, tmp_path, mock_oracle, mock_learner_with_uncertainty):
        """`feature_extraction_time` is reported separately from training/prediction time.

        Used by the publication compute-bottleneck breakdown (Fig. 5) to isolate
        HDF5 cache speedup. Must be mutually exclusive with training_time and
        prediction_time so stacked bars do not exceed total_time.
        """
        from tests.fixtures.master_dataframe import create_initialized_master_df as initialize_master_dataframe

        compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(10)],
            'SMILES': ['CCO'] * 10,
        })
        initial_ids = compounds['ID'].head(3).to_list()
        initial_values = pd.Series([0.1, 0.5, 0.9], index=initial_ids)
        master_df = initialize_master_dataframe(
            valid_compounds=compounds,
            target_col='Activity',
            initial_labeled_ids=initial_ids,
            initial_measurements=initial_values,
        )

        config = CycleConfig(strategy='random', batch_fraction=0.2)
        _updated_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='run',
        )

        assert 'feature_extraction_time' in metrics
        assert isinstance(metrics['feature_extraction_time'], float)
        assert metrics['feature_extraction_time'] >= 0.0
        # When a real featurizer ('morgan') runs, some non-zero time is expected.
        assert metrics['feature_extraction_time'] > 0.0
        # Mutual exclusivity: feature time must not double-count training/prediction.
        component_sum = (
            metrics['training_time']
            + metrics['prediction_time']
            + metrics['acquisition_time']
            + metrics['feature_extraction_time']
        )
        # Allow small slack for un-instrumented work (oracle, evaluation, bookkeeping)
        # but components must not exceed total_time.
        assert component_sum <= metrics['total_time'] + 1e-3

    def test_multiple_strategies(self, tmp_path, mock_oracle, mock_learner_with_uncertainty):
        """Test cycle execution with different acquisition strategies."""
        from tests.fixtures.master_dataframe import create_initialized_master_df as initialize_master_dataframe

        compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(12)],
            'SMILES': ['CCO'] * 12
        })

        initial_ids = compounds['ID'].head(3).to_list()
        initial_values = pd.Series([0.2, 0.5, 0.8], index=initial_ids)

        oracle = mock_oracle
        learner = mock_learner_with_uncertainty

        strategies_to_test = ['random', 'greedy']

        for strategy in strategies_to_test:
            master_df = initialize_master_dataframe(
        valid_compounds=compounds,
        target_col='Activity'
            ,
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values
    )

            config = CycleConfig(strategy=strategy, batch_fraction=0.25)
            updated_master_df, metrics = execute_cycle(
                compounds_df=master_df,
                cycle=0,
                config=config,
                learner=learner,
                oracle=oracle,
                target_col='Activity',
                featurizer='morgan',
                cache_dir=tmp_path,
                original_pool_size=len(compounds),
                score_direction='higher',
                mode='run'
            )

            assert metrics['strategy'] == strategy
            assert metrics['selected_count'] == 3
            labeled_count = (updated_master_df['status'] == 'labeled').sum()
            assert labeled_count == 6


@pytest.mark.integration
class TestMasterDataFrameCycleIntegration:
    """Test master DataFrame integration in cycle execution."""

    def test_master_df_prediction_parquets(self, tmp_path, mock_oracle, mock_learner_with_uncertainty):
        """Test that per-cycle prediction parquets are written to output_dir.

        Replaces the old test that asserted prediction_cycle_N columns on the
        master DF — those columns no longer exist.
        """
        compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(10)],
            'SMILES': ['CCO'] * 10
        })

        master_df = create_test_master_df(compounds, initial_labeled_count=2)

        oracle = mock_oracle
        learner = mock_learner_with_uncertainty

        config_0 = CycleConfig(strategy='random', batch_fraction=0.2)
        master_df_after_0, metrics_0 = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config_0,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='run',
            output_dir=tmp_path,
        )

        config_1 = CycleConfig(strategy='greedy', batch_fraction=0.2)
        master_df_after_1, metrics_1 = execute_cycle(
            compounds_df=master_df_after_0,
            cycle=1,
            config=config_1,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='run',
            output_dir=tmp_path,
        )

        # Master DF stays narrow — no prediction_cycle_* columns at any time.
        assert 'prediction_cycle_0' not in master_df_after_1.columns
        assert 'prediction_cycle_1' not in master_df_after_1.columns

        # Each cycle wrote a parquet file; metrics carries the path.
        assert metrics_0.get('parquet_path') is not None
        assert metrics_1.get('parquet_path') is not None
        assert (tmp_path / 'prediction_cycle_0.parquet').exists()
        assert (tmp_path / 'prediction_cycle_1.parquet').exists()

    def test_master_df_status_updates(self, tmp_path, mock_oracle, mock_learner):
        """Test status updates in master DataFrame."""
        compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(10)],
            'SMILES': ['CCO'] * 10
        })

        # Create master DataFrame with 2 labeled
        master_df = create_test_master_df(compounds, initial_labeled_count=2)

        oracle = mock_oracle
        learner = mock_learner

        # Execute cycle that selects 3 compounds
        config = CycleConfig(strategy='random', batch_fraction=0.3)
        updated_master_df, _ = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='run'
        )

        # Verify status changed from 'unlabeled' to 'labeled' for selected compounds
        labeled_count = (updated_master_df['status'] == 'labeled').sum()
        assert labeled_count > 2

        # Check labeled_cycle is set correctly
        newly_labeled = updated_master_df.filter(pl.col('labeled_cycle') == 0)
        assert len(newly_labeled) > 0

        # Validate selected_cycle is populated
        assert (newly_labeled['selected_cycle'] == 0).all()

    def test_master_df_immutability(self, tmp_path, mock_oracle, mock_learner):
        """Test that cycle execution doesn't mutate input master_df."""
        compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(8)],
            'SMILES': ['CCO'] * 8
        })

        # Create master DataFrame
        master_df_original = create_test_master_df(compounds, initial_labeled_count=2)

        # Store copy of original
        master_df_copy = master_df_original.clone()

        oracle = mock_oracle
        learner = mock_learner

        # Execute cycle
        config = CycleConfig(strategy='random', batch_fraction=0.25)
        updated_master_df, _ = execute_cycle(
            compounds_df=master_df_original,
            cycle=0,
            config=config,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='run'
        )

        # Verify original master_df unchanged (functional update)
        assert master_df_original.equals(master_df_copy)

        # Confirm returned master_df is different object
        assert updated_master_df is not master_df_original

    def test_benchmark_mode_stores_unlabeled_predictions(self, tmp_path, mock_oracle, mock_learner_with_uncertainty):
        """Test benchmark mode stores unlabeled-only predictions in master_df.

        Both run and benchmark modes now predict on unlabeled compounds only for efficiency.
        The difference is that benchmark mode calculates additional discovery/ranking metrics.
        """
        from tests.fixtures.master_dataframe import create_initialized_master_df as initialize_master_dataframe

        compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(12)],
            'SMILES': ['CCO'] * 12
        })

        # Add ground truth for benchmark mode
        compounds = compounds.with_columns(
            pl.Series('Activity', np.random.uniform(0, 1, len(compounds)))
        )

        # Create master DataFrame with 3 labeled
        initial_ids = compounds['ID'].head(3).to_list()
        initial_values = pd.Series([0.2, 0.5, 0.8], index=initial_ids)
        master_df = initialize_master_dataframe(
        valid_compounds=compounds,
        target_col='Activity'
        ,
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values
    )

        oracle = mock_oracle
        learner = mock_learner_with_uncertainty

        # Execute benchmark mode cycle
        config = CycleConfig(strategy='greedy', batch_fraction=0.25)
        updated_master_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='benchmark',
            original_pool=compounds,
            output_dir=tmp_path,
        )

        # Predictions are stored in the cycle parquet, not on the master DF.
        cycle_predictions = pl.read_parquet(tmp_path / 'prediction_cycle_0.parquet')
        pred_count = cycle_predictions.height
        unlabeled_count = (master_df['status'] == 'unlabeled').sum()

        # Benchmark mode now predicts on unlabeled only (same as run mode)
        assert pred_count == unlabeled_count, \
            f"Expected predictions for unlabeled only ({unlabeled_count}), got {pred_count}"

    def test_evaluation_uses_correct_prediction_pool_run_mode(self, tmp_path, mock_oracle, mock_learner_with_uncertainty):
        """Test evaluation uses unlabeled predictions in run mode.

        Both run and benchmark modes now use identical prediction logic (unlabeled only).
        Verify by checking that predictions in master_df are only populated for unlabeled compounds.
        """
        from tests.fixtures.master_dataframe import create_initialized_master_df as initialize_master_dataframe

        compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(10)],
            'SMILES': ['CCO'] * 10
        })

        compounds = compounds.with_columns(
            pl.Series('Activity', np.random.uniform(0, 1, len(compounds)))
        )

        # Create master DataFrame with 3 labeled
        initial_ids = compounds['ID'].head(3).to_list()
        initial_values = pd.Series([0.3, 0.6, 0.9], index=initial_ids)
        master_df = initialize_master_dataframe(
        valid_compounds=compounds,
        target_col='Activity'
        ,
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values
    )

        oracle = mock_oracle
        learner = mock_learner_with_uncertainty

        # Get unlabeled count before
        unlabeled_count = (master_df['status'] == 'unlabeled').sum()

        config = CycleConfig(strategy='greedy', batch_fraction=0.2)
        updated_master_df, _ = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='run',
            output_dir=tmp_path,
        )

        # Verify predictions only exist for unlabeled compounds (run mode behavior)
        cycle_predictions = pl.read_parquet(tmp_path / 'prediction_cycle_0.parquet')
        pred_count = cycle_predictions.height
        assert pred_count == unlabeled_count, \
            f"Run mode should predict only {unlabeled_count} unlabeled compounds, got {pred_count} predictions"

    def test_evaluation_uses_correct_prediction_pool_benchmark_mode(self, tmp_path, mock_oracle, mock_learner_with_uncertainty):
        """Test evaluation uses unlabeled predictions in benchmark mode.

        Both run and benchmark modes now use identical prediction logic (unlabeled only).
        Benchmark mode differs only in calculating additional discovery/ranking metrics.
        """
        from tests.fixtures.master_dataframe import create_initialized_master_df as initialize_master_dataframe

        compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(10)],
            'SMILES': ['CCO'] * 10
        })

        compounds = compounds.with_columns(
            pl.Series('Activity', np.random.uniform(0, 1, len(compounds)))
        )

        # Create master DataFrame with 3 labeled
        initial_ids = compounds['ID'].head(3).to_list()
        initial_values = pd.Series([0.3, 0.6, 0.9], index=initial_ids)
        master_df = initialize_master_dataframe(
        valid_compounds=compounds,
        target_col='Activity'
        ,
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values
    )

        oracle = mock_oracle
        learner = mock_learner_with_uncertainty

        # Get unlabeled count for comparison
        unlabeled_count = (master_df['status'] == 'unlabeled').sum()

        config = CycleConfig(strategy='greedy', batch_fraction=0.2)
        updated_master_df, _ = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='benchmark',
            original_pool=compounds,
            output_dir=tmp_path,
        )

        # Verify predictions exist only for unlabeled (same as run mode)
        cycle_predictions = pl.read_parquet(tmp_path / 'prediction_cycle_0.parquet')
        pred_count = cycle_predictions.height
        assert pred_count == unlabeled_count, \
            f"Benchmark mode should predict {unlabeled_count} unlabeled (same as run mode), got {pred_count} predictions"

    def test_benchmark_and_run_modes_use_same_prediction_logic(self, tmp_path, mock_oracle, mock_learner_with_uncertainty):
        """Test that benchmark and run modes use identical prediction logic.

        Both modes should predict on the same compound set (unlabeled only).
        The only difference should be in metrics computation.
        """
        from tests.fixtures.master_dataframe import create_initialized_master_df as initialize_master_dataframe

        compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(15)],
            'SMILES': ['CCO'] * 15
        })
        compounds = compounds.with_columns(
            pl.Series('Activity', np.random.uniform(0, 1, len(compounds)))
        )

        # Create master DataFrame with 5 labeled
        initial_ids = compounds['ID'].head(5).to_list()
        initial_values = pd.Series([0.2, 0.4, 0.6, 0.8, 1.0], index=initial_ids)
        master_df_run = initialize_master_dataframe(
        valid_compounds=compounds,
        target_col='Activity'
        ,
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values
    )
        master_df_benchmark = master_df_run.clone()

        oracle = mock_oracle
        learner_run = mock_learner_with_uncertainty
        learner_benchmark = mock_learner_with_uncertainty

        config = CycleConfig(strategy='greedy', batch_fraction=0.2)

        # Each mode writes its parquet to its own subdir to avoid collision.
        run_dir = tmp_path / 'run'
        run_dir.mkdir()
        benchmark_dir = tmp_path / 'benchmark'
        benchmark_dir.mkdir()

        # Execute run mode cycle
        updated_df_run, metrics_run = execute_cycle(
            compounds_df=master_df_run,
            cycle=0,
            config=config,
            learner=learner_run,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='run',
            output_dir=run_dir,
        )

        # Execute benchmark mode cycle
        updated_df_benchmark, metrics_benchmark = execute_cycle(
            compounds_df=master_df_benchmark,
            cycle=0,
            config=config,
            learner=learner_benchmark,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='benchmark',
            original_pool=compounds,
            output_dir=benchmark_dir,
        )

        # Verify both modes predicted on the same number of compounds
        pred_count_run = pl.read_parquet(run_dir / 'prediction_cycle_0.parquet').height
        pred_count_benchmark = pl.read_parquet(benchmark_dir / 'prediction_cycle_0.parquet').height

        assert pred_count_run == pred_count_benchmark, \
            f"Run and benchmark modes should predict on same compounds: run={pred_count_run}, benchmark={pred_count_benchmark}"

        # Verify both equal the unlabeled count
        unlabeled_count = (master_df_run['status'] == 'unlabeled').sum()
        assert pred_count_run == unlabeled_count, \
            f"Both modes should predict on {unlabeled_count} unlabeled compounds, run mode got {pred_count_run}"
        assert pred_count_benchmark == unlabeled_count, \
            f"Both modes should predict on {unlabeled_count} unlabeled compounds, benchmark mode got {pred_count_benchmark}"

    def test_selected_and_labeled_cycle_consistency(self, tmp_path, mock_oracle, mock_learner):
        """Test selected_cycle and labeled_cycle are set correctly and remain unchanged."""
        from tests.fixtures.master_dataframe import create_initialized_master_df as initialize_master_dataframe

        compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(10)],
            'SMILES': ['CCO'] * 10
        })

        compounds = compounds.with_columns(
            pl.Series('Activity', np.random.uniform(0, 1, len(compounds)))
        )

        # Create master DataFrame with 2 labeled initially
        initial_ids = compounds['ID'].head(2).to_list()
        initial_values = pd.Series([0.3, 0.7], index=initial_ids)
        master_df = initialize_master_dataframe(
        valid_compounds=compounds,
        target_col='Activity'
        ,
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values
    )

        oracle = mock_oracle
        learner = mock_learner

        # Execute cycle 0
        config_0 = CycleConfig(strategy='random', batch_fraction=0.2)
        master_df_0, _ = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config_0,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='run'
        )

        # Get compounds selected in cycle 0 (newly labeled, excluding initially labeled)
        newly_labeled_cycle_0 = master_df_0.filter(
            (pl.col('labeled_cycle') == 0) &
            (~pl.col('ID').is_in(initial_ids))
        )

        # Verify selected_cycle and labeled_cycle are both 0 for newly selected compounds
        for row in newly_labeled_cycle_0.to_dicts():
            assert row['selected_cycle'] == 0, \
                f"Compound {row['ID']} should have selected_cycle=0"
            assert row['labeled_cycle'] == 0, \
                f"Compound {row['ID']} should have labeled_cycle=0"

        # Store IDs selected in cycle 0
        cycle_0_ids = newly_labeled_cycle_0['ID'].to_list()
        assert len(cycle_0_ids) > 0, "Should have selected compounds in cycle 0"

        # Execute cycle 1
        config_1 = CycleConfig(strategy='greedy', batch_fraction=0.2)
        master_df_1, _ = execute_cycle(
            compounds_df=master_df_0,
            cycle=1,
            config=config_1,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(compounds),
            score_direction='higher',
            mode='run'
        )

        # Verify compounds from cycle 0 still have their original selected_cycle and labeled_cycle
        for cid in cycle_0_ids:
            row = master_df_1.filter(pl.col('ID') == cid).to_dicts()[0]
            assert row['selected_cycle'] == 0, \
                f"Compound {cid} selected in cycle 0 should retain selected_cycle=0, got {row['selected_cycle']}"
            assert row['labeled_cycle'] == 0, \
                f"Compound {cid} labeled in cycle 0 should retain labeled_cycle=0, got {row['labeled_cycle']}"

        # Verify new compounds selected in cycle 1 have selected_cycle=1
        newly_labeled_cycle_1 = master_df_1.filter(
            pl.col('labeled_cycle') == 1
        )

        for row in newly_labeled_cycle_1.to_dicts():
            assert row['selected_cycle'] == 1, \
                f"Compound {row['ID']} selected in cycle 1 should have selected_cycle=1"
            assert row['labeled_cycle'] == 1, \
                f"Compound {row['ID']} labeled in cycle 1 should have labeled_cycle=1"
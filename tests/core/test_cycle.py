import pytest
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path

from learnm8.core.cycle import execute_cycle, _calculate_cycle_metrics, _apply_pruning, _select_compounds
from learnm8.core.config import CycleConfig
from tests.fixtures.master_dataframe import create_initialized_master_df as initialize_master_dataframe


@pytest.mark.integration
class TestExecuteCycle:

    def test_cycle_1_with_initialization(self, sample_compounds, mock_learner, mock_oracle, tmp_path):
        """Test that cycle 1 (first AL cycle) trains and predicts after cycle 0 (initialization)."""
        from learnm8.core.initialization import initialize_master_dataframe_empty, select_initial_batch

        master_df = initialize_master_dataframe_empty(sample_compounds, 'Activity')

        assert (master_df['status'] == 'unlabeled').all()

        master_df, cycle_0_metrics = select_initial_batch(
            compounds_df=master_df,
            oracle=mock_oracle,
            target_col='Activity',
            strategy='random',
            batch_fraction=0.2,
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(sample_compounds),
            random_state=42
        )

        init_labeled = master_df.filter(pl.col('labeled_cycle') == 0)
        assert len(init_labeled) == 1
        assert (init_labeled['status'] == 'labeled').all()
        assert cycle_0_metrics['cycle'] == 0

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.2)

        updated_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=1,
            config=config,
            learner=mock_learner,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(sample_compounds),
            mode='run'
        )

        cycle_1_labeled = updated_df.filter(pl.col('labeled_cycle') == 1)
        assert len(cycle_1_labeled) == 1

        init_labeled = updated_df.filter(pl.col('labeled_cycle') == 0)
        assert len(init_labeled) == 1

        total_labeled = (updated_df['status'] == 'labeled').sum()
        assert total_labeled == 2

        assert 'prediction_cycle_1' in updated_df.columns
        assert metrics['cycle'] == 1
        assert metrics['strategy'] == 'greedy'

    def test_cycle_execution_requires_labeled_data(self, sample_compounds, mock_learner, mock_oracle, tmp_path):
        """Test that AL cycle execution fails if no labeled data (missing initialization)."""
        from learnm8.core.initialization import initialize_master_dataframe_empty

        master_df = initialize_master_dataframe_empty(sample_compounds, 'Activity')

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.1)

        with pytest.raises(RuntimeError, match="No labeled compounds available"):
            execute_cycle(
                compounds_df=master_df,
                cycle=1,
                config=config,
                learner=mock_learner,
                oracle=mock_oracle,
                target_col='Activity',
                featurizer='morgan',
                cache_dir=tmp_path,
                original_pool_size=len(sample_compounds),
                mode='run'
            )

    def test_run_mode_basic_execution(self, sample_compounds, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        initial_ids = sample_compounds['ID'].head(2).to_list()
        initial_values = dict(zip(initial_ids, [0.3, 0.6]))

        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            target_col='Activity',
            initial_labeled_ids=initial_ids,
            initial_measurements=initial_values
        )

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        updated_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=1,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(sample_compounds),
            mode='run'
        )

        assert len(updated_df) == len(master_df)
        assert 'prediction_cycle_1' in updated_df.columns
        assert metrics['cycle'] == 1
        assert metrics['strategy'] == 'greedy'
        assert metrics['selected_count'] > 0

    def test_benchmark_mode_requires_original_pool(self, sample_master_df, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        with pytest.raises(ValueError, match="original_pool is required for benchmark mode"):
            execute_cycle(
                compounds_df=sample_master_df,
                cycle=0,
                config=config,
                learner=mock_learner_with_uncertainty,
                oracle=mock_oracle,
                target_col='Activity',
                featurizer='morgan',
                cache_dir=tmp_path,
                original_pool_size=100,
                mode='benchmark',
                original_pool=None
            )

    def test_benchmark_mode_predicts_unlabeled_only(self, sample_compounds, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        initial_ids = sample_compounds['ID'].head(2).to_list()
        initial_values = dict(zip(initial_ids, [0.3, 0.6]))

        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            target_col='Activity',
            initial_labeled_ids=initial_ids,
            initial_measurements=initial_values
        )

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        updated_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(sample_compounds),
            mode='benchmark',
            original_pool=sample_compounds
        )

        pred_col = 'prediction_cycle_0'
        pred_count = (~updated_df[pred_col].is_null()).sum()
        unlabeled_count = (master_df['status'] == 'unlabeled').sum()

        assert pred_count == unlabeled_count

    def test_pruning_integration(self, sample_master_df, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        config = CycleConfig(
            'greedy',
            n_cycles=1,
            batch_fraction=0.05,
            pruning_strategy='score',
            pruning_params={'pruning_fraction': 0.3}
        )

        updated_df, metrics = execute_cycle(
            compounds_df=sample_master_df,
            cycle=0,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=100,
            mode='run'
        )

        assert metrics['pruned_count'] >= 0

    def test_score_direction_higher(self, sample_master_df, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        updated_df, metrics = execute_cycle(
            compounds_df=sample_master_df,
            cycle=0,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=100,
            score_direction='higher',
            mode='run'
        )

        assert metrics['best_so_far'] is not None

    def test_score_direction_lower(self, sample_master_df, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        updated_df, metrics = execute_cycle(
            compounds_df=sample_master_df,
            cycle=0,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=100,
            score_direction='lower',
            mode='run'
        )

        assert metrics['best_so_far'] is not None

    def test_invalid_score_direction_raises_error(self, sample_master_df, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        with pytest.raises(ValueError, match="score_direction must be 'higher' or 'lower'"):
            execute_cycle(
                compounds_df=sample_master_df,
                cycle=0,
                config=config,
                learner=mock_learner_with_uncertainty,
                oracle=mock_oracle,
                target_col='Activity',
                featurizer='morgan',
                cache_dir=tmp_path,
                original_pool_size=100,
                score_direction='invalid',
                mode='run'
            )

    def test_batch_size_computation_with_fraction(self, sample_master_df, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.4)

        updated_df, metrics = execute_cycle(
            compounds_df=sample_master_df,
            cycle=0,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=5,
            mode='run'
        )

        expected_batch_size = int(5 * 0.4)
        assert metrics['selected_count'] == expected_batch_size

    def test_training_failure_raises_error(self, sample_master_df, mock_oracle, tmp_path):
        from learnm8.core.interfaces import Learner

        class FailingLearner(Learner):
            def train(self, features, targets):
                raise RuntimeError("Training failed")

            def predict(self, features):
                return np.array([]), None

            def supports_uncertainty(self):
                return False

            def get_name(self):
                return "FailingLearner"

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        with pytest.raises(RuntimeError, match="Training failed in cycle 0"):
            execute_cycle(
                compounds_df=sample_master_df,
                cycle=0,
                config=config,
                learner=FailingLearner(),
                oracle=mock_oracle,
                target_col='Activity',
                featurizer='morgan',
                cache_dir=tmp_path,
                original_pool_size=100,
                mode='run'
            )

    def test_no_unlabeled_compounds_returns_unchanged(self, sample_compounds, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        all_labeled = sample_compounds['ID'].to_list()
        all_values = dict(zip(all_labeled, np.random.rand(len(all_labeled))))

        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            target_col='Activity',
            initial_labeled_ids=all_labeled,
            initial_measurements=all_values
        )

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        updated_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(sample_compounds),
            mode='run'
        )

        assert metrics['selected_count'] == 0
        assert metrics['remaining_unlabeled'] == 0


@pytest.mark.integration
class TestCalculateCycleMetrics:

    def test_basic_metrics_structure(self, sample_master_df):
        predictions = np.random.rand(50)
        uncertainties = np.random.rand(50)
        selected_ids = ['COMP_0003', 'COMP_0004']
        pruned_ids = []

        metrics = _calculate_cycle_metrics(
            sample_master_df,
            cycle=0,
            strategy='greedy',
            predictions=predictions,
            uncertainties=uncertainties,
            selected_ids=selected_ids,
            pruned_ids=pruned_ids,
            target_col='Activity',
            score_direction='higher'
        )

        assert 'cycle' in metrics
        assert 'strategy' in metrics
        assert 'batch_size' in metrics
        assert 'selected_count' in metrics
        assert 'remaining_unlabeled' in metrics
        assert 'cumulative_labeled' in metrics
        assert 'cumulative_pruned' in metrics

    def test_metrics_key_name_remaining_unlabeled(self, sample_master_df):
        predictions = np.random.rand(50)
        selected_ids = ['COMP_0003']

        metrics = _calculate_cycle_metrics(
            sample_master_df,
            cycle=0,
            strategy='greedy',
            predictions=predictions,
            uncertainties=None,
            selected_ids=selected_ids,
            pruned_ids=[],
            target_col='Activity',
            score_direction='higher'
        )

        assert 'remaining_unlabeled' in metrics
        assert 'remaining_pool' not in metrics

    def test_prediction_statistics(self, sample_master_df):
        predictions = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        selected_ids = ['COMP_0003']

        metrics = _calculate_cycle_metrics(
            sample_master_df,
            cycle=0,
            strategy='greedy',
            predictions=predictions,
            uncertainties=None,
            selected_ids=selected_ids,
            pruned_ids=[],
            target_col='Activity',
            score_direction='higher'
        )

        assert metrics['prediction_mean'] == pytest.approx(0.5)
        assert metrics['prediction_min'] == pytest.approx(0.1)
        assert metrics['prediction_max'] == pytest.approx(0.9)

    def test_uncertainty_statistics(self, sample_master_df):
        predictions = np.random.rand(50)
        uncertainties = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
        selected_ids = ['COMP_0003']

        metrics = _calculate_cycle_metrics(
            sample_master_df,
            cycle=0,
            strategy='greedy',
            predictions=predictions,
            uncertainties=uncertainties,
            selected_ids=selected_ids,
            pruned_ids=[],
            target_col='Activity',
            score_direction='higher'
        )

        assert metrics['has_uncertainty'] is True
        assert 'uncertainty_mean' in metrics
        assert 'uncertainty_std' in metrics


@pytest.mark.integration
class TestApplyPruning:

    def test_pruning_reduces_pool(self, sample_compounds):
        pool = sample_compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', np.random.rand(len(pool)))
        )
        predictions = pool['prediction'].to_numpy()
        uncertainties = np.random.rand(len(pool))

        pruned_pool, stats = _apply_pruning(
            pool,
            predictions,
            uncertainties,
            strategy='score',
            params={'pruning_fraction': 0.3},
            score_direction='higher'
        )

        assert len(pruned_pool) <= len(pool)
        assert stats['pruned_count'] >= 0

    def test_pruning_returns_stats(self, sample_compounds):
        pool = sample_compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', np.random.rand(len(pool)))
        )
        predictions = pool['prediction'].to_numpy()

        pruned_pool, stats = _apply_pruning(
            pool,
            predictions,
            None,
            strategy='score',
            params={'pruning_fraction': 0.2},
            score_direction='higher'
        )

        assert 'pruned_count' in stats
        assert 'pruned_ids' in stats
        assert 'original_count' in stats
        assert 'remaining_count' in stats
        assert 'pruning_fraction' in stats

    def test_pruning_failure_raises_error(self, sample_compounds):
        pool = sample_compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', np.random.rand(len(pool)))
        )
        predictions = pool['prediction'].to_numpy()

        with pytest.raises(RuntimeError, match="Pruning failed with strategy"):
            _apply_pruning(
                pool,
                predictions,
                None,
                strategy='invalid_strategy',
                params={},
                score_direction='higher'
            )


@pytest.mark.integration
class TestSelectCompounds:

    def test_greedy_selection(self, sample_compounds):
        pool = sample_compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', np.random.rand(len(pool)))
        )
        batch_size = 2

        selected_df = _select_compounds(
            pool,
            strategy='greedy',
            batch_size=batch_size,
            score_direction='higher',
            acquisition_params={}
        )

        assert len(selected_df) == batch_size

    def test_random_strategy_selects_requested_batch_size(self, sample_compounds):
        pool = sample_compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', np.random.rand(len(pool)))
        )
        batch_size = 2

        selected_df = _select_compounds(
            pool,
            strategy='random',
            batch_size=batch_size,
            score_direction='higher',
            acquisition_params={}
        )

        assert len(selected_df) == batch_size

    def test_unknown_strategy_raises_error(self, sample_compounds):
        pool = sample_compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', np.random.rand(len(pool)))
        )

        with pytest.raises(ValueError, match="Unknown acquisition strategy"):
            _select_compounds(
                pool,
                strategy='invalid_strategy',
                batch_size=10,
                score_direction='higher',
                acquisition_params={}
            )

    def test_missing_uncertainty_for_ucb_raises_error(self, sample_compounds):
        pool = sample_compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', np.random.rand(len(pool)))
        )

        with pytest.raises(ValueError, match="requires uncertainty estimates"):
            _select_compounds(
                pool,
                strategy='ucb',
                batch_size=10,
                score_direction='higher',
                acquisition_params={}
            )


@pytest.mark.integration
class TestEdgeCaseHandling:

    def test_nan_predictions_handling(self, sample_master_df, mock_oracle, tmp_path):
        from learnm8.core.interfaces import Learner

        class NaNPredictingLearner(Learner):
            def train(self, features, targets):
                pass

            def predict(self, features):
                predictions = np.full(features.shape[0], np.nan)
                return predictions, None

            def supports_uncertainty(self):
                return False

            def get_name(self):
                return "NaNLearner"

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        with pytest.raises(RuntimeError, match="Predictions contain NaN values"):
            execute_cycle(
                compounds_df=sample_master_df,
                cycle=0,
                config=config,
                learner=NaNPredictingLearner(),
                oracle=mock_oracle,
                target_col='Activity',
                featurizer='morgan',
                cache_dir=tmp_path,
                original_pool_size=100,
                mode='run'
            )

    def test_infinite_predictions_handling(self, sample_master_df, mock_oracle, tmp_path):
        from learnm8.core.interfaces import Learner

        class InfPredictingLearner(Learner):
            def train(self, features, targets):
                pass

            def predict(self, features):
                predictions = np.full(features.shape[0], np.inf)
                return predictions, None

            def supports_uncertainty(self):
                return False

            def get_name(self):
                return "InfLearner"

        config = CycleConfig('random', n_cycles=1, batch_fraction=0.05)

        updated_df, metrics = execute_cycle(
            compounds_df=sample_master_df,
            cycle=0,
            config=config,
            learner=InfPredictingLearner(),
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=100,
            mode='run'
        )

        assert 'prediction_cycle_0' in updated_df.columns
        pred_col = 'prediction_cycle_0'
        pred_values = updated_df[pred_col].drop_nulls().to_numpy()
        assert np.all(np.isinf(pred_values))
        assert metrics['selected_count'] > 0

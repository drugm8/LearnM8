from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from learnm8.core.config import CycleConfig
from learnm8.core.cycle import (
    _apply_pruning,
    _calculate_cycle_metrics,
    _select_compounds,
    build_selection_order_lookup,
    execute_cycle,
)
from learnm8.core.interfaces import Oracle
from learnm8.exceptions import (
    AcquisitionError,
    LearnerError,
    LearnM8Error,
    OracleError,
    PruningError,
)
from tests.fixtures.master_dataframe import (
    create_initialized_master_df as initialize_master_dataframe,
)


@pytest.mark.integration
class TestExecuteCycle:

    def test_cycle_1_with_initialization(self, sample_compounds, mock_learner, mock_oracle, tmp_path):
        """Test that cycle 1 (first AL cycle) trains and predicts after cycle 0 (initialization)."""
        from learnm8.core.initialization import (
            initialize_master_dataframe_empty,
            select_initial_batch,
        )

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
            oracle_type='run',
            output_dir=tmp_path,
        )

        cycle_1_labeled = updated_df.filter(pl.col('labeled_cycle') == 1)
        assert len(cycle_1_labeled) == 1

        init_labeled = updated_df.filter(pl.col('labeled_cycle') == 0)
        assert len(init_labeled) == 1

        total_labeled = (updated_df['status'] == 'labeled').sum()
        assert total_labeled == 2

        # Master DF stays narrow; predictions live in the cycle parquet.
        assert 'prediction_cycle_1' not in updated_df.columns
        assert metrics['parquet_path'] == tmp_path / 'prediction_cycle_1.parquet'
        assert metrics['parquet_path'].exists()
        assert metrics['cycle'] == 1
        assert metrics['strategy'] == 'greedy'

    def test_cycle_execution_requires_labeled_data(self, sample_compounds, mock_learner, mock_oracle, tmp_path):
        """Test that AL cycle execution fails if no labeled data (missing initialization)."""
        from learnm8.core.initialization import initialize_master_dataframe_empty

        master_df = initialize_master_dataframe_empty(sample_compounds, 'Activity')

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.1)

        with pytest.raises(LearnerError, match="No labeled compounds available"):
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
                oracle_type='run'
            )

    def test_run_mode_basic_execution(self, sample_compounds, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        initial_ids = sample_compounds['ID'].head(2).to_list()
        initial_values = dict(zip(initial_ids, [0.3, 0.6], strict=False))

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
            oracle_type='run',
            output_dir=tmp_path,
        )

        assert len(updated_df) == len(master_df)
        assert 'prediction_cycle_1' not in updated_df.columns
        assert metrics['parquet_path'] == tmp_path / 'prediction_cycle_1.parquet'
        assert metrics['parquet_path'].exists()
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
                oracle_type='benchmark',
                original_pool=None
            )

    def test_benchmark_mode_predicts_unlabeled_only(self, sample_compounds, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        initial_ids = sample_compounds['ID'].head(2).to_list()
        initial_values = dict(zip(initial_ids, [0.3, 0.6], strict=False))

        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            target_col='Activity',
            initial_labeled_ids=initial_ids,
            initial_measurements=initial_values
        )

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        _updated_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(sample_compounds),
            oracle_type='benchmark',
            original_pool=sample_compounds,
            output_dir=tmp_path,
        )

        # Predictions are now in parquet, not on the master DF.
        cycle_predictions = pl.read_parquet(metrics['parquet_path'])
        pred_count = cycle_predictions.height
        unlabeled_count = (master_df['status'] == 'unlabeled').sum()

        assert pred_count == unlabeled_count

    @pytest.mark.slow
    def test_pruning_integration(self, sample_master_df, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        config = CycleConfig(
            'greedy',
            n_cycles=1,
            batch_fraction=0.05,
            pruning_strategy='score',
            pruning_params={'pruning_fraction': 0.3}
        )

        _updated_df, metrics = execute_cycle(
            compounds_df=sample_master_df,
            cycle=0,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=100,
            oracle_type='run'
        )

        assert metrics['pruned_count'] >= 0

    def test_score_direction_higher(self, sample_master_df, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        _updated_df, metrics = execute_cycle(
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
            oracle_type='run'
        )

        assert metrics['best_so_far'] is not None

    def test_score_direction_lower(self, sample_master_df, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        _updated_df, metrics = execute_cycle(
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
            oracle_type='run'
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
                oracle_type='run'
            )

    def test_batch_size_computation_with_fraction(self, sample_master_df, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.4)

        _updated_df, metrics = execute_cycle(
            compounds_df=sample_master_df,
            cycle=0,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=5,
            oracle_type='run'
        )

        expected_batch_size = int(5 * 0.4)
        assert metrics['selected_count'] == expected_batch_size

    def test_batch_size_floor_one_for_tiny_pool(self, sample_compounds, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        pool_of_ten = sample_compounds.clone()[:10]
        master_df = initialize_master_dataframe(
            valid_compounds=pool_of_ten,
            target_col='Activity',
            initial_labeled_ids=[pool_of_ten[0, 'ID']],
            initial_measurements={pool_of_ten[0, 'ID']: 0.5}
        )
        config = CycleConfig('random', n_cycles=1, batch_fraction=0.001)

        _updated_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=10,
            oracle_type='run'
        )
        assert metrics['selected_count'] == 1

    def test_training_failure_raises_error(self, sample_master_df, mock_oracle, tmp_path):
        from learnm8.core.interfaces import Learner

        class FailingLearner(Learner):
            def train(self, features, targets):
                raise RuntimeError("Training failed")

            def predict(self, features, smiles=None, *, compute_uncertainty=True):
                assert isinstance(compute_uncertainty, bool)
                return np.array([]), None

            def supports_uncertainty(self):
                return False

            def get_name(self):
                return "FailingLearner"

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        with pytest.raises(LearnerError, match="Training failed in cycle 0"):
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
                oracle_type='run'
            )

    def test_no_unlabeled_compounds_returns_unchanged(self, sample_compounds, mock_learner_with_uncertainty, mock_oracle, tmp_path):
        all_labeled = sample_compounds['ID'].to_list()
        all_values = dict(zip(all_labeled, np.random.rand(len(all_labeled)), strict=False))

        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            target_col='Activity',
            initial_labeled_ids=all_labeled,
            initial_measurements=all_values
        )

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        _updated_df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(sample_compounds),
            oracle_type='run'
        )

        assert metrics['selected_count'] == 0
        assert metrics['remaining_unlabeled'] == 0

    def test_empty_cycle_schema_drops_uncertainty_column(
        self, sample_compounds, mock_learner_with_uncertainty, mock_oracle, tmp_path
    ):
        """Feature 023 FR-009 / D5: the empty-cycle fallback predictions schema
        must drop the uncertainty column when compute_uncertainty resolves to
        False (greedy acquisition + no force).
        """
        all_labeled = sample_compounds['ID'].to_list()
        all_values = dict(
            zip(all_labeled, np.random.rand(len(all_labeled)), strict=False)
        )
        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            target_col='Activity',
            initial_labeled_ids=all_labeled,
            initial_measurements=all_values,
        )

        # greedy → acq.requires_uncertainty() False; force_uncertainty=False.
        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        _df, metrics = execute_cycle(
            compounds_df=master_df,
            cycle=0,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(sample_compounds),
            oracle_type='run',
            force_uncertainty=False,
        )

        # 'selected_predictions' is the empty schema-bearing DF for the empty-pool
        # fallback path; it must NOT carry an uncertainty column when the cycle
        # opted out of uncertainty.
        empty = metrics['selected_predictions']
        if empty is not None and len(empty) == 0:
            assert 'uncertainty' not in empty.columns, (
                'Empty-cycle predictions schema must drop uncertainty column '
                'when compute_uncertainty=False (FR-009 / D5).'
            )


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

    @staticmethod
    def _make_pool(sample_compounds, with_uncertainty=False):
        """Build a selection_pool DataFrame with prediction (and optionally
        uncertainty) columns from sample_compounds."""
        pool = sample_compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', np.random.rand(len(pool)))
        )
        if with_uncertainty:
            pool = pool.with_columns(
                pl.Series('uncertainty', np.random.rand(len(pool)))
            )
        return pool

    def test_pruning_reduces_pool(self, sample_compounds):
        pool = self._make_pool(sample_compounds, with_uncertainty=True)
        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds, target_col='Activity'
        )

        _compounds_df, pruned_pool, stats = _apply_pruning(
            selection_pool=pool,
            compounds_df=master_df,
            cycle=0,
            target_col='Activity',
            pruning_strategy='score',
            pruning_params={'pruning_fraction': 0.3},
            score_direction='higher',
        )

        assert len(pruned_pool) <= len(pool)
        assert stats['pruned_count'] >= 0

    def test_pruning_returns_stats(self, sample_compounds):
        pool = self._make_pool(sample_compounds)
        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds, target_col='Activity'
        )

        _compounds_df, _pruned_pool, stats = _apply_pruning(
            selection_pool=pool,
            compounds_df=master_df,
            cycle=0,
            target_col='Activity',
            pruning_strategy='score',
            pruning_params={'pruning_fraction': 0.2},
            score_direction='higher',
        )

        assert 'pruned_count' in stats
        assert 'pruned_ids' in stats
        assert 'original_count' in stats
        assert 'remaining_count' in stats
        assert 'pruning_fraction' in stats

    def test_pruning_failure_raises_error(self, sample_compounds):
        pool = self._make_pool(sample_compounds)
        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds, target_col='Activity'
        )

        with pytest.raises(PruningError, match="Pruning failed with strategy"):
            _apply_pruning(
                selection_pool=pool,
                compounds_df=master_df,
                cycle=0,
                target_col='Activity',
                pruning_strategy='invalid_strategy',
                pruning_params={},
                score_direction='higher',
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

    @pytest.mark.slow
    def test_nan_predictions_handling(self, sample_master_df, mock_oracle, tmp_path):
        from learnm8.core.interfaces import Learner

        class NaNPredictingLearner(Learner):
            def train(self, features, targets):
                pass

            def predict(self, features, smiles=None, *, compute_uncertainty=True):
                assert isinstance(compute_uncertainty, bool)
                predictions = np.full(features.shape[0], np.nan)
                return predictions, None

            def supports_uncertainty(self):
                return False

            def get_name(self):
                return "NaNLearner"

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.05)

        with pytest.raises(LearnerError, match="NaN predictions detected"):
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
                oracle_type='run'
            )

    def test_infinite_predictions_handling(self, sample_master_df, mock_oracle, tmp_path):
        from learnm8.core.interfaces import Learner

        class InfPredictingLearner(Learner):
            def train(self, features, targets):
                pass

            def predict(self, features, smiles=None, *, compute_uncertainty=True):
                assert isinstance(compute_uncertainty, bool)
                predictions = np.full(features.shape[0], np.inf)
                return predictions, None

            def supports_uncertainty(self):
                return False

            def get_name(self):
                return "InfLearner"

        config = CycleConfig('random', n_cycles=1, batch_fraction=0.05)

        _updated_df, metrics = execute_cycle(
            compounds_df=sample_master_df,
            cycle=0,
            config=config,
            learner=InfPredictingLearner(),
            oracle=mock_oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=100,
            oracle_type='run',
            output_dir=tmp_path,
        )

        # Predictions are now persisted to parquet rather than the master DF.
        cycle_predictions = pl.read_parquet(metrics['parquet_path'])
        pred_values = cycle_predictions['prediction'].drop_nulls().to_numpy()
        assert np.all(np.isinf(pred_values))
        assert metrics['selected_count'] > 0


@pytest.mark.integration
class TestNarrowExceptPaths:
    """Regression tests for the 4 narrow-except branches preserved by the
    execute_cycle decomposition refactor (REQ-2). Each test forces a specific
    typed exception inside one of the 8 try/except blocks in cycle.py and
    asserts it propagates (or is logged) exactly as pre-refactor code does.
    """

    def test_oracle_error_propagates_as_oracle_error(
        self, sample_compounds, mock_learner, tmp_path
    ):
        """OracleError from oracle.measure() must bubble out unchanged
        (cycle.py:474 — `except OracleError: raise`)."""

        class RaisingOracle(Oracle):
            def measure(self, compounds, properties):
                raise OracleError("forced oracle failure for narrow-except test")

        initial_ids = sample_compounds['ID'].head(2).to_list()
        initial_values = dict(zip(initial_ids, [0.3, 0.6], strict=True))
        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            target_col='Activity',
            initial_labeled_ids=initial_ids,
            initial_measurements=initial_values,
        )

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.1)

        with pytest.raises(OracleError, match="forced oracle failure"):
            execute_cycle(
                compounds_df=master_df,
                cycle=1,
                config=config,
                learner=mock_learner,
                oracle=RaisingOracle(),
                target_col='Activity',
                featurizer='morgan',
                cache_dir=tmp_path,
                original_pool_size=len(sample_compounds),
                oracle_type='run',
                output_dir=tmp_path,
            )

    def test_acquisition_error_propagates_as_acquisition_error(
        self, sample_compounds, mock_learner, mock_oracle, tmp_path
    ):
        """AcquisitionError raised inside acq_func.select() must propagate
        unchanged through `_select_compounds`'s `except AcquisitionError: raise`
        (cycle.py:902)."""

        class RaisingAcqClass:
            def __init__(self, score_direction, **kwargs):
                self.score_direction = score_direction
                self._skip_unique_id_check = False

            def requires_uncertainty(self):
                return False

            def select(self, pool, batch_size):
                raise AcquisitionError("forced acquisition failure for narrow-except test")

        initial_ids = sample_compounds['ID'].head(2).to_list()
        initial_values = dict(zip(initial_ids, [0.3, 0.6], strict=True))
        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            target_col='Activity',
            initial_labeled_ids=initial_ids,
            initial_measurements=initial_values,
        )

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.1)

        with (
            patch('learnm8.acquisition.get_acquisition_function', return_value=RaisingAcqClass),
            pytest.raises(AcquisitionError, match="forced acquisition failure"),
        ):
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
                    oracle_type='run',
                    output_dir=tmp_path,
                )

    def test_pruning_error_propagates_as_pruning_error(
        self, sample_compounds, mock_learner_with_uncertainty, mock_oracle, tmp_path
    ):
        """PruningError raised inside pruner.prune() must propagate unchanged
        through `_apply_pruning`'s `except PruningError: raise`
        (cycle.py:819)."""

        class RaisingPruner:
            def prune(self, pool, predictions, uncertainties):
                raise PruningError("forced pruning failure for narrow-except test")

        def fake_create_pruning_strategy(strategy_name, parameters):
            return RaisingPruner()

        initial_ids = sample_compounds['ID'].head(2).to_list()
        initial_values = dict(zip(initial_ids, [0.3, 0.6], strict=True))
        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            target_col='Activity',
            initial_labeled_ids=initial_ids,
            initial_measurements=initial_values,
        )

        config = CycleConfig(
            'greedy',
            n_cycles=1,
            batch_fraction=0.1,
            pruning_strategy='score',
            pruning_params={'pruning_fraction': 0.3},
        )

        with patch(
            'learnm8.pruning.create_pruning_strategy',
            side_effect=fake_create_pruning_strategy,
        ), pytest.raises(PruningError, match="forced pruning failure"):
            execute_cycle(
                compounds_df=master_df,
                cycle=1,
                config=config,
                learner=mock_learner_with_uncertainty,
                oracle=mock_oracle,
                target_col='Activity',
                featurizer='morgan',
                cache_dir=tmp_path,
                original_pool_size=len(sample_compounds),
                oracle_type='run',
                output_dir=tmp_path,
            )

    def test_evaluation_column_missing_logs_warning_and_continues(
        self, sample_compounds, mock_learner, mock_oracle, tmp_path
    ):
        """ColumnNotFoundError raised inside evaluate_cycle() must be caught
        by the eval-wrap try/except (cycle.py:588), logged as a warning, and
        execution must continue and return metrics. This is the only narrow
        except path that does NOT propagate. Verified by asserting the patched
        evaluate_cycle was reached and execute_cycle still returned a complete
        metrics dict (rather than relying on caplog, which is unreliable across
        the full-suite logging configuration)."""

        def fake_evaluate_cycle(*args, **kwargs):
            raise pl.exceptions.ColumnNotFoundError(
                "forced column-missing failure for narrow-except test"
            )

        initial_ids = sample_compounds['ID'].head(2).to_list()
        initial_values = dict(zip(initial_ids, [0.3, 0.6], strict=True))
        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            target_col='Activity',
            initial_labeled_ids=initial_ids,
            initial_measurements=initial_values,
        )

        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.1)

        with patch(
            'learnm8.core.cycle.evaluate_cycle',
            side_effect=fake_evaluate_cycle,
        ) as mocked_eval:
            _updated_df, metrics = execute_cycle(
                compounds_df=master_df,
                cycle=1,
                config=config,
                learner=mock_learner,
                oracle=mock_oracle,
                target_col='Activity',
                featurizer='morgan',
                cache_dir=tmp_path,
                original_pool_size=len(sample_compounds),
                oracle_type='run',
                output_dir=tmp_path,
            )

        # The patched evaluate_cycle was reached (raised ColumnNotFoundError).
        assert mocked_eval.called
        # Execution continued past the eval-wrap try/except (REQ-2 behaviour).
        assert metrics['cycle'] == 1
        assert 'training_time' in metrics
        assert 'evaluation_time' in metrics
        # Eval-derived keys are absent because the exception aborted the
        # `metrics.update(eval_metrics)` call.
        assert 'r2' not in metrics
        assert 'mae' not in metrics


@pytest.mark.unit
class TestBuildSelectionOrderLookup:
    """build_selection_order_lookup: strict id->order map (Item 14)."""

    def test_maps_ids_to_their_index(self):
        lookup = build_selection_order_lookup(['C3', 'C1', 'C2'])
        assert lookup('C3') == 0
        assert lookup('C1') == 1
        assert lookup('C2') == 2

    def test_missing_id_raises_instead_of_sentinel(self):
        lookup = build_selection_order_lookup(['C1', 'C2'], context='Selected')
        with pytest.raises(LearnM8Error, match='missing from the selection order map'):
            lookup('C_not_present')

    def test_error_message_includes_context_and_id(self):
        lookup = build_selection_order_lookup(['C1'], context='Initial-batch')
        with pytest.raises(LearnM8Error, match=r"Initial-batch compound ID 'ghost'"):
            lookup('ghost')

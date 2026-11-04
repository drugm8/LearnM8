"""Tests for learner creation via API."""

import numpy as np
import polars as pl
import pytest

from learnm8.api import _create_learner, list_available_learners
from learnm8.core.interfaces import Learner
from learnm8.exceptions import ConfigurationError, LearnerError
from learnm8.learners.ensemble.dt_ensemble import DTEnsemble
from learnm8.learners.sklearn.random_forest import RandomForestLearner

_GPU_ONLY_KEYS = {'rf_fil', 'ridge_cuml', 'gpu_gp', 'svgp'}


@pytest.mark.unit
class TestLearnerCreation:
    """Test learner instantiation through API."""

    def test_create_individual_learner_with_random_state(self):
        """Test creating individual learner with random_state."""
        learner = _create_learner('rf', random_state=42)

        assert isinstance(learner, RandomForestLearner)
        assert learner.random_state == 42

    def test_create_ensemble_learner_with_random_state(self):
        """Test creating ensemble learner converts random_state to random_states."""
        learner = _create_learner('dt_ensemble', random_state=42)

        assert isinstance(learner, DTEnsemble)
        assert learner.random_states == [42, 123, 356]
        assert len(learner.learners) == 3

    def test_create_ensemble_learner_reproducibility(self):
        """Test that same random_state produces identical ensemble configuration."""
        learner1 = _create_learner('dt_ensemble', random_state=100)
        learner2 = _create_learner('dt_ensemble', random_state=100)

        assert learner1.random_states == learner2.random_states
        assert learner1.random_states == [100, 181, 414]

    def test_create_ensemble_learner_diversity(self):
        """Test that different random_states produce different configurations."""
        learner1 = _create_learner('dt_ensemble', random_state=42)
        learner2 = _create_learner('dt_ensemble', random_state=100)

        assert learner1.random_states != learner2.random_states

    @pytest.mark.parametrize('learner_str', list_available_learners())
    def test_create_all_available_learners(self, learner_str):
        """Every registered learner key can be created."""
        try:
            learner = _create_learner(learner_str, random_state=42)
        except ImportError as e:
            if learner_str in _GPU_ONLY_KEYS:
                pytest.skip(f"'{learner_str}' needs unavailable GPU libs: {e}")
            raise
        except (LearnerError, RuntimeError) as e:
            if learner_str in _GPU_ONLY_KEYS and any(
                t in str(e).lower() for t in ('cuda', 'cuml', 'gpu')
            ):
                pytest.skip(f"'{learner_str}' needs unavailable GPU: {e}")
            raise
        assert isinstance(learner, Learner)

    def test_bare_ensemble_shortcut_removed(self):
        """The bare 'ensemble' shortcut is no longer a registered learner."""
        assert 'ensemble' not in list_available_learners()
        with pytest.raises(ConfigurationError, match='Unknown learner'):
            _create_learner('ensemble', random_state=42)

    def test_rf_ensemble_removed(self):
        """The deprecated 'rf_ensemble' learner was removed in 1.0."""
        assert 'rf_ensemble' not in list_available_learners()
        with pytest.raises(ConfigurationError, match='Unknown learner'):
            _create_learner('rf_ensemble', random_state=42)

    def test_unknown_learner_raises_error(self):
        """Test error handling for unknown learner string."""
        with pytest.raises(ConfigurationError, match='Unknown learner'):
            _create_learner('nonexistent_learner', random_state=42)

    def test_chemprop_fine_tuning_flag_honoured(self, tmp_path):
        """enable_chemprop_fine_tuning reaches a single ChempropLearner, seed kept."""
        from learnm8.learners.torch.chemprop_learner import ChempropLearner

        learner = _create_learner(
            'chemprop',
            random_state=7,
            enable_chemprop_fine_tuning=True,
            checkpoint_dir=tmp_path,
        )
        assert isinstance(learner, ChempropLearner)
        assert learner.enable_fine_tuning is True
        assert learner.checkpoint_dir == tmp_path
        assert learner.random_state == 7

    def test_chemprop_fine_tuning_requires_checkpoint_dir(self):
        with pytest.raises(ConfigurationError, match='checkpoint_dir required'):
            _create_learner(
                'chemprop', random_state=42, enable_chemprop_fine_tuning=True
            )


@pytest.mark.slow
class TestAPIWithEnsembles:
    """Test run_active_learning with ensemble learners."""

    def test_run_active_learning_with_string_ensemble(
        self, small_real_compounds, tmp_path
    ):
        """Test Pattern 1: String-based ensemble creation."""
        from learnm8 import run_active_learning

        oracle_df = small_real_compounds.clone()
        oracle_df = oracle_df.with_columns(
            pl.Series('Activity', np.random.beta(2, 5, len(oracle_df)))
        )
        oracle_path = tmp_path / 'oracle.csv'
        oracle_df.write_csv(oracle_path)

        results = run_active_learning(
            compound_pool=small_real_compounds,
            oracle=str(oracle_path),
            learner='dt_ensemble',
            target_col='Activity',
            featurizer='morgan',
            random_state=42,
            n_cycles=2,
            batch_fraction=0.1,
            cache_dir=tmp_path / '.cache',
            output_dir=tmp_path / 'output',
        )

        assert results is not None
        assert 'compounds_df' in results
        assert len(results['cycle_metrics']) == 2

    def test_run_active_learning_with_instance_ensemble(
        self, small_real_compounds, tmp_path
    ):
        """Test Pattern 2: Pre-constructed ensemble."""
        from learnm8 import run_active_learning
        from learnm8.learners.ensemble import DTEnsemble

        oracle_df = small_real_compounds.clone()
        oracle_df = oracle_df.with_columns(
            pl.Series('Activity', np.random.beta(2, 5, len(oracle_df)))
        )
        oracle_path = tmp_path / 'oracle.csv'
        oracle_df.write_csv(oracle_path)

        ensemble = DTEnsemble(random_states=[10, 20, 30])

        results = run_active_learning(
            compound_pool=small_real_compounds,
            oracle=str(oracle_path),
            learner=ensemble,
            target_col='Activity',
            featurizer='morgan',
            n_cycles=2,
            batch_fraction=0.1,
            cache_dir=tmp_path / '.cache',
            output_dir=tmp_path / 'output',
        )

        assert results is not None
        assert 'compounds_df' in results

    def test_ensemble_reproducibility_via_api(self, small_real_compounds, tmp_path):
        """Test that same random_state produces reproducible results."""
        from learnm8 import run_active_learning

        oracle_df = small_real_compounds.clone()
        oracle_df = oracle_df.with_columns(
            pl.Series('Activity', np.random.beta(2, 5, len(oracle_df)))
        )
        oracle_path = tmp_path / 'oracle.csv'
        oracle_df.write_csv(oracle_path)

        results1 = run_active_learning(
            compound_pool=small_real_compounds,
            oracle=str(oracle_path),
            learner='dt_ensemble',
            target_col='Activity',
            featurizer='morgan',
            random_state=42,
            n_cycles=2,
            batch_fraction=0.1,
            cache_dir=tmp_path / '.cache1',
            output_dir=tmp_path / 'output1',
        )

        results2 = run_active_learning(
            compound_pool=small_real_compounds,
            oracle=str(oracle_path),
            learner='dt_ensemble',
            target_col='Activity',
            featurizer='morgan',
            random_state=42,
            n_cycles=2,
            batch_fraction=0.1,
            cache_dir=tmp_path / '.cache2',
            output_dir=tmp_path / 'output2',
        )

        df1 = results1['compounds_df']
        df2 = results2['compounds_df']

        selected1 = set(df1.filter(pl.col('status') == 'labeled')['ID'])
        selected2 = set(df2.filter(pl.col('status') == 'labeled')['ID'])
        assert selected1 == selected2

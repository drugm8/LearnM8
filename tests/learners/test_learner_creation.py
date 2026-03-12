"""Tests for learner creation via API."""

import pytest
import numpy as np
import polars as pl
from pathlib import Path
from learnm8.api import _create_learner, list_available_learners
from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.learners.ensemble.rf_ensemble import RFEnsemble


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
        learner = _create_learner('rf_ensemble', random_state=42)

        assert isinstance(learner, RFEnsemble)
        assert learner.random_states == [42, 123, 356]
        assert len(learner.learners) == 3

    def test_create_ensemble_learner_reproducibility(self):
        """Test that same random_state produces identical ensemble configuration."""
        learner1 = _create_learner('rf_ensemble', random_state=100)
        learner2 = _create_learner('rf_ensemble', random_state=100)

        assert learner1.random_states == learner2.random_states
        assert learner1.random_states == [100, 181, 414]

    def test_create_ensemble_learner_diversity(self):
        """Test that different random_states produce different configurations."""
        learner1 = _create_learner('rf_ensemble', random_state=42)
        learner2 = _create_learner('rf_ensemble', random_state=100)

        assert learner1.random_states != learner2.random_states

    def test_create_all_available_learners(self):
        """Test that all registered learners can be created."""
        available = list_available_learners()

        for learner_str in available:
            if learner_str == 'ensemble':
                continue

            try:
                learner = _create_learner(learner_str, random_state=42)
                assert learner is not None
            except Exception as e:
                pytest.fail(f"Failed to create learner '{learner_str}': {e}")

    def test_unknown_learner_raises_error(self):
        """Test error handling for unknown learner string."""
        with pytest.raises(ValueError, match="Unknown learner"):
            _create_learner('nonexistent_learner', random_state=42)


@pytest.mark.unit
class TestAPIWithEnsembles:
    """Test run_active_learning with ensemble learners."""

    def test_run_active_learning_with_string_ensemble(self, small_real_compounds, tmp_path):
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
            learner='rf_ensemble',
            target_col='Activity',
            featurizer='morgan',
            random_state=42,
            n_cycles=2,
            batch_fraction=0.1,
            cache_dir=tmp_path / '.cache',
            output_dir=tmp_path / 'output'
        )

        assert results is not None
        assert 'compounds_df' in results
        assert len(results['cycle_metrics']) == 2

    def test_run_active_learning_with_instance_ensemble(self, small_real_compounds, tmp_path):
        """Test Pattern 2: Pre-constructed ensemble."""
        from learnm8 import run_active_learning
        from learnm8.learners.ensemble import RFEnsemble

        oracle_df = small_real_compounds.clone()
        oracle_df = oracle_df.with_columns(
            pl.Series('Activity', np.random.beta(2, 5, len(oracle_df)))
        )
        oracle_path = tmp_path / 'oracle.csv'
        oracle_df.write_csv(oracle_path)

        ensemble = RFEnsemble(
            n_estimators=50,
            random_states=[10, 20, 30]
        )

        results = run_active_learning(
            compound_pool=small_real_compounds,
            oracle=str(oracle_path),
            learner=ensemble,
            target_col='Activity',
            featurizer='morgan',
            n_cycles=2,
            batch_fraction=0.1,
            cache_dir=tmp_path / '.cache',
            output_dir=tmp_path / 'output'
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
            learner='rf_ensemble',
            target_col='Activity',
            featurizer='morgan',
            random_state=42,
            n_cycles=2,
            batch_fraction=0.1,
            cache_dir=tmp_path / '.cache1',
            output_dir=tmp_path / 'output1'
        )

        results2 = run_active_learning(
            compound_pool=small_real_compounds,
            oracle=str(oracle_path),
            learner='rf_ensemble',
            target_col='Activity',
            featurizer='morgan',
            random_state=42,
            n_cycles=2,
            batch_fraction=0.1,
            cache_dir=tmp_path / '.cache2',
            output_dir=tmp_path / 'output2'
        )

        df1 = results1['compounds_df']
        df2 = results2['compounds_df']

        selected1 = set(df1.filter(pl.col('status') == 'labeled')['ID'])
        selected2 = set(df2.filter(pl.col('status') == 'labeled')['ID'])
        assert selected1 == selected2

"""Integration tests for batch prediction in execute_cycle()."""

import pytest
import polars as pl
import numpy as np
from pathlib import Path

from learnm8.core.cycle import execute_cycle
from learnm8.core.config import CycleConfig
from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.oracles.csv_oracle import CSVOracle


@pytest.fixture
def oracle_csv(initialized_compounds, tmp_path):
    """Create oracle CSV with activity values for all compounds."""
    import csv
    csv_path = tmp_path / 'oracle.csv'
    ids = initialized_compounds['ID'].to_list()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Activity'])
        for i, cid in enumerate(ids):
            writer.writerow([cid, 5.0 + i * 0.1])
    return csv_path


@pytest.fixture
def initialized_compounds(sample_compounds, tmp_path):
    """Create initialized master DataFrame with some labeled compounds."""
    compounds_df = sample_compounds.clone()
    compounds_df = compounds_df.with_columns([
        pl.lit('unlabeled').alias('status'),
        pl.lit(None).cast(pl.Int64).alias('selected_cycle'),
        pl.lit(None).cast(pl.Int64).alias('labeled_cycle'),
        pl.lit(None).cast(pl.Int64).alias('pruned_cycle'),
        pl.lit(None).cast(pl.Float64).alias('Activity')
    ])

    labeled_indices = [0, 1, 2]
    compounds_df = compounds_df.with_columns([
        pl.when(pl.col('ID').is_in(compounds_df[labeled_indices]['ID']))
        .then(pl.lit('labeled'))
        .otherwise(pl.col('status')).alias('status'),

        pl.when(pl.col('ID').is_in(compounds_df[labeled_indices]['ID']))
        .then(pl.lit(0))
        .otherwise(pl.col('selected_cycle')).alias('selected_cycle'),

        pl.when(pl.col('ID').is_in(compounds_df[labeled_indices]['ID']))
        .then(pl.lit(0))
        .otherwise(pl.col('labeled_cycle')).alias('labeled_cycle'),

        pl.when(pl.col('ID').is_in(compounds_df[labeled_indices]['ID']))
        .then(pl.lit(5.0))
        .otherwise(pl.col('Activity')).alias('Activity')
    ])

    return compounds_df


@pytest.mark.integration
class TestCycleBatchPrediction:
    """Test batch prediction integration in execute_cycle()."""

    def test_auto_batch_size_small_dataset(self, initialized_compounds, oracle_csv, tmp_path):
        """Small dataset (≤100k) should use single batch automatically."""
        learner = RandomForestLearner(random_state=42)
        oracle = CSVOracle(str(oracle_csv))

        config = CycleConfig(strategy='greedy', n_cycles=1, batch_fraction=0.1)

        compounds_df, metrics = execute_cycle(
            compounds_df=initialized_compounds,
            cycle=1,
            config=config,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(initialized_compounds),
            score_direction='higher',
            mode='run',
            prediction_batch_size=None
        )

        assert 'batch_size' in metrics
        assert compounds_df.filter(pl.col('selected_cycle') == 1).height > 0

    def test_explicit_batch_size(self, initialized_compounds, oracle_csv, tmp_path):
        """Explicit batch size should be used for prediction."""
        learner = RandomForestLearner(random_state=42)
        oracle = CSVOracle(str(oracle_csv))

        config = CycleConfig(strategy='greedy', n_cycles=1, batch_fraction=0.1)

        compounds_df, metrics = execute_cycle(
            compounds_df=initialized_compounds,
            cycle=1,
            config=config,
            learner=learner,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(initialized_compounds),
            score_direction='higher',
            mode='run',
            prediction_batch_size=5
        )

        assert 'batch_size' in metrics
        assert compounds_df.filter(pl.col('selected_cycle') == 1).height > 0

    def test_predictions_identical_across_batch_sizes(self, initialized_compounds, oracle_csv, tmp_path):
        """Predictions should be identical regardless of batch size."""
        learner1 = RandomForestLearner(random_state=42)
        learner2 = RandomForestLearner(random_state=42)
        oracle = CSVOracle(str(oracle_csv))

        config = CycleConfig(strategy='greedy', n_cycles=1, batch_fraction=0.1)

        compounds_df1, metrics1 = execute_cycle(
            compounds_df=initialized_compounds.clone(),
            cycle=1,
            config=config,
            learner=learner1,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(initialized_compounds),
            score_direction='higher',
            mode='run',
            prediction_batch_size=None
        )

        compounds_df2, metrics2 = execute_cycle(
            compounds_df=initialized_compounds.clone(),
            cycle=1,
            config=config,
            learner=learner2,
            oracle=oracle,
            target_col='Activity',
            featurizer='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(initialized_compounds),
            score_direction='higher',
            mode='run',
            prediction_batch_size=5
        )

        selected1 = set(compounds_df1.filter(pl.col('selected_cycle') == 1)['ID'].to_list())
        selected2 = set(compounds_df2.filter(pl.col('selected_cycle') == 1)['ID'].to_list())

        assert selected1 == selected2

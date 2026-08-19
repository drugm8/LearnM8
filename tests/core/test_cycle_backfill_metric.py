"""Tests that `acquisition_backfill_fraction` reaches the cycle metrics dict.

Feature 027 (REQ-11): simulated annealing reports what share of each batch came
from greedy backfill rather than the annealing walk. The metric must be present
for every strategy so downstream consumers can read it unconditionally, but
`None` for strategies that do not anneal — a fabricated 0.0 would read as
"annealing supplied the whole batch", which is the exact misattribution this
feature exists to prevent.
"""

import numpy as np
import pytest

from learnm8.core.config import CycleConfig
from learnm8.core.cycle import execute_cycle


def _master_df(compounds, initial_labeled_count=5):
    from tests.fixtures.master_dataframe import create_initialized_master_df

    initial_ids = compounds.slice(0, initial_labeled_count)['ID'].to_list()
    rng = np.random.default_rng(0)
    return create_initialized_master_df(
        valid_compounds=compounds,
        target_col='Activity',
        initial_labeled_ids=initial_ids,
        initial_measurements=dict(
            zip(
                initial_ids,
                rng.uniform(0.1, 0.9, initial_labeled_count),
                strict=True,
            )
        ),
    )


def _run_cycle(strategy, compounds, tmp_path, oracle, learner, **acq_params):
    config = CycleConfig(
        strategy=strategy, batch_fraction=0.2, acquisition_params=acq_params or None
    )
    _, metrics = execute_cycle(
        compounds_df=_master_df(compounds),
        cycle=0,
        config=config,
        learner=learner,
        oracle=oracle,
        target_col='Activity',
        featurizer='morgan',
        cache_dir=tmp_path,
        original_pool_size=len(compounds),
        score_direction='higher',
        oracle_type='run',
        output_dir=tmp_path,
    )
    return metrics


@pytest.mark.integration
class TestAcquisitionBackfillMetric:
    def test_metric_is_a_float_for_simulated_annealing(
        self, medium_real_compounds, tmp_path, mock_oracle, mock_learner
    ):
        metrics = _run_cycle(
            'simulated_annealing',
            medium_real_compounds.clone(),
            tmp_path,
            mock_oracle,
            mock_learner,
            neighbor_strategy='score_band',
            band_width=20,
        )

        assert 'acquisition_backfill_fraction' in metrics
        fraction = metrics['acquisition_backfill_fraction']
        assert isinstance(fraction, float)
        assert 0.0 <= fraction <= 1.0

    def test_metric_is_none_for_greedy(
        self, medium_real_compounds, tmp_path, mock_oracle, mock_learner
    ):
        """No fabricated zeros: greedy does not anneal, so it has no fraction."""
        metrics = _run_cycle(
            'greedy', medium_real_compounds.clone(), tmp_path, mock_oracle, mock_learner
        )

        assert 'acquisition_backfill_fraction' in metrics
        assert metrics['acquisition_backfill_fraction'] is None

    def test_metric_is_none_for_random(
        self, medium_real_compounds, tmp_path, mock_oracle, mock_learner
    ):
        metrics = _run_cycle(
            'random', medium_real_compounds.clone(), tmp_path, mock_oracle, mock_learner
        )

        assert metrics['acquisition_backfill_fraction'] is None

    def test_metric_key_is_always_present(
        self, medium_real_compounds, tmp_path, mock_oracle, mock_learner
    ):
        """Consumers must be able to read the key without a guard."""
        compounds = medium_real_compounds.clone()
        for strategy in ('greedy', 'random', 'topk'):
            metrics = _run_cycle(
                strategy, compounds, tmp_path, mock_oracle, mock_learner
            )
            assert 'acquisition_backfill_fraction' in metrics, strategy

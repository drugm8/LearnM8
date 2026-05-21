"""Regression test for the cumulative-selected off-by-one in execute_cycle.

For benchmark cycles 1..N the cumulative discovery / ranking metrics must
INCLUDE the batch selected in that same cycle. A stale `cumulative_selected_ids`
(captured before `_select_and_measure` labels the new batch) excludes it.
"""

import numpy as np
import polars as pl
import pytest

from learnm8.core.config import CycleConfig
from learnm8.core.cycle import execute_cycle
from learnm8.core.interfaces import Learner
from learnm8.oracles.csv_oracle import CSVOracle
from tests.fixtures.master_dataframe import create_initialized_master_df


class _PerfectLearner(Learner):
    """Learner whose predictions equal a fixed per-ID ground-truth value.

    Greedy acquisition then deterministically selects the true top compounds.
    """

    def __init__(self, smiles_to_value: dict[str, float]):
        self._smiles_to_value = smiles_to_value
        self.is_trained = False

    def train(self, features, targets, smiles=None):
        self.is_trained = True

    def predict(self, features, smiles=None, *, compute_uncertainty=True):
        values = np.asarray(
            [self._smiles_to_value[s] for s in smiles], dtype=np.float64
        )
        return values, None

    def get_name(self):
        return 'perfect-learner'

    def supports_uncertainty(self):
        return False

    def requires_smiles(self):
        return True


def _build_pool(n=12):
    """Pool of n compounds; target = descending integer rank.

    Compound i has target value (n - i), so COMP_000 is the strongest. SMILES
    are distinct, valid alkane chains (canonicalization is idempotent for them,
    so the cycle-side canonical SMILES still keys the learner correctly).
    """
    ids = [f'COMP_{i:03d}' for i in range(n)]
    smiles = ['C' * (i + 1) for i in range(n)]
    targets = [float(n - i) for i in range(n)]
    return ids, smiles, targets


@pytest.mark.integration
def test_benchmark_cycle_discovery_includes_current_batch(tmp_path):
    n = 12
    ids, smiles, targets = _build_pool(n)

    smiles_to_value = dict(zip(smiles, targets, strict=True))
    id_to_value = dict(zip(ids, targets, strict=True))

    pool = pl.DataFrame({'ID': ids, 'SMILES': smiles})

    # Oracle CSV with the full ground truth.
    oracle_csv = tmp_path / 'oracle.csv'
    pl.DataFrame({'ID': ids, 'SMILES': smiles, 'Activity': targets}).write_csv(
        oracle_csv
    )
    oracle = CSVOracle(str(oracle_csv))

    # Cycle-0 init: label the two WEAKEST compounds (ranks 11, 12), neither in
    # the true top-10. So pre-cycle top_10_discovery is 0.
    init_ids = ids[-2:]
    init_measurements = {cid: id_to_value[cid] for cid in init_ids}
    master_df = create_initialized_master_df(
        valid_compounds=pool,
        target_col='Activity',
        initial_labeled_ids=init_ids,
        initial_measurements=init_measurements,
    )

    learner = _PerfectLearner(smiles_to_value)

    # batch_fraction picks ceil(12 * 0.4) = 5 compounds; greedy selects the
    # 5 strongest unlabeled compounds (COMP_000..COMP_004), all in true top-10.
    config = CycleConfig(strategy='greedy', batch_fraction=0.4)

    _updated_df, metrics = execute_cycle(
        compounds_df=master_df,
        cycle=1,
        config=config,
        learner=learner,
        oracle=oracle,
        target_col='Activity',
        featurizer='morgan',
        cache_dir=tmp_path,
        original_pool_size=n,
        score_direction='higher',
        oracle_type='benchmark',
        original_pool=oracle.ground_truth,
        output_dir=tmp_path,
    )

    selected_count = metrics['selected_count']
    assert selected_count == 5

    # cumulative_labeled counts the new batch: 2 init + 5 selected = 7.
    assert metrics['cumulative_labeled'] == 7

    # All 5 selected this cycle ARE in the true top-10. Discovery must reflect
    # them: 5 of 10 true top-10 discovered -> 50.0%. With the off-by-one the
    # stale cumulative set excludes the cycle's own batch -> 0.0%.
    assert metrics['top_10_discovery'] == pytest.approx(50.0)

"""Run-mode oracle failures: per-compound failures mark ``failed``, not raise."""

import math

import numpy as np
import polars as pl
import pytest

from learnm8 import run_active_learning
from learnm8.core.interfaces import Oracle
from learnm8.exceptions import OracleError

pytestmark = pytest.mark.integration

POOL_SIZE = 40


def _pool() -> pl.DataFrame:
    df = pl.read_csv('tests/data/small_molecules.csv').head(POOL_SIZE)
    return df.select(['ID', 'SMILES'])


class _FlakyOracle(Oracle):
    """NaN for ``nan_ids``, omitted rows for ``omit_ids``, else a numeric value."""

    def __init__(self, pool, nan_ids, omit_ids):
        rng = np.random.default_rng(0)
        self.values = dict(
            zip(pool['ID'].to_list(), rng.normal(size=pool.height), strict=True)
        )
        self.nan_ids = set(nan_ids)
        self.omit_ids = set(omit_ids)

    def measure(self, compounds, properties):
        rows = []
        for cid in compounds['ID'].to_list():
            if cid in self.omit_ids:
                continue
            v = math.nan if cid in self.nan_ids else self.values[cid]
            rows.append((cid, v))
        return pl.DataFrame(
            rows, schema={'ID': pl.Utf8, 'Activity': pl.Float64}, orient='row'
        )


@pytest.mark.parametrize('streaming', ['never', 'always'])
def test_failed_compounds_are_marked_and_never_reselected(tmp_path, streaming):
    pool = _pool()
    ids = pool['ID'].to_list()
    nan_ids = ids[0::5]
    omit_ids = ids[2::7]
    failing = set(nan_ids) | set(omit_ids)
    oracle = _FlakyOracle(pool, nan_ids, omit_ids)

    results = run_active_learning(
        compound_pool=pool,
        oracle=oracle,
        target_col='Activity',
        learner='rf',
        featurizer='morgan',
        strategy='greedy',
        n_cycles=3,
        batch_fraction=0.2,
        random_state=1,
        output_dir=tmp_path,
        streaming=streaming,
    )

    df = results['compounds_df']
    assert len(df.columns) == 7
    failed = df.filter(pl.col('status') == 'failed')
    selected = df.filter(pl.col('selected_cycle').is_not_null())
    expected_failed = set(selected['ID'].to_list()) & failing
    assert expected_failed
    assert set(failed['ID'].to_list()) == expected_failed
    assert failed['Activity'].null_count() == failed.height
    assert (failed['selected_cycle'] == failed['labeled_cycle']).all()

    labeled = df.filter(pl.col('status') == 'labeled')
    assert not set(labeled['ID'].to_list()) & failing
    assert labeled['Activity'].null_count() == 0

    metrics = results['cycle_metrics']
    for m in metrics:
        c = m['cycle']
        n = failed.filter(pl.col('labeled_cycle') == c).height
        assert m['n_failed'] == n
        # Each compound is selected in exactly one cycle (never reselected).
        assert m['selected_count'] == df.filter(pl.col('selected_cycle') == c).height
    assert sum(m['n_failed'] for m in metrics) == failed.height


def test_all_failed_initial_batch_raises(tmp_path):
    pool = _pool()
    oracle = _FlakyOracle(pool, nan_ids=pool['ID'].to_list(), omit_ids=[])
    with pytest.raises(OracleError, match='failed'):
        run_active_learning(
            compound_pool=pool,
            oracle=oracle,
            target_col='Activity',
            learner='rf',
            featurizer='morgan',
            n_cycles=3,
            batch_fraction=0.2,
            output_dir=tmp_path,
        )

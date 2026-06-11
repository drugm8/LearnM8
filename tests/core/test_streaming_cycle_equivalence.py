"""End-to-end equivalence: streaming path vs legacy path through execute_cycle.

Deterministic strategies (greedy/ucb/ei/pi/entropy) must select the IDENTICAL
batch under streaming and legacy on tie-free data, and persisted parquet content
must match. Also covers selection_path observability, pruning parity, and the
compute_uncertainty=False column-drop.
"""

import numpy as np
import polars as pl
import pytest

from learnm8.core.config import CycleConfig
from learnm8.core.cycle import execute_cycle
from learnm8.core.initialization import initialize_master_dataframe_empty
from learnm8.core.interfaces import Learner, Oracle
from learnm8.core.persistence import read_cycle_predictions

pytestmark = pytest.mark.unit

N = 400


def _make_pool(n=N, seed=0):
    """Diverse, valid, deterministic acyclic C/O/N SMILES → distinct fingerprints.

    Distinct fingerprints make the learner's predictions genuinely tie-free, so
    the streaming-vs-legacy selection comparison is not confounded by tie order.
    """
    rng = np.random.default_rng(seed)
    atoms = np.array(['C', 'C', 'C', 'C', 'O', 'N'])
    smiles = []
    lengths = rng.integers(6, 26, size=n)
    for i in range(n):
        chars = rng.choice(atoms, size=int(lengths[i]))
        parts = []
        for j, a in enumerate(chars):
            parts.append(str(a))
            if a == 'C' and 0 < j < len(chars) - 1 and rng.random() < 0.15:
                parts.append('(C)')
        smiles.append(''.join(parts))
    return pl.DataFrame({'ID': [f'c{i}' for i in range(n)], 'SMILES': smiles})


class _FeatureLearner(Learner):
    """Deterministic learner: prediction = hash(ID-free) of feature sum; supports unc."""

    def __init__(self, with_uncertainty=False, seed=0):
        self._with_unc = with_uncertainty
        self._seed = seed
        self._coef = None

    def train(self, features, targets, smiles=None):
        rng = np.random.default_rng(self._seed)
        self._coef = rng.normal(size=features.shape[1])

    def predict(self, features, smiles=None, *, compute_uncertainty=True):
        # Deterministic prediction in (0, 1) via a sigmoid of a fixed linear
        # projection. Chunk-position-independent and tie-free (distinct
        # fingerprints), and on the same scale as the oracle targets so EI/PI
        # are non-degenerate (current_best lies within the prediction range).
        raw = (features.astype(np.float64) @ self._coef) / np.sqrt(features.shape[1])
        preds = 1.0 / (1.0 + np.exp(-raw))
        if self._with_unc and compute_uncertainty:
            unc = np.abs(np.sin(preds)) + 0.01
            return preds, unc
        return preds, None

    def supports_uncertainty(self):
        return self._with_unc

    def requires_smiles(self):
        return False

    def get_name(self):
        return 'FeatureLearner'


class _DictOracle(Oracle):
    def __init__(self, pool, target_col, seed=1):
        rng = np.random.default_rng(seed)
        self._vals = dict(
            zip(pool['ID'].to_list(), rng.uniform(0, 1, size=pool.height), strict=True)
        )
        self._target = target_col

    def measure(self, compounds, properties):
        ids = compounds['ID'].to_list()
        return pl.DataFrame({'ID': ids, self._target: [self._vals[i] for i in ids]})


def _run_cycle(tmp_path, strategy, streaming, *, with_unc=False, pruning=None,
               featurizer='morgan', acq_params=None):
    pool = _make_pool()
    master = initialize_master_dataframe_empty(pool, 'Activity')
    # Seed cycle 0: label a few compounds so training has data.
    from learnm8.core.dataframe_ops import update_status

    init_ids = [f'c{i}' for i in range(8)]
    oracle = _DictOracle(pool, 'Activity')
    init_vals = pl.Series('v', [oracle._vals[i] for i in init_ids])
    master = update_status(master, init_ids, 'labeled', 0, 'Activity', init_vals)

    learner = _FeatureLearner(with_uncertainty=with_unc)
    out = tmp_path / streaming
    out.mkdir(parents=True)
    config = CycleConfig(
        strategy, n_cycles=1, batch_fraction=0.05,
        pruning_strategy=pruning,
        pruning_params={'pruning_fraction': 0.3} if pruning else None,
        acquisition_params=acq_params,
    )
    df, metrics = execute_cycle(
        compounds_df=master, cycle=1, config=config, learner=learner, oracle=oracle,
        target_col='Activity', featurizer=featurizer, cache_dir=tmp_path / 'cache',
        original_pool_size=N, oracle_type='run', output_dir=out,
        random_state=42, n_jobs=1, streaming=streaming,
    )
    return df, metrics, out


class TestDeterministicEquivalence:
    @pytest.mark.parametrize('strategy,with_unc,acq', [
        ('greedy', False, None),
        ('ucb', True, {'beta': 2.0}),
        ('ei', True, None),
        ('pi', True, None),
        ('entropy', True, None),
    ])
    def test_streaming_matches_legacy_selection(self, tmp_path, strategy, with_unc, acq):
        _, m_legacy, _ = _run_cycle(tmp_path / 'L', strategy, 'never',
                                    with_unc=with_unc, acq_params=acq)
        _, m_stream, _ = _run_cycle(tmp_path / 'S', strategy, 'always',
                                    with_unc=with_unc, acq_params=acq)
        assert m_legacy['selection_path'] == 'legacy'
        assert m_stream['selection_path'] == 'streaming'
        assert sorted(m_stream['selected_ids']) == sorted(m_legacy['selected_ids'])

    def test_prediction_stats_parity(self, tmp_path):
        _, m_legacy, _ = _run_cycle(tmp_path / 'L', 'greedy', 'never')
        _, m_stream, _ = _run_cycle(tmp_path / 'S', 'greedy', 'always')
        for key in ('prediction_mean', 'prediction_std', 'prediction_median',
                    'prediction_min', 'prediction_max'):
            assert m_legacy[key] == pytest.approx(m_stream[key], rel=1e-5, abs=1e-6)


class TestPruningParity:
    def test_pruned_count_and_disjoint(self, tmp_path):
        _df_l, m_l, _ = _run_cycle(tmp_path / 'L', 'greedy', 'never', pruning='score')
        df_s, m_s, _ = _run_cycle(tmp_path / 'S', 'greedy', 'always', pruning='score')
        assert m_s['pruned_count'] == m_l['pruned_count']
        assert m_s['pruned_count'] > 0
        # selected compounds are never pruned
        pruned = set(df_s.filter(pl.col('status') == 'pruned')['ID'].to_list())
        assert not (set(m_s['selected_ids']) & pruned)


class TestUncertaintyColumnDrop:
    def test_no_uncertainty_column_when_skipped(self, tmp_path):
        # greedy + non-uncertainty learner → compute_uncertainty False → no column
        _, m, _out = _run_cycle(tmp_path, 'greedy', 'always', with_unc=False)
        df = read_cycle_predictions(m['parquet_path'])
        assert 'uncertainty' not in df.columns
        assert m['has_uncertainty'] is False

    def test_uncertainty_column_present_when_used(self, tmp_path):
        _, m, _out = _run_cycle(tmp_path, 'ucb', 'always', with_unc=True,
                               acq_params={'beta': 1.0})
        df = read_cycle_predictions(m['parquet_path'])
        assert 'uncertainty' in df.columns
        assert m['has_uncertainty'] is True


class TestEligibilityFallback:
    def test_always_on_ineligible_raises(self, tmp_path):
        from learnm8.exceptions import ConfigurationError
        # simulated_annealing is not streamable → 'always' must raise
        with pytest.raises(ConfigurationError):
            _run_cycle(tmp_path, 'simulated_annealing', 'always')

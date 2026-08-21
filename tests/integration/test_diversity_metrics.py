"""Unit tests for _calculate_cycle_metrics diversity integration
and one integration test for seeded reproducibility (feature 013).

The first three tests replaced expensive pipeline runs with mocked
evaluate_cycle, saving ~25s from the non-slow suite. Diversity metrics
are validated at the _calculate_cycle_metrics dict level rather than
CSV parse-back.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from learnm8 import run_active_learning
from learnm8.core.cycle import _calculate_cycle_metrics
from learnm8.evaluation.metrics.similarity import DIVERSITY_KEYS

SMILES_POOL = [
    'CCO',
    'CCC',
    'CCCO',
    'c1ccccc1',
    'C1CCCCC1',
    'Nc1ccccc1',
    'CC(=O)Oc1ccccc1C(=O)O',
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
    'C1=CC=C(C=C1)C(=O)O',
    'CCN(CC)CC',
    'CCCC',
    'CCCCC',
    'CCCCCC',
    'Nc1ncnc2[nH]cnc12',
    'Cn1cnc2c1c(=O)[nH]c(=O)n2C',
    'Oc1ccccc1',
    'CCOc1ccccc1',
    'ClCC(Cl)Cl',
    'BrCCBr',
    'CC(=O)O',
    'CCO[H]',
    'CCN',
    'CCCCCCN',
    'CCSCC',
    'Cc1ccccc1',
    'Cc1ccc(C)cc1',
    'CC(C)C',
    'CC(C)(C)C',
    'C1CCNCC1',
]


def _make_dataset(n: int = 30) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    smis = (SMILES_POOL * ((n // len(SMILES_POOL)) + 1))[:n]
    return pl.DataFrame(
        {
            'ID': [f'mol_{i}' for i in range(n)],
            'SMILES': smis,
            'Activity': rng.uniform(-1.0, 1.0, size=n).tolist(),
        }
    )


def _make_mock_eval_return(
    *, all_none: bool = False, none_keys: frozenset[str] = frozenset()
) -> dict:
    result: dict = {}
    for key in DIVERSITY_KEYS:
        result[key] = None if (all_none or key in none_keys) else 0.5
    result['fingerprint_used'] = None if all_none else 'morgan_fp_2_2048'
    return result


@pytest.fixture
def workdir():
    tmp = Path(tempfile.mkdtemp(prefix='learnm8_diversity_it_'))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.unit
class TestCalculateCycleMetricsDiversity:
    """Unit tests verifying _calculate_cycle_metrics integrates diversity
    keys from evaluate_cycle with stable schema regardless of disable."""

    @staticmethod
    def _call_with_mock(sample_master_df, mock_eval_return, **kwargs):
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        labeled_ids = (
            sample_master_df.filter(pl.col('status') == 'labeled')
            .get_column('ID')
            .to_list()
        )
        cycle_predictions = pl.DataFrame(
            {
                'ID': labeled_ids,
                'prediction': [float(0.5 + i * 0.1) for i in range(len(labeled_ids))],
            }
        )

        with patch('learnm8.core.cycle.evaluate_cycle', return_value=mock_eval_return):
            return _calculate_cycle_metrics(
                compounds_df=sample_master_df,
                cycle=1,
                strategy='greedy',
                predictions=predictions,
                uncertainties=None,
                selected_ids=['FAKE_001'],
                pruned_ids=[],
                target_col='Activity',
                score_direction='higher',
                cycle_predictions=cycle_predictions,
                **kwargs,
            )

    def test_all_six_keys_present_with_correct_types(self, sample_master_df):
        mock_eval = _make_mock_eval_return()
        metrics = self._call_with_mock(sample_master_df, mock_eval)

        for key in DIVERSITY_KEYS:
            assert key in metrics, f'missing column {key}'
            val = metrics[key]
            assert val is None or isinstance(val, float), (
                f'{key} dtype should be float or None, got {type(val)}'
            )
        assert 'fingerprint_used' in metrics
        fp = metrics['fingerprint_used']
        assert fp is None or isinstance(fp, str)

    def test_disable_true_emits_none_for_all_keys(self, sample_master_df):
        mock_eval = _make_mock_eval_return(all_none=True)
        metrics = self._call_with_mock(
            sample_master_df,
            mock_eval,
            disable_molecular_similarity=True,
        )

        for key in DIVERSITY_KEYS:
            assert metrics[key] is None, f'{key} should be all-null'
        assert metrics['fingerprint_used'] is None

    def test_disable_iterable_emits_none_for_named_only(self, sample_master_df):
        disabled_key = 'scaffold_diversity_index_batch'
        mock_eval = _make_mock_eval_return(none_keys=frozenset({disabled_key}))
        metrics = self._call_with_mock(
            sample_master_df,
            mock_eval,
            disable_molecular_similarity=[disabled_key],
        )

        assert metrics[disabled_key] is None
        other_key = 'scaffold_diversity_index_cumulative'
        assert metrics[other_key] == 0.5


@pytest.mark.integration
class TestDiversityReproducibility:
    """One integration test for seeded reproducibility across pipelines."""

    def test_seeded_runs_produce_identical_metric_columns(self, workdir):
        df = _make_dataset(n=24)
        out_a = workdir / 'run_seed_a'
        out_b = workdir / 'run_seed_b'
        for out in (out_a, out_b):
            run_active_learning(
                compound_pool=df,
                oracle=df.select(['ID', 'Activity']),
                learner='rf',
                target_col='Activity',
                featurizer='morgan',
                n_cycles=2,
                batch_fraction=0.15,
                random_state=12345,
                output_dir=out,
            )
        schema_overrides = {key: pl.Float64 for key in DIVERSITY_KEYS}
        schema_overrides['fingerprint_used'] = pl.Utf8
        ma = pl.read_csv(
            out_a / 'cycle_metrics.csv',
            comment_prefix='#',
            schema_overrides=schema_overrides,
        )
        mb = pl.read_csv(
            out_b / 'cycle_metrics.csv',
            comment_prefix='#',
            schema_overrides=schema_overrides,
        )
        for key in DIVERSITY_KEYS:
            va = ma.get_column(key).to_list()
            vb = mb.get_column(key).to_list()
            for i, (a, b) in enumerate(zip(va, vb, strict=True)):
                if a is None and b is None:
                    continue
                assert a is not None and b is not None, (
                    f'row {i} key {key} mismatch on null: {a} vs {b}'
                )
                assert np.isclose(a, b, equal_nan=True), (
                    f'row {i} key {key} differs: {a} vs {b}'
                )

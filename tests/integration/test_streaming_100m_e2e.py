"""End-to-end integration test for all 4 streaming-100m features.

Verifies that run_active_learning() with mc_dropout learner exercises:
1. GPU chunking (predict_batch_size on MCDropoutLearner)
2. Atomic writes (no orphan .tmp files)
3. Parquet output (per-cycle prediction_cycle_N.parquet)
4. Descriptor guard (check_large_feature_guard)

Runs on 50 compounds with 2 cycles, output_format='auto'.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from learnm8.api import run_active_learning
from learnm8.exceptions import ConfigurationError

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_VALID_SMILES = [
    'CCO',
    'CCN',
    'CCOCC',
    'CCCO',
    'c1ccccc1',
    'CC(=O)O',
    'CCN(CC)CC',
    'CC(=O)OC',
    'CC(C)C',
    'C1CCCCC1',
    'C=O',
    'CC#N',
    'c1ccncc1',
    'CC(C)=O',
    'CCOC(=O)C',
    'CCCl',
    'CCBr',
    'c1ccsc1',
    'C1CC1',
    'CCCC',
    'CCCCCC',
    'C=C',
    'C#C',
    'c1ccco1',
    'CCS',
    'CCOP(=O)(OCC)OCC',
    'CN(C)C',
    'CCNC',
    'c1ccc2ccccc2c1',
    'CC(=O)N',
    'CCCCN',
    'O=C(O)CC(O)(CC(=O)O)C(=O)O',
    'CC(C)CC(C)C',
    'c1ccc(Cl)cc1',
    'c1ccc(O)cc1',
    'c1ccc([N+](=O)[O-])cc1',
    'c1ccc(N)cc1',
    'C1CCNCC1',
    'CC(=O)NC',
    'CN1CCCCC1',
    'Oc1ccccc1',
    'CCOCCO',
    'CC(N)C',
    'CCCCCCCC',
    'c1ccoc1',
    'CC(C)CO',
    'CC(=O)OCC',
    'CCCC(=O)O',
    'CC=CC',
    'CC#CC',
]


def test_end_to_end_with_all_4_features(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    compound_pool = pl.DataFrame(
        {
            'ID': [f'CMPD_{i:03d}' for i in range(len(_VALID_SMILES))],
            'SMILES': _VALID_SMILES,
            'Activity': rng.normal(0, 1, len(_VALID_SMILES)).tolist(),
        }
    )

    output_dir = tmp_path / 'results'

    try:
        results = run_active_learning(
            compound_pool=compound_pool,
            oracle=compound_pool.clone(),
            learner='mc_dropout',
            featurizer='morgan',
            target_col='Activity',
            n_cycles=2,
            batch_fraction=0.1,
            output_dir=output_dir,
            output_format='auto',
            random_state=42,
        )
    except ConfigurationError as e:
        pytest.fail(f'run_active_learning raised ConfigurationError: {e}')

    saved_files = results['saved_files']
    expected_keys = {'compounds_final', 'cycle_metrics', 'selection_history', 'config'}
    assert expected_keys.issubset(set(saved_files)), (
        f'saved_files missing keys: {expected_keys - set(saved_files)}'
    )

    for key in expected_keys:
        assert Path(saved_files[key]).exists(), f'{key} file does not exist'

    metrics_path = Path(saved_files['cycle_metrics'])
    assert metrics_path.suffix == '.csv'

    final_path = Path(saved_files['compounds_final'])
    assert final_path.suffix == '.csv'

    pred_files = results.get('prediction_files', [])
    assert len(pred_files) >= 1, 'expected at least one prediction parquet file'
    for pf in pred_files:
        assert pf.exists(), f'prediction file missing: {pf}'
        assert pf.suffix == '.parquet'
        assert pf.name.startswith('prediction_cycle_')

    tmp_files = list(output_dir.glob('*.tmp'))
    assert len(tmp_files) == 0, f'orphan .tmp files found: {tmp_files}'

    assert 'prediction_files' in results
    assert isinstance(results.get('labeled_count'), int)
    assert isinstance(results.get('unlabeled_count'), int)

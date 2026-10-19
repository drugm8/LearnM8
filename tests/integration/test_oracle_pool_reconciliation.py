"""Integration tests for pool/oracle reconciliation with a CSV oracle.

A target column containing non-numeric values (e.g. ``"no_score"`` from a
failed docking run) is rejected with ``ValidationError`` rather than silently
dropped. A pool that is larger than the oracle's known IDs is reconciled by
filtering the pool to the measurable compounds.
"""

from pathlib import Path

import polars as pl
import pytest

from learnm8 import run_active_learning
from learnm8.exceptions import ValidationError

ROWS = [
    ('c01', 'CCO', '0.10'),
    ('c02', 'CCC', '0.20'),
    ('c03', 'CCN', 'no_score'),
    ('c04', 'c1ccccc1', '0.30'),
    ('c05', 'COC', '0.40'),
    ('c06', 'CC(=O)O', 'no_score'),
    ('c07', 'CCBr', '0.50'),
    ('c08', 'CCS', '0.60'),
    ('c09', 'CCF', '0.70'),
    ('c10', 'CCI', '0.80'),
    ('c11', 'C1CCCCC1', 'no_score'),
    ('c12', 'CCOCC', '0.90'),
]


def _write(rows, path: Path, columns=('ID', 'SMILES', 'Activity')) -> Path:
    pl.DataFrame(rows, schema=list(columns), orient='row').write_csv(path)
    return path


@pytest.mark.integration
def test_non_numeric_target_values_raise_validation_error(tmp_path):
    """Three 'no_score' rows in the target column are rejected, not dropped."""
    csv_path = _write(ROWS, tmp_path / 'mixed_score.csv')

    with pytest.raises(ValidationError, match='numeric'):
        run_active_learning(
            compound_pool=str(csv_path),
            oracle=str(csv_path),
            target_col='Activity',
            learner='rf',
            featurizer='morgan',
            n_cycles=2,
            batch_fraction=0.5,
            random_state=42,
            output_dir=str(tmp_path / 'out'),
        )


@pytest.mark.integration
def test_pool_larger_than_oracle_is_filtered_to_known_ids(tmp_path):
    """Pool compounds absent from the oracle are marked invalid, not selected."""
    numeric = [r for r in ROWS if r[2] != 'no_score']
    oracle_csv = _write(numeric, tmp_path / 'oracle.csv')
    pool_csv = _write([r[:2] for r in ROWS], tmp_path / 'pool.csv', ('ID', 'SMILES'))

    results = run_active_learning(
        compound_pool=str(pool_csv),
        oracle=str(oracle_csv),
        target_col='Activity',
        learner='rf',
        featurizer='morgan',
        n_cycles=2,
        batch_fraction=0.5,
        random_state=42,
        output_dir=str(tmp_path / 'out'),
    )

    validation_result = results['validation_result']
    valid_ids = set(validation_result.valid_compounds['ID'].to_list())
    invalid_ids = set(validation_result.invalid_compounds['ID'].to_list())

    assert valid_ids == {r[0] for r in numeric}
    assert invalid_ids == {'c03', 'c06', 'c11'}
    for cid in invalid_ids:
        assert 'oracle' in validation_result.validation_errors[cid].lower()

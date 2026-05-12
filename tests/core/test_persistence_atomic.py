import errno
import logging
from unittest.mock import patch

import polars as pl
import pytest

from learnm8.core.persistence import _atomic_write, save_results
from learnm8.core.validation import ValidationResult
from learnm8.exceptions import LearnM8Warning, PersistenceError

pytestmark = pytest.mark.unit


def test_atomic_write_crash_leaves_no_partial_file(tmp_path):
    final_path = tmp_path / 'output.parquet'
    tmp_file = final_path.with_name(f'{final_path.name}.tmp')

    def write_fn(path):
        path.touch()
        raise OSError('simulated crash')

    with pytest.raises(PersistenceError, match='Atomic write failed'):
        _atomic_write(final_path, write_fn)

    assert not final_path.exists()
    assert not tmp_file.exists()


def test_atomic_write_falls_back_on_exdev(tmp_path):
    final_path = tmp_path / 'output.parquet'

    def write_fn(path):
        path.touch()

    def make_exdev(src, dst):
        ex = OSError()
        ex.errno = errno.EXDEV
        raise ex

    with (
        patch(
            'learnm8.core.persistence.os.replace', side_effect=make_exdev
        ) as mock_replace,
        patch('learnm8.core.persistence.shutil.copyfile') as mock_copyfile,
        pytest.warns(LearnM8Warning, match='Atomic rename failed'),
    ):
        _atomic_write(final_path, write_fn)

    mock_replace.assert_called_once()
    mock_copyfile.assert_called_once_with(tmp_path / 'output.parquet.tmp', final_path)


def test_atomic_write_calls_fsync_on_file_and_directory(tmp_path):
    final_path = tmp_path / 'output.parquet'

    def write_fn(path):
        path.touch()

    with patch('learnm8.core.persistence.os.fsync') as mock_fsync:
        _atomic_write(final_path, write_fn)

    assert mock_fsync.call_count == 2


def test_atomic_write_orphan_tmp_overwritten(tmp_path, caplog):
    final_path = tmp_path / 'output.parquet'
    tmp_file = final_path.with_name(f'{final_path.name}.tmp')
    tmp_file.touch()

    def write_fn(path):
        path.touch()

    with caplog.at_level(logging.INFO, logger='learnm8.core.persistence'):
        _atomic_write(final_path, write_fn)

    assert any('Overwriting orphan .tmp file' in rec.message for rec in caplog.records)
    assert final_path.exists()


def test_atomic_write_unique_tmp_paths_per_invocation(tmp_path):
    paths_seen = set()

    def write_fn(path):
        paths_seen.add(str(path))
        path.touch()

    _atomic_write(tmp_path / 'a.parquet', write_fn)
    _atomic_write(tmp_path / 'b.parquet', write_fn)

    assert len(paths_seen) == 2
    assert any('a.parquet.tmp' in p for p in paths_seen)
    assert any('b.parquet.tmp' in p for p in paths_seen)


def test_selection_history_uses_prebuilt_lookup(tmp_path):
    """Verify selection history builds a lookup dict with a single compounds
    scan instead of per-row filtering inside the cycle loop."""
    n_compounds = 30
    n_cycles = 5
    n_per_cycle = n_compounds // n_cycles

    compounds_df = pl.DataFrame(
        {
            'ID': [f'C{i:03d}' for i in range(n_compounds)],
            'SMILES': [f'CC{"C" * (i % 5)}' for i in range(n_compounds)],
            'status': ['labeled'] * n_compounds,
            'labeled_cycle': list(range(n_compounds)),
            'selected_cycle': [c for c in range(n_cycles) for _ in range(n_per_cycle)],
            'pruned_cycle': [-1] * n_compounds,
            'Activity': [float(i) for i in range(n_compounds)],
        }
    )

    cycle_metrics = []
    for c in range(n_cycles):
        preds = pl.DataFrame(
            {
                'ID': [
                    f'C{i:03d}' for i in range(c * n_per_cycle, (c + 1) * n_per_cycle)
                ],
                'prediction': [float(i) * 0.5 for i in range(n_per_cycle)],
                'uncertainty': [0.1] * n_per_cycle,
            }
        )
        cycle_metrics.append(
            {
                'cycle': c,
                'strategy': 'greedy',
                'batch_size': n_per_cycle,
                'selected_predictions': preds,
            }
        )

    validation_result = ValidationResult(
        valid_compounds=compounds_df,
        invalid_compounds=pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8}),
        validation_errors={},
    )

    config = {'target_col': 'Activity'}
    output_dir = tmp_path / 'output'

    original_filter = pl.DataFrame.filter
    call_count = [0]

    def counting_filter(self, predicate):
        call_count[0] += 1
        return original_filter(self, predicate)

    with patch.object(pl.DataFrame, 'filter', counting_filter):
        save_results(
            compounds_df,
            cycle_metrics,
            validation_result,
            config,
            output_dir,
        )

    assert call_count[0] < n_cycles


def test_selection_history_matches_legacy_output(tmp_path):
    """Verify the optimized selection history produces the same rows."""
    compounds_df = pl.DataFrame(
        {
            'ID': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'SMILES': ['CC', 'CCO', 'CCN', 'CCC', 'CCOO'],
            'status': ['labeled', 'labeled', 'labeled', 'unlabeled', 'unlabeled'],
            'labeled_cycle': [0, 0, 0, -1, -1],
            'selected_cycle': [0, 0, 0, 1, 1],
            'pruned_cycle': [-1, -1, -1, -1, -1],
            'Activity': [5.0, 3.0, 7.0, None, None],
        }
    )

    selected_preds_cycle1 = pl.DataFrame(
        {
            'ID': ['C004', 'C005'],
            'prediction': [8.0, 2.5],
            'uncertainty': [0.1, 0.2],
        }
    )

    cycle_metrics = [
        {
            'cycle': 0,
            'strategy': 'random',
            'batch_size': 3,
            'selected_predictions': pl.DataFrame(
                schema={
                    'ID': pl.Utf8,
                    'prediction': pl.Float64,
                    'uncertainty': pl.Float64,
                }
            ).slice(0, 0),
        },
        {
            'cycle': 1,
            'strategy': 'greedy',
            'batch_size': 2,
            'selected_predictions': selected_preds_cycle1,
        },
    ]

    validation_result = ValidationResult(
        valid_compounds=compounds_df,
        invalid_compounds=pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8}),
        validation_errors={},
    )

    config = {'target_col': 'Activity'}
    output_dir = tmp_path / 'output'

    saved_files = save_results(
        compounds_df,
        cycle_metrics,
        validation_result,
        config,
        output_dir,
    )

    selection_path = saved_files['selection_history']
    assert selection_path.exists()

    selections = pl.read_csv(selection_path, comment_prefix='#')
    assert len(selections) == 2

    row0 = selections.row(0, named=True)
    assert row0['cycle'] == 1
    assert row0['strategy'] == 'greedy'
    assert row0['ID'] == 'C004'
    assert row0['SMILES'] == 'CCC'
    assert row0['prediction_at_selection'] == 8.0
    assert row0['uncertainty_at_selection'] == 0.1

    row1 = selections.row(1, named=True)
    assert row1['cycle'] == 1
    assert row1['ID'] == 'C005'
    assert row1['SMILES'] == 'CCOO'
    assert row1['prediction_at_selection'] == 2.5
    assert row1['uncertainty_at_selection'] == 0.2

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest

from learnm8.core.persistence import (
    _add_csv_metadata,
    _organize_columns,
    prediction_parquet_path,
    save_results,
    write_cycle_predictions,
)
from learnm8.core.validation import ValidationResult

pytestmark = pytest.mark.unit


def test_organize_columns():
    df = pl.DataFrame({
        'ID': [1, 2],
        'SMILES': ['CCO', 'CCC'],
        'status': ['labeled', 'unlabeled'],
        'prediction_cycle_0': [0.5, 0.6],
        'prediction_cycle_1': [0.7, 0.8],
        'uncertainty_cycle_0': [0.1, 0.2],
        'extra_col': ['a', 'b']
    })

    column_groups = [
        ['ID', 'SMILES', 'status'],
        ['prediction_cycle_0', 'prediction_cycle_1'],
        ['uncertainty_cycle_0']
    ]

    result = _organize_columns(df, column_groups)

    expected_order = ['ID', 'SMILES', 'status', 'prediction_cycle_0',
                     'prediction_cycle_1', 'uncertainty_cycle_0', 'extra_col']
    assert list(result.columns) == expected_order


def test_add_csv_metadata(tmp_path):
    csv_file = tmp_path / 'test.csv'
    df = pl.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df.write_csv(csv_file)

    metadata = {
        'Description': 'Test file',
        'Count': 2,
        '': ''
    }

    _add_csv_metadata(csv_file, metadata)

    with open(csv_file, 'r') as f:
        lines = f.readlines()

    assert '# LearnM8 Active Learning Results\n' in lines
    assert '# Description: Test file\n' in lines
    assert '# Count: 2\n' in lines
    assert '#\n' in lines
    assert 'A,B\n' in lines


def test_save_results_basic(tmp_path):
    np.random.seed(42)

    compounds_df = pl.DataFrame({
        'ID': [f'COMP_{i:03d}' for i in range(10)],
        'SMILES': ['CCO'] * 10,
        'status': ['labeled'] * 3 + ['unlabeled'] * 7,
        'labeled_cycle': [-1, -1, -1] + [None] * 7,
        'selected_cycle': [-1, -1, -1] + [None] * 7,
        'pruned_cycle': [None] * 10,
        'Activity': [0.3, 0.6, 0.9] + [None] * 7,
    })

    cycle_metrics = [
        {
            'cycle': 0,
            'strategy': 'random',
            'batch_size': 2,
            'selected_count': 2,
            'remaining_unlabeled': 5,
            'cumulative_labeled': 5,
            'cumulative_pruned': 0,
            'selected_ids': ['COMP_003', 'COMP_004'],
            'pruned_ids': [],
            'prediction_mean': 0.55,
            'prediction_std': 0.13,
            'prediction_min': 0.4,
            'prediction_max': 0.7,
            'prediction_median': 0.55,
            'measured_mean': 0.5,
            'measured_std': 0.1,
            'measured_min': 0.4,
            'measured_max': 0.6,
            'measured_best': 0.6,
            'best_so_far': 0.9
        },
        {
            'cycle': 1,
            'strategy': 'greedy',
            'batch_size': 2,
            'selected_count': 2,
            'remaining_unlabeled': 3,
            'cumulative_labeled': 7,
            'cumulative_pruned': 0,
            'selected_ids': ['COMP_005', 'COMP_006'],
            'pruned_ids': [],
            'prediction_mean': 0.70,
            'prediction_std': 0.07,
            'prediction_min': 0.65,
            'prediction_max': 0.75,
            'prediction_median': 0.70,
            'measured_mean': 0.68,
            'measured_std': 0.05,
            'measured_min': 0.65,
            'measured_max': 0.7,
            'measured_best': 0.7,
            'best_so_far': 0.9
        }
    ]

    validation_result = ValidationResult(
        valid_compounds=compounds_df.select(['ID', 'SMILES']),
        invalid_compounds=pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8}),
        validation_errors={}
    )

    config = {
        'target_col': 'Activity',
        'featurizer': 'morgan',
        'score_direction': 'higher',
        'mode': 'run',
        'n_cycles': 2,
        'random_state': 42
    }

    saved_files = save_results(
        compounds_df=compounds_df,
        cycle_metrics=cycle_metrics,
        validation_result=validation_result,
        config=config,
        output_dir=tmp_path,
        output_format='auto',
    )

    assert 'compounds_final' in saved_files
    assert 'cycle_metrics' in saved_files
    assert 'selection_history' in saved_files
    assert 'config' in saved_files

    assert saved_files['compounds_final'].exists()
    assert saved_files['cycle_metrics'].exists()
    assert saved_files['selection_history'].exists()
    assert saved_files['config'].exists()

    assert saved_files['compounds_final'].suffix == '.csv'
    assert saved_files['cycle_metrics'].suffix == '.csv'
    assert saved_files['selection_history'].suffix == '.csv'


def test_save_results_compounds_final_structure(tmp_path):
    compounds_df = pl.DataFrame({
        'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
        'SMILES': ['CCO', 'CCC', 'CCN'],
        'status': ['labeled', 'labeled', 'unlabeled'],
        'labeled_cycle': [-1, 0, None],
        'selected_cycle': [-1, 0, None],
        'pruned_cycle': [None, None, None],
        'Activity': [0.3, 0.6, None],
    })

    cycle_metrics = [{
        'cycle': 0, 'strategy': 'random', 'batch_size': 1, 'selected_count': 1,
        'remaining_unlabeled': 1, 'cumulative_labeled': 2, 'cumulative_pruned': 0,
        'selected_ids': ['COMP_002'], 'pruned_ids': []
    }]

    validation_result = ValidationResult(
        valid_compounds=compounds_df.select(['ID', 'SMILES']),
        invalid_compounds=pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8}),
        validation_errors={}
    )

    config = {'target_col': 'Activity', 'featurizer': 'morgan', 'n_cycles': 1}

    saved_files = save_results(compounds_df, cycle_metrics, validation_result, config, tmp_path, output_format='auto')

    assert saved_files['compounds_final'].suffix == '.csv'
    final_df = pd.read_csv(saved_files['compounds_final'], comment='#')

    # The slim CSV holds only the narrow base columns + target; predictions
    # live in per-cycle parquet files.
    assert 'ID' in final_df.columns
    assert 'SMILES' in final_df.columns
    assert 'status' in final_df.columns
    assert 'Activity' in final_df.columns

    assert not any(c.startswith('prediction_cycle_') for c in final_df.columns)
    assert not any(c.startswith('uncertainty_cycle_') for c in final_df.columns)


def test_save_results_cycle_metrics_no_list_columns(tmp_path):
    compounds_df = pl.DataFrame({
        'ID': ['COMP_001'],
        'SMILES': ['CCO'],
        'status': ['labeled'],
        'labeled_cycle': [0],
        'selected_cycle': [0],
        'pruned_cycle': [None],
        'Activity': [0.5]
    })

    cycle_metrics = [{
        'cycle': 0,
        'strategy': 'random',
        'batch_size': 1,
        'selected_count': 1,
        'remaining_unlabeled': 0,
        'cumulative_labeled': 1,
        'cumulative_pruned': 0,
        'selected_ids': ['COMP_001'],
        'pruned_ids': []
    }]

    validation_result = ValidationResult(
        valid_compounds=compounds_df.select(['ID', 'SMILES']),
        invalid_compounds=pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8}),
        validation_errors={}
    )

    config = {'target_col': 'Activity', 'featurizer': 'morgan'}

    saved_files = save_results(compounds_df, cycle_metrics, validation_result, config, tmp_path)

    metrics_df = pd.read_csv(saved_files['cycle_metrics'], comment='#')

    assert 'selected_ids' not in metrics_df.columns
    assert 'pruned_ids' not in metrics_df.columns


def test_save_results_selection_history(tmp_path):
    compounds_df = pl.DataFrame({
        'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
        'SMILES': ['CCO', 'CCC', 'CCN'],
        'status': ['labeled', 'labeled', 'labeled'],
        'labeled_cycle': [-1, 0, 1],
        'selected_cycle': [-1, 0, 1],
        'pruned_cycle': [None, None, None],
        'Activity': [0.3, 0.6, 0.9],
    })

    # Cycle-time captures replace the old prediction_cycle_N columns.
    cycle_metrics = [
        {
            'cycle': 0, 'strategy': 'random', 'batch_size': 1,
            'selected_count': 1, 'remaining_unlabeled': 1,
            'cumulative_labeled': 2, 'cumulative_pruned': 0,
            'selected_ids': ['COMP_002'], 'pruned_ids': [],
            'selected_predictions': pl.DataFrame({
                'ID': ['COMP_002'],
                'prediction': [0.5],
                'uncertainty': [0.15],
            }),
        },
        {
            'cycle': 1, 'strategy': 'greedy', 'batch_size': 1,
            'selected_count': 1, 'remaining_unlabeled': 0,
            'cumulative_labeled': 3, 'cumulative_pruned': 0,
            'selected_ids': ['COMP_003'], 'pruned_ids': [],
            'selected_predictions': pl.DataFrame({
                'ID': ['COMP_003'],
                'prediction': [0.85],
                'uncertainty': [0.1],
            }),
        }
    ]

    validation_result = ValidationResult(
        valid_compounds=compounds_df.select(['ID', 'SMILES']),
        invalid_compounds=pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8}),
        validation_errors={}
    )

    config = {'target_col': 'Activity', 'featurizer': 'morgan'}

    saved_files = save_results(compounds_df, cycle_metrics, validation_result, config, tmp_path, output_format='auto')

    assert saved_files['selection_history'].suffix == '.csv'
    history_df = pd.read_csv(saved_files['selection_history'], comment='#')

    assert len(history_df) == 2
    assert list(history_df['ID']) == ['COMP_002', 'COMP_003']
    assert list(history_df['cycle']) == [0, 1]
    assert list(history_df['strategy']) == ['random', 'greedy']
    assert history_df.loc[0, 'prediction_at_selection'] == 0.5
    assert history_df.loc[0, 'uncertainty_at_selection'] == 0.15
    assert history_df.loc[1, 'prediction_at_selection'] == 0.85
    assert history_df.loc[1, 'uncertainty_at_selection'] == 0.1


def test_save_results_validation_report(tmp_path):
    compounds_df = pl.DataFrame({
        'ID': ['COMP_001'],
        'SMILES': ['CCO'],
        'status': ['labeled'],
        'labeled_cycle': [0],
        'selected_cycle': [0],
        'pruned_cycle': [None],
        'Activity': [0.5]
    })

    invalid_compounds = pl.DataFrame({
        'ID': ['INVALID_001', 'INVALID_002'],
        'SMILES': ['INVALID', 'BADSMILES']
    })

    validation_errors = {
        'INVALID_001': 'Invalid SMILES syntax',
        'INVALID_002': 'Feature extraction failed'
    }

    cycle_metrics = [{
        'cycle': 0, 'strategy': 'random', 'batch_size': 1,
        'selected_count': 1, 'remaining_unlabeled': 0,
        'cumulative_labeled': 1, 'cumulative_pruned': 0,
        'selected_ids': ['COMP_001'], 'pruned_ids': []
    }]

    validation_result = ValidationResult(
        valid_compounds=compounds_df.select(['ID', 'SMILES']),
        invalid_compounds=invalid_compounds,
        validation_errors=validation_errors
    )

    config = {'target_col': 'Activity', 'featurizer': 'morgan'}

    saved_files = save_results(compounds_df, cycle_metrics, validation_result, config, tmp_path)

    assert 'validation_report' in saved_files
    assert saved_files['validation_report'].exists()

    invalid_df = pd.read_csv(saved_files['validation_report'], comment='#')

    assert len(invalid_df) == 2
    assert 'ID' in invalid_df.columns
    assert 'SMILES' in invalid_df.columns
    assert 'error' in invalid_df.columns
    assert set(invalid_df['ID']) == {'INVALID_001', 'INVALID_002'}


def test_save_results_config_json(tmp_path):
    compounds_df = pl.DataFrame({
        'ID': ['COMP_001'],
        'SMILES': ['CCO'],
        'status': ['labeled'],
        'labeled_cycle': [0],
        'selected_cycle': [0],
        'pruned_cycle': [None],
        'Activity': [0.5]
    })

    cycle_metrics = [{
        'cycle': 0, 'strategy': 'random', 'batch_size': 1,
        'selected_count': 1, 'remaining_unlabeled': 0,
        'cumulative_labeled': 1, 'cumulative_pruned': 0,
        'selected_ids': ['COMP_001'], 'pruned_ids': []
    }]

    validation_result = ValidationResult(
        valid_compounds=compounds_df.select(['ID', 'SMILES']),
        invalid_compounds=pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8}),
        validation_errors={}
    )

    config = {
        'target_col': 'Activity',
        'featurizer': 'morgan',
        'score_direction': 'higher',
        'mode': 'run',
        'n_cycles': 1,
        'random_state': 42
    }

    saved_files = save_results(compounds_df, cycle_metrics, validation_result, config, tmp_path)

    assert saved_files['config'].exists()

    with open(saved_files['config'], 'r') as f:
        loaded_config = json.load(f)

    assert loaded_config == config


def _make_predictions_df(n: int = 5, with_uncertainty: bool = True) -> pl.DataFrame:
    data = {
        'ID': [f'C{i}' for i in range(n)],
        'prediction': [float(i) * 0.1 for i in range(n)],
    }
    if with_uncertainty:
        data['uncertainty'] = [float(i) * 0.01 for i in range(n)]
    return pl.DataFrame(data)


def test_parquet_filename_convention(tmp_path):
    """prediction_parquet_path() is the single source of truth for naming."""
    p = prediction_parquet_path(tmp_path, 3)
    assert p == tmp_path / 'prediction_cycle_3.parquet'
    assert p.name == 'prediction_cycle_3.parquet'


def test_parquet_schema_matches_spec(tmp_path):
    """Written parquet has schema [ID: Utf8, prediction: Float32, uncertainty: Float32]."""
    df = _make_predictions_df(n=4, with_uncertainty=True)
    written = write_cycle_predictions(df, tmp_path, cycle=1)
    assert written is not None
    actual = pl.read_parquet(written)
    assert actual.columns == ['ID', 'prediction', 'uncertainty']
    assert actual.schema['ID'] == pl.Utf8
    assert actual.schema['prediction'] == pl.Float32
    assert actual.schema['uncertainty'] == pl.Float32


def test_parquet_roundtrip_values(tmp_path):
    """write_cycle_predictions then read returns identical values."""
    from learnm8.core.persistence import _apply_parquet_schema

    df = _make_predictions_df(n=10, with_uncertainty=True)
    path = write_cycle_predictions(df, tmp_path, cycle=2)
    assert path is not None
    actual = pl.read_parquet(path).sort('ID')
    expected = _apply_parquet_schema(df).sort('ID')
    assert actual.equals(expected)


def test_parquet_compression_is_zstd(tmp_path):
    """Parquet files are zstd-compressed."""
    import pyarrow.parquet as pq
    df = _make_predictions_df(n=5)
    path = write_cycle_predictions(df, tmp_path, cycle=0)
    pf = pq.ParquetFile(path)
    # Check compression of the first column chunk in row group 0.
    rg = pf.metadata.row_group(0)
    compressions = {rg.column(i).compression for i in range(rg.num_columns)}
    assert compressions == {'ZSTD'}, f'expected only ZSTD, got {compressions}'


def test_parquet_atomic_write_no_partial_file(tmp_path, monkeypatch):
    """A failed write leaves no .parquet file (only the .tmp is touched, then cleaned)."""
    from learnm8.exceptions import PersistenceError

    df = _make_predictions_df(n=3)

    def boom(self, path, *args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(pl.DataFrame, 'write_parquet', boom)

    final_path = prediction_parquet_path(tmp_path, 4)
    tmp_file = final_path.with_suffix('.parquet.tmp')

    with pytest.raises(PersistenceError, match='Atomic write failed'):
        write_cycle_predictions(df, tmp_path, cycle=4)

    assert not final_path.exists(), 'final parquet should not exist after failed write'
    assert not tmp_file.exists(), '.parquet.tmp should be cleaned up on failure'


def test_write_cycle_predictions_returns_none_when_output_dir_none():
    """output_dir=None short-circuits to no write and returns None."""
    df = _make_predictions_df(n=2)
    result = write_cycle_predictions(df, None, cycle=0)
    assert result is None

import pytest
import polars as pl
import pandas as pd
import numpy as np
import json
from pathlib import Path

from learnm8.core.persistence import save_results, _add_csv_metadata, _organize_columns
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
        'prediction_cycle_0': [None] * 3 + [0.4, 0.5, 0.6, 0.7] + [None] * 3,
        'uncertainty_cycle_0': [None] * 3 + [0.1, 0.15, 0.2, 0.25] + [None] * 3,
        'prediction_cycle_1': [None] * 5 + [0.65, 0.75] + [None] * 3,
        'uncertainty_cycle_1': [None] * 5 + [0.12, 0.18] + [None] * 3
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
        output_dir=tmp_path
    )

    assert 'compounds_final' in saved_files
    assert 'cycle_metrics' in saved_files
    assert 'selection_history' in saved_files
    assert 'config' in saved_files

    assert saved_files['compounds_final'].exists()
    assert saved_files['cycle_metrics'].exists()
    assert saved_files['selection_history'].exists()
    assert saved_files['config'].exists()


def test_save_results_compounds_final_structure(tmp_path):
    compounds_df = pl.DataFrame({
        'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
        'SMILES': ['CCO', 'CCC', 'CCN'],
        'status': ['labeled', 'labeled', 'unlabeled'],
        'labeled_cycle': [-1, 0, None],
        'selected_cycle': [-1, 0, None],
        'pruned_cycle': [None, None, None],
        'Activity': [0.3, 0.6, None],
        'prediction_cycle_0': [None, None, 0.5],
        'uncertainty_cycle_0': [None, None, 0.15]
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

    saved_files = save_results(compounds_df, cycle_metrics, validation_result, config, tmp_path)

    final_df = pd.read_csv(saved_files['compounds_final'], comment='#')

    assert 'ID' in final_df.columns
    assert 'SMILES' in final_df.columns
    assert 'status' in final_df.columns
    assert 'Activity' in final_df.columns
    assert 'prediction_cycle_0' in final_df.columns
    assert 'uncertainty_cycle_0' in final_df.columns

    pred_cols = [c for c in final_df.columns if c.startswith('prediction_cycle_')]
    unc_cols = [c for c in final_df.columns if c.startswith('uncertainty_cycle_')]

    assert pred_cols == sorted(pred_cols, key=lambda x: int(x.split('_')[-1]))
    assert unc_cols == sorted(unc_cols, key=lambda x: int(x.split('_')[-1]))


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
        'prediction_cycle_0': [None, 0.5, 0.7],
        'uncertainty_cycle_0': [None, 0.15, 0.2],
        'prediction_cycle_1': [None, None, 0.85],
        'uncertainty_cycle_1': [None, None, 0.1]
    })

    cycle_metrics = [
        {
            'cycle': 0, 'strategy': 'random', 'batch_size': 1,
            'selected_count': 1, 'remaining_unlabeled': 1,
            'cumulative_labeled': 2, 'cumulative_pruned': 0,
            'selected_ids': ['COMP_002'], 'pruned_ids': []
        },
        {
            'cycle': 1, 'strategy': 'greedy', 'batch_size': 1,
            'selected_count': 1, 'remaining_unlabeled': 0,
            'cumulative_labeled': 3, 'cumulative_pruned': 0,
            'selected_ids': ['COMP_003'], 'pruned_ids': []
        }
    ]

    validation_result = ValidationResult(
        valid_compounds=compounds_df.select(['ID', 'SMILES']),
        invalid_compounds=pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8}),
        validation_errors={}
    )

    config = {'target_col': 'Activity', 'featurizer': 'morgan'}

    saved_files = save_results(compounds_df, cycle_metrics, validation_result, config, tmp_path)

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

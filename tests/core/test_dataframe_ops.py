import pytest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np

from learnm8.core.dataframe_ops import (
    add_predictions,
    update_status,
    get_compounds_by_status,
    batch_update
)
from conftest import create_initialized_master_df as initialize_master_dataframe


@pytest.fixture
def sample_initialized_df():
    compounds = pl.DataFrame({
        'ID': [f'COMP_{i:03d}' for i in range(10)],
        'SMILES': ['CCO'] * 10
    })

    initial_ids = ['COMP_000', 'COMP_001', 'COMP_002']
    initial_values = pl.Series('ID', initial_ids, dtype=pl.Utf8)

    return initialize_master_dataframe(
        valid_compounds=compounds,
        target_col='Activity',
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values
    )


def test_add_predictions_single_cycle(sample_initialized_df):
    df = sample_initialized_df.clone()

    compound_ids = ['COMP_003', 'COMP_004', 'COMP_005']
    predictions = np.array([0.5, 0.6, 0.7])
    uncertainties = np.array([0.1, 0.15, 0.2])

    updated_df = add_predictions(df, cycle=0, compound_ids=compound_ids,
                                 predictions=predictions, uncertainties=uncertainties)

    assert 'prediction_cycle_0' in updated_df.columns
    assert 'uncertainty_cycle_0' in updated_df.columns

    for cid, pred, unc in zip(compound_ids, predictions, uncertainties):
        row = updated_df.filter(pl.col('ID') == cid)
        assert row.get_column('prediction_cycle_0')[0] == pred
        assert row.get_column('uncertainty_cycle_0')[0] == unc

    assert updated_df.filter(pl.col('ID') == 'COMP_006').get_column('prediction_cycle_0')[0] is None


def test_add_predictions_immutability(sample_initialized_df):
    df = sample_initialized_df.clone()
    original_df = df.clone()

    compound_ids = ['COMP_003']
    predictions = np.array([0.5])

    updated_df = add_predictions(df, cycle=0, compound_ids=compound_ids,
                                 predictions=predictions)

    assert_frame_equal(df, original_df)
    assert 'prediction_cycle_0' not in df.columns
    assert 'prediction_cycle_0' in updated_df.columns


def test_add_predictions_multiple_cycles(sample_initialized_df):
    df = sample_initialized_df.clone()

    df = add_predictions(df, cycle=0, compound_ids=['COMP_003', 'COMP_004'],
                        predictions=np.array([0.5, 0.6]))

    df = add_predictions(df, cycle=1, compound_ids=['COMP_003', 'COMP_005'],
                        predictions=np.array([0.55, 0.65]))

    assert df.filter(pl.col('ID') == 'COMP_003').get_column('prediction_cycle_0')[0] == 0.5
    assert df.filter(pl.col('ID') == 'COMP_003').get_column('prediction_cycle_1')[0] == 0.55
    assert df.filter(pl.col('ID') == 'COMP_005').get_column('prediction_cycle_1')[0] == 0.65
    assert df.filter(pl.col('ID') == 'COMP_005').get_column('prediction_cycle_0')[0] is None


def test_add_predictions_without_uncertainties(sample_initialized_df):
    df = sample_initialized_df.clone()

    compound_ids = ['COMP_003', 'COMP_004']
    predictions = np.array([0.5, 0.6])

    updated_df = add_predictions(df, cycle=0, compound_ids=compound_ids,
                                 predictions=predictions, uncertainties=None)

    assert 'prediction_cycle_0' in updated_df.columns
    assert 'uncertainty_cycle_0' not in updated_df.columns

    assert updated_df.filter(pl.col('ID') == 'COMP_003').get_column('prediction_cycle_0')[0] == 0.5
    assert updated_df.filter(pl.col('ID') == 'COMP_004').get_column('prediction_cycle_0')[0] == 0.6


def test_add_predictions_mismatched_lengths(sample_initialized_df):
    df = sample_initialized_df.clone()

    compound_ids = ['COMP_003', 'COMP_004']
    predictions = np.array([0.5, 0.6, 0.7])

    with pytest.raises(ValueError, match="compound_ids and predictions must have the same length"):
        add_predictions(df, cycle=0, compound_ids=compound_ids, predictions=predictions)


def test_add_predictions_duplicate_ids(sample_initialized_df):
    df = sample_initialized_df.clone()

    compound_ids = ['COMP_003', 'COMP_003']
    predictions = np.array([0.5, 0.6])

    with pytest.raises(ValueError, match="compound_ids contains duplicates"):
        add_predictions(df, cycle=0, compound_ids=compound_ids, predictions=predictions)


def test_update_status_to_labeled(sample_initialized_df):
    df = sample_initialized_df.clone()

    compound_ids = ['COMP_003', 'COMP_004']
    target_values = pl.Series('ID', compound_ids, dtype=pl.Utf8)

    updated_df = update_status(
        df, compound_ids=compound_ids, new_status='labeled',
        cycle=0, target_col='Activity', target_values=target_values
    )

    for cid in compound_ids:
        row = updated_df.filter(pl.col('ID') == cid)
        assert row.get_column('status')[0] == 'labeled'
        assert row.get_column('labeled_cycle')[0] == 0
        assert row.get_column('selected_cycle')[0] == 0


def test_update_status_immutability(sample_initialized_df):
    df = sample_initialized_df.clone()
    original_df = df.clone()

    compound_ids = ['COMP_003']
    target_values = pl.Series('ID', compound_ids, dtype=pl.Utf8)

    updated_df = update_status(
        df, compound_ids=compound_ids, new_status='labeled',
        cycle=0, target_col='Activity', target_values=target_values
    )

    assert_frame_equal(df, original_df)
    assert df.filter(pl.col('ID') == 'COMP_003').get_column('status')[0] == 'unlabeled'
    assert updated_df.filter(pl.col('ID') == 'COMP_003').get_column('status')[0] == 'labeled'


def test_update_status_to_pruned(sample_initialized_df):
    df = sample_initialized_df.clone()

    compound_ids = ['COMP_003', 'COMP_004']

    updated_df = update_status(
        df, compound_ids=compound_ids, new_status='pruned',
        cycle=1, target_col='Activity', target_values=None
    )

    for cid in compound_ids:
        row = updated_df.filter(pl.col('ID') == cid)
        assert row.get_column('status')[0] == 'pruned'
        assert row.get_column('pruned_cycle')[0] == 1


def test_update_status_preserves_first_selection(sample_initialized_df):
    df = sample_initialized_df.clone()

    target_vals_0 = pl.Series('ID', ['COMP_003'], dtype=pl.Utf8)
    df = update_status(df, ['COMP_003'], 'labeled', cycle=0,
                      target_col='Activity', target_values=target_vals_0)

    assert df.filter(pl.col('ID') == 'COMP_003').get_column('selected_cycle')[0] == 0
    assert df.filter(pl.col('ID') == 'COMP_003').get_column('labeled_cycle')[0] == 0

    target_vals_2 = pl.Series('ID', ['COMP_003'], dtype=pl.Utf8)
    df = update_status(df, ['COMP_003'], 'labeled', cycle=2,
                      target_col='Activity', target_values=target_vals_2)

    assert df.filter(pl.col('ID') == 'COMP_003').get_column('selected_cycle')[0] == 0
    assert df.filter(pl.col('ID') == 'COMP_003').get_column('labeled_cycle')[0] == 2


def test_update_status_invalid_status(sample_initialized_df):
    df = sample_initialized_df.clone()

    with pytest.raises(ValueError, match="new_status must be one of"):
        update_status(df, ['COMP_003'], 'invalid_status', cycle=0,
                     target_col='Activity')


def test_get_compounds_by_status_labeled(sample_initialized_df):
    df = sample_initialized_df.clone()

    labeled = get_compounds_by_status(df, 'labeled')

    assert len(labeled) == 3
    assert set(labeled.get_column('ID').to_list()) == {'COMP_000', 'COMP_001', 'COMP_002'}
    assert all(labeled.get_column('status') == 'labeled')


def test_get_compounds_by_status_unlabeled(sample_initialized_df):
    df = sample_initialized_df.clone()

    unlabeled = get_compounds_by_status(df, 'unlabeled')

    assert len(unlabeled) == 7
    assert all(unlabeled.get_column('status') == 'unlabeled')


def test_get_compounds_by_status_with_columns(sample_initialized_df):
    df = sample_initialized_df.clone()

    labeled = get_compounds_by_status(df, 'labeled', columns=['ID', 'SMILES', 'Activity'])

    assert set(labeled.columns) == {'ID', 'SMILES', 'Activity'}
    assert len(labeled) == 3


def test_get_compounds_by_status_invalid_status(sample_initialized_df):
    df = sample_initialized_df.clone()

    with pytest.raises(ValueError, match="status must be one of"):
        get_compounds_by_status(df, 'invalid_status')


def test_batch_update_combined_operations(sample_initialized_df):
    df = sample_initialized_df.clone()

    updates = {
        'predictions': (0, ['COMP_003', 'COMP_004'],
                       np.array([0.5, 0.6]), np.array([0.1, 0.15])),
        'status': (['COMP_003'], 'labeled', 0, 'Activity',
                  pl.Series('ID', ['COMP_003'], dtype=pl.Utf8))
    }

    updated_df = batch_update(df, updates)

    assert 'prediction_cycle_0' in updated_df.columns
    assert updated_df.filter(pl.col('ID') == 'COMP_003').get_column('prediction_cycle_0')[0] == 0.5
    assert updated_df.filter(pl.col('ID') == 'COMP_003').get_column('status')[0] == 'labeled'


def test_batch_update_predictions_only(sample_initialized_df):
    df = sample_initialized_df.clone()

    updates = {
        'predictions': (0, ['COMP_003'], np.array([0.5]), None)
    }

    updated_df = batch_update(df, updates)

    assert 'prediction_cycle_0' in updated_df.columns
    assert updated_df.filter(pl.col('ID') == 'COMP_003').get_column('prediction_cycle_0')[0] == 0.5


def test_batch_update_status_only(sample_initialized_df):
    df = sample_initialized_df.clone()

    updates = {
        'status': (['COMP_003'], 'pruned', 0, 'Activity', None)
    }

    updated_df = batch_update(df, updates)

    assert updated_df.filter(pl.col('ID') == 'COMP_003').get_column('status')[0] == 'pruned'


def test_batch_update_immutability(sample_initialized_df):
    df = sample_initialized_df.clone()
    original_df = df.clone()

    updates = {
        'predictions': (0, ['COMP_003'], np.array([0.5]), None)
    }

    updated_df = batch_update(df, updates)

    assert_frame_equal(df, original_df)
    assert 'prediction_cycle_0' not in df.columns
    assert 'prediction_cycle_0' in updated_df.columns

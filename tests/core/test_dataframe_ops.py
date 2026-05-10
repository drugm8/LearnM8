import pandas as pd
import polars as pl
import pytest

from learnm8.core.dataframe_ops import (
    get_compounds_by_status,
    update_status,
)
from learnm8.core.initialization import (
    STATUS_LABELED,
    STATUS_PRUNED,
    STATUS_UNLABELED,
    validate_master_dataframe,
)
from tests.fixtures.master_dataframe import create_initialized_master_df

pytestmark = pytest.mark.unit


def test_initialize_master_dataframe(sample_compounds):
    """Test initialization with valid compound pool."""
    all_ids = sample_compounds.get_column('ID').to_list()
    initial_ids = all_ids[:2]
    initial_values = pd.Series([0.5, 0.7], index=initial_ids)

    master_df = create_initialized_master_df(
        valid_compounds=sample_compounds,
        target_col='Activity',
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values,
    )

    # Verify schema (new version uses actual target column name, not 'target_value')
    assert all(
        col in master_df.columns
        for col in [
            'ID',
            'SMILES',
            'status',
            'labeled_cycle',
            'selected_cycle',
            'pruned_cycle',
            'Activity',
        ]
    )

    # Verify status values
    assert (
        master_df.filter(pl.col('ID').is_in(initial_ids))['status'] == STATUS_LABELED
    ).all()
    assert (
        master_df.filter(~pl.col('ID').is_in(initial_ids))['status'] == STATUS_UNLABELED
    ).all()

    # Verify target values set for initial compounds
    for comp_id in initial_ids:
        target_val = master_df.filter(pl.col('ID') == comp_id)['Activity'][0]
        assert target_val == initial_values[comp_id]


def test_initialize_master_dataframe_empty_initial(sample_compounds):
    """Test initialization with no initial labeled compounds."""
    master_df = create_initialized_master_df(
        valid_compounds=sample_compounds,
        target_col='Activity',
        initial_labeled_ids=[],
        initial_measurements=pd.Series(dtype='float64'),
    )

    # All compounds should be unlabeled
    assert (master_df['status'] == STATUS_UNLABELED).all()

    # No target values set (uses actual column name 'Activity', not 'target_value')
    assert master_df['Activity'].is_null().all()


def test_get_labeled_compounds(sample_master_df):
    """Test labeled compound extraction."""
    labeled = get_compounds_by_status(
        sample_master_df, 'labeled', columns=['ID', 'SMILES', 'Activity']
    )
    labeled_ids = (
        sample_master_df.filter(pl.col('status') == STATUS_LABELED)
        .get_column('ID')
        .to_list()
    )

    # Verify only labeled compounds returned
    assert len(labeled) == 2
    assert all(comp_id in labeled_ids for comp_id in labeled['ID'])

    # Verify Activity column present (renamed from target_value)
    assert 'Activity' in labeled.columns
    assert 'target_value' not in labeled.columns

    # Verify ID and SMILES columns present
    assert 'ID' in labeled.columns
    assert 'SMILES' in labeled.columns


def test_get_unlabeled_compounds(sample_master_df):
    """Test unlabeled compound extraction."""
    unlabeled = get_compounds_by_status(
        sample_master_df, 'unlabeled', columns=['ID', 'SMILES']
    )
    labeled_ids = (
        sample_master_df.filter(pl.col('status') == STATUS_LABELED)
        .get_column('ID')
        .to_list()
    )

    # Verify only unlabeled compounds returned
    assert len(unlabeled) == 3
    assert all(comp_id not in labeled_ids for comp_id in unlabeled['ID'])

    # Verify ID and SMILES columns present
    assert 'ID' in unlabeled.columns
    assert 'SMILES' in unlabeled.columns


def test_update_compound_status_to_labeled(sample_master_df):
    """Test status update to labeled."""
    original_df = sample_master_df.clone()
    unlabeled_ids = (
        sample_master_df.filter(pl.col('status') == STATUS_UNLABELED)
        .get_column('ID')
        .to_list()
    )
    compound_ids = unlabeled_ids[:2]
    target_values = pd.Series([0.8, 0.9], index=compound_ids)

    updated_df = update_status(
        df=sample_master_df,
        compound_ids=compound_ids,
        new_status=STATUS_LABELED,
        cycle=0,
        target_col='Activity',
        target_values=target_values,
    )

    # Verify status changed
    for comp_id in compound_ids:
        assert updated_df.filter(pl.col('ID') == comp_id)['status'][0] == STATUS_LABELED

    # Verify labeled_cycle set
    for comp_id in compound_ids:
        assert updated_df.filter(pl.col('ID') == comp_id)['labeled_cycle'][0] == 0

    # Verify Activity column set (not 'target_value')
    for comp_id in compound_ids:
        assert (
            updated_df.filter(pl.col('ID') == comp_id)['Activity'][0]
            == target_values[comp_id]
        )

    # Verify original DataFrame unchanged (immutability)
    # Compare non-categorical columns separately due to dtype differences
    for col in original_df.columns:
        if col == 'status':
            # Categorical column comparison
            assert (
                original_df[col].cast(pl.Utf8) == sample_master_df[col].cast(pl.Utf8)
            ).all()
        else:
            assert original_df[col].equals(sample_master_df[col])


def test_update_compound_status_to_pruned(sample_master_df):
    """Test status update to pruned."""
    unlabeled_ids = (
        sample_master_df.filter(pl.col('status') == STATUS_UNLABELED)
        .get_column('ID')
        .to_list()
    )
    compound_ids = unlabeled_ids[1:3]

    updated_df = update_status(
        df=sample_master_df,
        compound_ids=compound_ids,
        new_status=STATUS_PRUNED,
        cycle=1,
        target_col='Activity',
    )

    # Verify status changed
    for comp_id in compound_ids:
        assert updated_df.filter(pl.col('ID') == comp_id)['status'][0] == STATUS_PRUNED

    # Verify pruned_cycle set
    for comp_id in compound_ids:
        assert updated_df.filter(pl.col('ID') == comp_id)['pruned_cycle'][0] == 1


def test_update_compound_status_invalid_status(sample_master_df):
    """Test validation with invalid status."""
    any_id = sample_master_df.get_column('ID').to_list()[:1]
    with pytest.raises(ValueError, match='Must be one of'):
        update_status(
            df=sample_master_df,
            compound_ids=any_id,
            new_status='invalid',
            cycle=0,
            target_col='Activity',
        )


# NOTE: tests for add_predictions / get_prediction_columns were removed
# alongside the functions themselves. The master DataFrame is now narrow
# (no per-cycle prediction columns); per-cycle predictions are exercised by
# the parquet fixtures in tests/fixtures/master_dataframe.py and the cycle
# integration tests in tests/core/test_cycle*.


def test_validate_master_dataframe_valid(sample_master_df):
    """Test validation with valid DataFrame."""
    assert validate_master_dataframe(sample_master_df)


def test_validate_master_dataframe_missing_columns():
    """Test validation with missing columns."""
    invalid_df = pd.DataFrame({'ID': ['C1'], 'SMILES': ['CCO']})

    with pytest.raises(ValueError, match='missing required columns'):
        validate_master_dataframe(invalid_df)


def test_validate_master_dataframe_invalid_status(sample_master_df):
    """Test validation with invalid status values."""
    invalid_df = sample_master_df.clone()
    # Set invalid status by converting to object dtype first and modifying first row
    invalid_df = invalid_df.with_columns(
        pl.Series('status', invalid_df['status'].cast(pl.Utf8))
    )
    # Update first row to have invalid status
    invalid_df = (
        invalid_df.with_row_index('__row_idx')
        .with_columns(
            pl.when(pl.col('__row_idx') == 0)
            .then(pl.lit('invalid_status'))
            .otherwise(pl.col('status'))
            .alias('status')
        )
        .drop('__row_idx')
    )

    with pytest.raises(ValueError, match='Invalid status values found'):
        validate_master_dataframe(invalid_df)


def test_immutability(sample_master_df):
    """Test that all functions return new DataFrames."""
    original_df = sample_master_df.clone()
    unlabeled_ids = (
        sample_master_df.filter(pl.col('status') == STATUS_UNLABELED)
        .get_column('ID')
        .to_list()
    )
    target_id = unlabeled_ids[0]

    # Call update_status — verifies that immutable update pattern is preserved
    # without mutating the input DataFrame.
    _ = update_status(
        df=sample_master_df,
        compound_ids=[target_id],
        new_status=STATUS_LABELED,
        cycle=0,
        target_col='Activity',
        target_values=pd.Series([0.5], index=[target_id]),
    )

    # Verify original DataFrame unchanged (compare column by column due to categorical dtype)
    for col in original_df.columns:
        if col == 'status':
            # Categorical column comparison
            assert (
                original_df[col].cast(pl.Utf8) == sample_master_df[col].cast(pl.Utf8)
            ).all()
        else:
            assert original_df[col].equals(sample_master_df[col])


def test_large_dataset():
    """Test with large dataset (10K compounds)."""
    large_pool = pd.DataFrame(
        {
            'ID': [f'COMP_{i:06d}' for i in range(10000)],
            'SMILES': [f'C{i}' for i in range(10000)],
        }
    )

    initial_ids = [f'COMP_{i:06d}' for i in range(100)]
    initial_values = pd.Series([float(i) * 0.01 for i in range(100)], index=initial_ids)

    # Test initialization
    master_df = create_initialized_master_df(
        valid_compounds=large_pool,
        target_col='Activity',
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values,
    )

    # Verify correct size
    assert len(master_df) == 10000

    # Verify performance acceptable (test should complete quickly)
    labeled = get_compounds_by_status(
        master_df, 'labeled', columns=['ID', 'SMILES', 'Activity']
    )
    assert len(labeled) == 100

    unlabeled = get_compounds_by_status(
        master_df, 'unlabeled', columns=['ID', 'SMILES']
    )
    assert len(unlabeled) == 9900

import pytest
import pandas as pd
import numpy as np
from learnm8.core.initialization import initialize_master_dataframe
from learnm8.core.dataframe_ops import (
    add_predictions,
    update_status,
    get_compounds_by_status
)
from learnm8.core.data_structures import (
    get_prediction_columns,
    validate_master_dataframe,
    STATUS_LABELED,
    STATUS_UNLABELED,
    STATUS_PRUNED
)


@pytest.fixture
def sample_compound_pool():
    """Create sample compound pool for testing."""
    return pd.DataFrame({
        'ID': [f'COMP_{i:03d}' for i in range(100)],
        'SMILES': [f'C{i}' for i in range(100)]
    })


@pytest.fixture
def sample_master_df(sample_compound_pool):
    """Create sample master DataFrame."""
    return initialize_master_dataframe(
        valid_compounds=sample_compound_pool,
        initial_labeled_ids=['COMP_000', 'COMP_001', 'COMP_002'],
        initial_measurements=pd.Series([0.5, 0.7, 0.9], index=['COMP_000', 'COMP_001', 'COMP_002']),
        target_col='Activity'
    )


def test_initialize_master_dataframe(sample_compound_pool):
    """Test initialization with valid compound pool."""
    initial_ids = ['COMP_000', 'COMP_001']
    initial_values = pd.Series([0.5, 0.7], index=['COMP_000', 'COMP_001'])

    master_df = initialize_master_dataframe(
        valid_compounds=sample_compound_pool,
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values,
        target_col='Activity'
    )

    # Verify schema (new version uses actual target column name, not 'target_value')
    assert all(col in master_df.columns for col in ['ID', 'SMILES', 'status', 'labeled_cycle',
                                                      'selected_cycle', 'pruned_cycle', 'Activity'])

    # Verify status values
    assert (master_df[master_df['ID'].isin(initial_ids)]['status'] == STATUS_LABELED).all()
    assert (master_df[~master_df['ID'].isin(initial_ids)]['status'] == STATUS_UNLABELED).all()

    # Verify target values set for initial compounds
    for comp_id in initial_ids:
        target_val = master_df[master_df['ID'] == comp_id]['Activity'].iloc[0]
        assert target_val == initial_values[comp_id]


def test_initialize_master_dataframe_empty_initial(sample_compound_pool):
    """Test initialization with no initial labeled compounds."""
    master_df = initialize_master_dataframe(
        valid_compounds=sample_compound_pool,
        initial_labeled_ids=[],
        initial_measurements=pd.Series(dtype='float64'),
        target_col='Activity'
    )

    # All compounds should be unlabeled
    assert (master_df['status'] == STATUS_UNLABELED).all()

    # No target values set (uses actual column name 'Activity', not 'target_value')
    assert master_df['Activity'].isna().all()


def test_get_labeled_compounds(sample_master_df):
    """Test labeled compound extraction."""
    labeled = get_compounds_by_status(sample_master_df, 'labeled', columns=['ID', 'SMILES', 'Activity'])

    # Verify only labeled compounds returned
    assert len(labeled) == 3
    assert all(comp_id in ['COMP_000', 'COMP_001', 'COMP_002'] for comp_id in labeled['ID'])

    # Verify Activity column present (renamed from target_value)
    assert 'Activity' in labeled.columns
    assert 'target_value' not in labeled.columns

    # Verify ID and SMILES columns present
    assert 'ID' in labeled.columns
    assert 'SMILES' in labeled.columns


def test_get_unlabeled_compounds(sample_master_df):
    """Test unlabeled compound extraction."""
    unlabeled = get_compounds_by_status(sample_master_df, 'unlabeled', columns=['ID', 'SMILES'])

    # Verify only unlabeled compounds returned
    assert len(unlabeled) == 97
    assert all(comp_id not in ['COMP_000', 'COMP_001', 'COMP_002'] for comp_id in unlabeled['ID'])

    # Verify ID and SMILES columns present
    assert 'ID' in unlabeled.columns
    assert 'SMILES' in unlabeled.columns


def test_update_compound_status_to_labeled(sample_master_df):
    """Test status update to labeled."""
    original_df = sample_master_df.copy()
    compound_ids = ['COMP_010', 'COMP_011']
    target_values = pd.Series([0.8, 0.9], index=compound_ids)

    updated_df = update_status(
        df=sample_master_df,
        compound_ids=compound_ids,
        new_status=STATUS_LABELED,
        cycle=0,
        target_col='Activity',
        target_values=target_values
    )

    # Verify status changed
    for comp_id in compound_ids:
        assert updated_df[updated_df['ID'] == comp_id]['status'].iloc[0] == STATUS_LABELED

    # Verify labeled_cycle set
    for comp_id in compound_ids:
        assert updated_df[updated_df['ID'] == comp_id]['labeled_cycle'].iloc[0] == 0

    # Verify Activity column set (not 'target_value')
    for comp_id in compound_ids:
        assert updated_df[updated_df['ID'] == comp_id]['Activity'].iloc[0] == target_values[comp_id]

    # Verify original DataFrame unchanged (immutability)
    # Compare non-categorical columns separately due to dtype differences
    for col in original_df.columns:
        if col == 'status':
            # Categorical column comparison
            assert (original_df[col].astype(str) == sample_master_df[col].astype(str)).all()
        else:
            assert original_df[col].equals(sample_master_df[col])


def test_update_compound_status_to_pruned(sample_master_df):
    """Test status update to pruned."""
    compound_ids = ['COMP_020', 'COMP_021']

    updated_df = update_status(
        df=sample_master_df,
        compound_ids=compound_ids,
        new_status=STATUS_PRUNED,
        cycle=1,
        target_col='Activity'
    )

    # Verify status changed
    for comp_id in compound_ids:
        assert updated_df[updated_df['ID'] == comp_id]['status'].iloc[0] == STATUS_PRUNED

    # Verify pruned_cycle set
    for comp_id in compound_ids:
        assert updated_df[updated_df['ID'] == comp_id]['pruned_cycle'].iloc[0] == 1


def test_update_compound_status_invalid_status(sample_master_df):
    """Test validation with invalid status."""
    with pytest.raises(ValueError, match="new_status must be one of"):
        update_status(
            df=sample_master_df,
            compound_ids=['COMP_030'],
            new_status='invalid',
            cycle=0,
            target_col='Activity'
        )


def test_add_predictions_to_master(sample_master_df):
    """Test adding predictions."""
    compound_ids = ['COMP_010', 'COMP_011', 'COMP_012']
    predictions = np.array([0.6, 0.7, 0.8])

    updated_df = add_predictions(
        df=sample_master_df,
        cycle=0,
        compound_ids=compound_ids,
        predictions=predictions
    )

    # Verify prediction_cycle_0 column created
    assert 'prediction_cycle_0' in updated_df.columns

    # Verify values set correctly
    for i, comp_id in enumerate(compound_ids):
        pred_val = updated_df[updated_df['ID'] == comp_id]['prediction_cycle_0'].iloc[0]
        assert pred_val == predictions[i]

    # Verify NaN for compounds not predicted
    other_compound = updated_df[updated_df['ID'] == 'COMP_050']['prediction_cycle_0'].iloc[0]
    assert pd.isna(other_compound)


def test_add_predictions_with_uncertainties(sample_master_df):
    """Test adding predictions and uncertainties."""
    compound_ids = ['COMP_010', 'COMP_011']
    predictions = np.array([0.6, 0.7])
    uncertainties = np.array([0.1, 0.15])

    updated_df = add_predictions(
        df=sample_master_df,
        cycle=0,
        compound_ids=compound_ids,
        predictions=predictions,
        uncertainties=uncertainties
    )

    # Verify both columns created
    assert 'prediction_cycle_0' in updated_df.columns
    assert 'uncertainty_cycle_0' in updated_df.columns

    # Verify uncertainty values set correctly
    for i, comp_id in enumerate(compound_ids):
        unc_val = updated_df[updated_df['ID'] == comp_id]['uncertainty_cycle_0'].iloc[0]
        assert unc_val == uncertainties[i]


def test_add_predictions_multiple_cycles(sample_master_df):
    """Test adding predictions across multiple cycles."""
    # Add predictions for cycle 0
    updated_df = add_predictions(
        df=sample_master_df,
        cycle=0,
        compound_ids=['COMP_010'],
        predictions=np.array([0.5])
    )

    # Add predictions for cycle 1
    updated_df = add_predictions(
        df=updated_df,
        cycle=1,
        compound_ids=['COMP_010'],
        predictions=np.array([0.6])
    )

    # Add predictions for cycle 2
    updated_df = add_predictions(
        df=updated_df,
        cycle=2,
        compound_ids=['COMP_010'],
        predictions=np.array([0.7])
    )

    # Verify all columns created
    assert 'prediction_cycle_0' in updated_df.columns
    assert 'prediction_cycle_1' in updated_df.columns
    assert 'prediction_cycle_2' in updated_df.columns

    # Verify prediction values for all cycles
    assert updated_df[updated_df['ID'] == 'COMP_010']['prediction_cycle_0'].iloc[0] == 0.5
    assert updated_df[updated_df['ID'] == 'COMP_010']['prediction_cycle_1'].iloc[0] == 0.6
    assert updated_df[updated_df['ID'] == 'COMP_010']['prediction_cycle_2'].iloc[0] == 0.7


def test_get_prediction_columns(sample_master_df):
    """Test prediction column extraction."""
    # Add predictions for multiple cycles
    updated_df = sample_master_df.copy()
    for cycle in [0, 1, 2]:
        updated_df = add_predictions(
            df=updated_df,
            cycle=cycle,
            compound_ids=['COMP_010'],
            predictions=np.array([0.5]),
            uncertainties=np.array([0.1])
        )

    pred_cols, unc_cols = get_prediction_columns(updated_df)

    # Verify correct columns returned
    assert pred_cols == ['prediction_cycle_0', 'prediction_cycle_1', 'prediction_cycle_2']
    assert unc_cols == ['uncertainty_cycle_0', 'uncertainty_cycle_1', 'uncertainty_cycle_2']


def test_get_prediction_columns_no_predictions(sample_master_df):
    """Test with no predictions."""
    pred_cols, unc_cols = get_prediction_columns(sample_master_df)

    # Verify empty lists returned
    assert pred_cols == []
    assert unc_cols == []


def test_validate_master_dataframe_valid(sample_master_df):
    """Test validation with valid DataFrame."""
    assert validate_master_dataframe(sample_master_df) == True


def test_validate_master_dataframe_missing_columns():
    """Test validation with missing columns."""
    invalid_df = pd.DataFrame({'ID': ['C1'], 'SMILES': ['CCO']})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_master_dataframe(invalid_df)


def test_validate_master_dataframe_invalid_status(sample_master_df):
    """Test validation with invalid status values."""
    invalid_df = sample_master_df.copy()
    # Set invalid status by converting to object dtype first
    invalid_df['status'] = invalid_df['status'].astype('object')
    invalid_df.loc[0, 'status'] = 'invalid_status'

    with pytest.raises(ValueError, match="Invalid status values found"):
        validate_master_dataframe(invalid_df)


def test_immutability(sample_master_df):
    """Test that all functions return new DataFrames."""
    original_df = sample_master_df.copy()

    # Call update functions
    _ = update_status(
        df=sample_master_df,
        compound_ids=['COMP_010'],
        new_status=STATUS_LABELED,
        cycle=0,
        target_col='Activity',
        target_values=pd.Series([0.5], index=['COMP_010'])
    )

    _ = add_predictions(
        df=sample_master_df,
        cycle=0,
        compound_ids=['COMP_010'],
        predictions=np.array([0.5])
    )

    # Verify original DataFrame unchanged (compare column by column due to categorical dtype)
    for col in original_df.columns:
        if col == 'status':
            # Categorical column comparison
            assert (original_df[col].astype(str) == sample_master_df[col].astype(str)).all()
        else:
            assert original_df[col].equals(sample_master_df[col])


def test_large_dataset():
    """Test with large dataset (10K compounds)."""
    large_pool = pd.DataFrame({
        'ID': [f'COMP_{i:06d}' for i in range(10000)],
        'SMILES': [f'C{i}' for i in range(10000)]
    })

    initial_ids = [f'COMP_{i:06d}' for i in range(100)]
    initial_values = pd.Series([float(i) * 0.01 for i in range(100)], index=initial_ids)

    # Test initialization
    master_df = initialize_master_dataframe(
        valid_compounds=large_pool,
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values,
        target_col='Activity'
    )

    # Verify correct size
    assert len(master_df) == 10000

    # Verify performance acceptable (test should complete quickly)
    labeled = get_compounds_by_status(master_df, 'labeled', columns=['ID', 'SMILES', 'Activity'])
    assert len(labeled) == 100

    unlabeled = get_compounds_by_status(master_df, 'unlabeled', columns=['ID', 'SMILES'])
    assert len(unlabeled) == 9900

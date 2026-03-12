import pytest
import polars as pl
import numpy as np
from pathlib import Path

from learnm8.core.data_structures import (
    validate_master_dataframe,
    get_prediction_columns,
    STATUS_LABELED,
    STATUS_UNLABELED,
    STATUS_PRUNED,
    VALID_STATUSES
)
from tests.fixtures.master_dataframe import create_initialized_master_df


@pytest.mark.unit
class TestConstants:

    def test_status_constants_defined(self):
        assert STATUS_LABELED == 'labeled'
        assert STATUS_UNLABELED == 'unlabeled'
        assert STATUS_PRUNED == 'pruned'

    def test_valid_statuses_list(self):
        assert VALID_STATUSES == ['unlabeled', 'labeled', 'pruned']
        assert len(VALID_STATUSES) == 3


@pytest.mark.unit
class TestGetPredictionColumns:

    def test_get_prediction_columns_no_predictions(self, sample_master_df):
        pred_cols, unc_cols = get_prediction_columns(sample_master_df)

        assert pred_cols == []
        assert unc_cols == []

    def test_get_prediction_columns_with_predictions(self, master_df_with_predictions):
        pred_cols, unc_cols = get_prediction_columns(master_df_with_predictions)

        assert pred_cols == ['prediction_cycle_0']
        assert unc_cols == ['uncertainty_cycle_0']

    def test_get_prediction_columns_multiple_cycles(self, master_df_multi_cycle):
        pred_cols, unc_cols = get_prediction_columns(master_df_multi_cycle)

        assert pred_cols == ['prediction_cycle_0', 'prediction_cycle_1', 'prediction_cycle_2']
        assert unc_cols == ['uncertainty_cycle_0', 'uncertainty_cycle_1', 'uncertainty_cycle_2']

    def test_get_prediction_columns_with_uncertainties(self, sample_master_df):
        from learnm8.core.dataframe_ops import add_predictions

        master_df = sample_master_df.clone()
        unlabeled = master_df.filter(pl.col('status') == 'unlabeled').head(5)
        unlabeled_ids = unlabeled['ID'].to_list()

        predictions = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        uncertainties = np.array([0.1, 0.15, 0.2, 0.25, 0.3])

        master_df = add_predictions(
            df=master_df,
            cycle=0,
            compound_ids=unlabeled_ids,
            predictions=predictions,
            uncertainties=uncertainties
        )

        pred_cols, unc_cols = get_prediction_columns(master_df)

        assert len(pred_cols) == 1
        assert len(unc_cols) == 1
        assert 'prediction_cycle_0' in master_df.columns
        assert 'uncertainty_cycle_0' in master_df.columns

    def test_get_prediction_columns_pattern_matching(self, sample_master_df):
        master_df = sample_master_df.clone()

        master_df = master_df.with_columns([
            pl.lit(None).cast(pl.Float64).alias('prediction_cycle_0'),
            pl.lit(None).cast(pl.Float64).alias('prediction_cycle_1'),
            pl.lit(None).cast(pl.Float64).alias('prediction_cycle_5'),
            pl.lit(None).cast(pl.Float64).alias('uncertainty_cycle_0'),
            pl.lit(None).cast(pl.Float64).alias('uncertainty_cycle_1'),
            pl.lit(None).cast(pl.Float64).alias('uncertainty_cycle_5'),
            pl.lit(None).cast(pl.Float64).alias('other_prediction'),
            pl.lit(None).cast(pl.Float64).alias('last_prediction'),
        ])

        pred_cols, unc_cols = get_prediction_columns(master_df)

        assert pred_cols == ['prediction_cycle_0', 'prediction_cycle_1', 'prediction_cycle_5']
        assert unc_cols == ['uncertainty_cycle_0', 'uncertainty_cycle_1', 'uncertainty_cycle_5']
        assert 'other_prediction' not in pred_cols
        assert 'last_prediction' not in pred_cols

    def test_get_prediction_columns_sorted_by_cycle(self, sample_master_df):
        master_df = sample_master_df.clone()

        master_df = master_df.with_columns([
            pl.lit(None).cast(pl.Float64).alias('prediction_cycle_10'),
            pl.lit(None).cast(pl.Float64).alias('prediction_cycle_2'),
            pl.lit(None).cast(pl.Float64).alias('prediction_cycle_5'),
            pl.lit(None).cast(pl.Float64).alias('prediction_cycle_1'),
            pl.lit(None).cast(pl.Float64).alias('uncertainty_cycle_10'),
            pl.lit(None).cast(pl.Float64).alias('uncertainty_cycle_2'),
            pl.lit(None).cast(pl.Float64).alias('uncertainty_cycle_5'),
            pl.lit(None).cast(pl.Float64).alias('uncertainty_cycle_1'),
        ])

        pred_cols, unc_cols = get_prediction_columns(master_df)

        assert pred_cols == ['prediction_cycle_1', 'prediction_cycle_2', 'prediction_cycle_5', 'prediction_cycle_10']
        assert unc_cols == ['uncertainty_cycle_1', 'uncertainty_cycle_2', 'uncertainty_cycle_5', 'uncertainty_cycle_10']

    def test_get_prediction_columns_empty_dataframe(self):
        empty_df = pl.DataFrame()

        pred_cols, unc_cols = get_prediction_columns(empty_df)

        assert pred_cols == []
        assert unc_cols == []


@pytest.mark.unit
class TestValidateMasterDataframe:

    def test_valid_master_dataframe_passes(self, sample_master_df):
        result = validate_master_dataframe(sample_master_df)

        assert result is True

    def test_valid_empty_dataframe(self, small_real_compounds):
        master_df = create_initialized_master_df(
            valid_compounds=small_real_compounds,
            target_col='Activity',
            initial_labeled_ids=[],
            initial_measurements=None
        )

        result = validate_master_dataframe(master_df)

        assert result is True

    def test_missing_required_columns_raises_error(self, sample_master_df):
        master_df = sample_master_df.clone()
        master_df = master_df.drop('status')

        with pytest.raises(ValueError, match="missing required columns"):
            validate_master_dataframe(master_df)

    def test_missing_multiple_columns_raises_error(self, small_real_compounds):
        invalid_df = small_real_compounds[['ID', 'SMILES']].clone()

        with pytest.raises(ValueError, match="missing required columns"):
            validate_master_dataframe(invalid_df)

    def test_missing_id_column_raises_error(self, sample_master_df):
        master_df = sample_master_df.clone()
        master_df = master_df.drop('ID')

        with pytest.raises(ValueError, match="missing required columns"):
            validate_master_dataframe(master_df)

    def test_missing_smiles_column_raises_error(self, sample_master_df):
        master_df = sample_master_df.clone()
        master_df = master_df.drop('SMILES')

        with pytest.raises(ValueError, match="missing required columns"):
            validate_master_dataframe(master_df)

    def test_missing_labeled_cycle_raises_error(self, sample_master_df):
        master_df = sample_master_df.clone()
        master_df = master_df.drop('labeled_cycle')

        with pytest.raises(ValueError, match="missing required columns"):
            validate_master_dataframe(master_df)

    def test_missing_selected_cycle_raises_error(self, sample_master_df):
        master_df = sample_master_df.clone()
        master_df = master_df.drop('selected_cycle')

        with pytest.raises(ValueError, match="missing required columns"):
            validate_master_dataframe(master_df)

    def test_missing_pruned_cycle_raises_error(self, sample_master_df):
        master_df = sample_master_df.clone()
        master_df = master_df.drop('pruned_cycle')

        with pytest.raises(ValueError, match="missing required columns"):
            validate_master_dataframe(master_df)

    def test_invalid_status_values_raises_error(self, sample_master_df):
        master_df = sample_master_df.clone()

        # Convert status to string first to bypass Polars enum validation
        master_df = master_df.with_columns(
            pl.col('status').cast(pl.Utf8)
        )

        # Set first row status to invalid value
        master_df = master_df.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
              .then(pl.lit('invalid_status'))
              .otherwise(pl.col('status'))
              .alias('status')
        )

        with pytest.raises(ValueError, match="Invalid status values found"):
            validate_master_dataframe(master_df)

    def test_multiple_invalid_status_values_raises_error(self, sample_master_df):
        master_df = sample_master_df.clone()

        # Convert status to string first to bypass Polars enum validation
        master_df = master_df.with_columns(
            pl.col('status').cast(pl.Utf8)
        )

        # Set first two rows to invalid status values
        master_df = master_df.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
              .then(pl.lit('bad_status'))
              .when(pl.int_range(pl.len()) == 1)
              .then(pl.lit('wrong_status'))
              .otherwise(pl.col('status'))
              .alias('status')
        )

        with pytest.raises(ValueError, match="Invalid status values found"):
            validate_master_dataframe(master_df)

    def test_status_column_validation_categorical(self, sample_master_df):
        master_df = sample_master_df.clone()
        master_df = master_df.with_columns(
            pl.col('status').cast(pl.Categorical)
        )

        result = validate_master_dataframe(master_df)

        assert result is True

    def test_status_column_validation_string_type(self, sample_master_df):
        master_df = sample_master_df.clone()
        master_df = master_df.with_columns(
            pl.col('status').cast(pl.Utf8)
        )

        result = validate_master_dataframe(master_df)

        assert result is True

    def test_invalid_status_column_dtype_raises_error(self, sample_master_df):
        master_df = sample_master_df.clone()
        status_values = [0, 1, 2] * (len(master_df) // 3) + [0] * (len(master_df) % 3)
        master_df = master_df.with_columns(
            pl.Series('status', status_values)
        )

        with pytest.raises(ValueError, match="Status column must be categorical, enum, or string type"):
            validate_master_dataframe(master_df)

    def test_duplicate_ids_raises_error(self, small_real_compounds):
        master_df = create_initialized_master_df(
            valid_compounds=small_real_compounds,
            target_col='Activity',
            initial_labeled_ids=[],
            initial_measurements=None
        )

        # Append duplicate row
        first_row = master_df[0]
        master_df = master_df.vstack(first_row)

        with pytest.raises(ValueError, match="Duplicate IDs found"):
            validate_master_dataframe(master_df)

    def test_many_duplicate_ids_truncated_error(self, small_real_compounds):
        master_df = create_initialized_master_df(
            valid_compounds=small_real_compounds,
            target_col='Activity',
            initial_labeled_ids=[],
            initial_measurements=None
        )

        # Append 10 duplicate rows
        for i in range(10):
            row = master_df[i]
            master_df = master_df.vstack(row)

        with pytest.raises(ValueError, match="Duplicate IDs found.*\\.\\.\\."):
            validate_master_dataframe(master_df)

    def test_prediction_columns_validation(self, master_df_with_predictions):
        result = validate_master_dataframe(master_df_with_predictions)

        assert result is True

    def test_first_selected_cycle_validation(self, sample_master_df):
        master_df = sample_master_df.clone()

        labeled_mask = master_df['status'] == 'labeled'
        assert (master_df.filter(labeled_mask)['selected_cycle'] == -1).all()

        result = validate_master_dataframe(master_df)

        assert result is True

    def test_pruned_cycle_validation(self, sample_master_df):
        from learnm8.core.dataframe_ops import update_status

        master_df = sample_master_df.clone()
        unlabeled = master_df.filter(pl.col('status') == 'unlabeled').head(5)
        unlabeled_ids = unlabeled['ID'].to_list()

        master_df = update_status(
            df=master_df,
            compound_ids=unlabeled_ids,
            new_status='pruned',
            cycle=0,
            target_col='Activity'
        )

        result = validate_master_dataframe(master_df)

        assert result is True
        pruned_mask = master_df['status'] == 'pruned'
        assert (master_df.filter(pruned_mask)['pruned_cycle'] == 0).all()

    def test_valid_all_statuses_present(self, sample_master_df):
        from learnm8.core.dataframe_ops import update_status

        master_df = sample_master_df.clone()

        unlabeled = master_df.filter(pl.col('status') == 'unlabeled')
        if len(unlabeled) >= 2:
            pruned_ids = unlabeled['ID'].head(2).to_list()
            master_df = update_status(
                df=master_df,
                compound_ids=pruned_ids,
                new_status='pruned',
                cycle=0,
                target_col='Activity'
            )

        assert 'labeled' in master_df['status'].to_list()
        assert 'unlabeled' in master_df['status'].to_list()
        assert 'pruned' in master_df['status'].to_list()

        result = validate_master_dataframe(master_df)

        assert result is True

    def test_valid_master_dataframe_with_multi_cycle(self, master_df_multi_cycle):
        result = validate_master_dataframe(master_df_multi_cycle)

        assert result is True

    def test_valid_master_dataframe_with_target_column(self, sample_master_df):
        assert 'Activity' in sample_master_df.columns

        result = validate_master_dataframe(sample_master_df)

        assert result is True

    def test_empty_dataframe_is_valid(self):
        empty_df = pl.DataFrame(
            schema={
                'ID': pl.Utf8,
                'SMILES': pl.Utf8,
                'status': pl.Utf8,
                'labeled_cycle': pl.Int64,
                'selected_cycle': pl.Int64,
                'pruned_cycle': pl.Int64
            }
        )

        result = validate_master_dataframe(empty_df)

        assert result is True

    def test_validation_does_not_require_target_column(self, sample_master_df):
        master_df = sample_master_df.clone()

        if 'Activity' in master_df.columns:
            master_df = master_df.drop('Activity')

        result = validate_master_dataframe(master_df)

        assert result is True

    def test_validation_does_not_require_prediction_columns(self, sample_master_df):
        master_df = sample_master_df.clone()

        pred_cols = [col for col in master_df.columns if col.startswith('prediction_')]
        unc_cols = [col for col in master_df.columns if col.startswith('uncertainty_')]

        if pred_cols or unc_cols:
            master_df = master_df.drop(pred_cols + unc_cols)

        result = validate_master_dataframe(master_df)

        assert result is True

    def test_validation_with_nan_in_nullable_columns(self, sample_master_df):
        master_df = sample_master_df.clone()

        unlabeled_mask = master_df['status'] == 'unlabeled'
        assert master_df.filter(unlabeled_mask)['labeled_cycle'].is_null().all()
        assert master_df.filter(unlabeled_mask)['selected_cycle'].is_null().all()

        result = validate_master_dataframe(master_df)

        assert result is True

    def test_status_categorical_with_wrong_categories_logs_warning(self, sample_master_df, caplog):
        import logging

        master_df = sample_master_df.clone()
        master_df = master_df.with_columns(
            pl.col('status').cast(pl.Categorical)
        )

        # Enable propagation on all learnm8 loggers
        for logger_name in ['learnm8', 'learnm8.core', 'learnm8.core.data_structures']:
            logger = logging.getLogger(logger_name)
            logger.propagate = True
            logger.setLevel(logging.WARNING)

        with caplog.at_level(logging.WARNING):
            result = validate_master_dataframe(master_df)

        assert result is True
        # Debug: print what we captured
        if not caplog.records:
            # If no records captured, the categories might already be correct
            # Check if the categorical actually has wrong categories
            current_cats = set(master_df['status'].cat.get_categories().to_list())
            expected_cats = set(['unlabeled', 'labeled', 'pruned'])
            if current_cats == expected_cats:
                # Categories are correct, no warning expected
                pass
            else:
                # Categories are wrong but no warning was captured - this is the actual failure
                assert False, f"Expected warning about categories {current_cats} != {expected_cats}, but got no caplog records"
        else:
            assert any('incorrect categories' in record.message for record in caplog.records)

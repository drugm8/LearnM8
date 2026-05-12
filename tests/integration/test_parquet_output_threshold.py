import warnings

import polars as pl
import pytest

from learnm8.core.persistence import _resolve_output_format, save_results
from learnm8.core.validation import ValidationResult
from learnm8.exceptions import LearnM8Warning

pytestmark = pytest.mark.integration


def _make_compounds_df(n_rows: int, target_col: str = 'Activity') -> pl.DataFrame:
    return pl.DataFrame(
        {
            'ID': [f'CMP_{i:08d}' for i in range(n_rows)],
            'SMILES': ['CCO'] * n_rows,
            'status': ['labeled'] * n_rows,
            'labeled_cycle': list(range(n_rows)),
            'selected_cycle': [0] * n_rows,
            'pruned_cycle': [-1] * n_rows,
            target_col: [0.5] * n_rows,
        }
    )


def _make_validation_result(compounds_df: pl.DataFrame) -> ValidationResult:
    return ValidationResult(
        valid_compounds=compounds_df,
        invalid_compounds=pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8}),
        validation_errors={},
    )


def _make_cycle_metrics(n_cycles: int = 1, batch_size: int = 0) -> list[dict]:
    return [
        {
            'cycle': i,
            'strategy': 'greedy',
            'batch_size': batch_size,
            'selected_predictions': pl.DataFrame(
                schema={
                    'ID': pl.Utf8,
                    'prediction': pl.Float64,
                    'uncertainty': pl.Float64,
                }
            ),
        }
        for i in range(n_cycles)
    ]


class TestResolveOutputFormat:
    def test_auto_parquet_above_1M_rows(self):
        fmt, warning = _resolve_output_format('auto', 1_100_000, 'compounds_final')
        assert fmt == 'parquet'
        assert warning is None

    def test_auto_csv_at_or_below_1M(self):
        fmt, warning = _resolve_output_format('auto', 100_000, 'compounds_final')
        assert fmt == 'csv'
        assert warning is None

    def test_boundary_exactly_1M_csv(self):
        fmt, warning = _resolve_output_format('auto', 1_000_000, 'compounds_final')
        assert fmt == 'csv'
        assert warning is None

    def test_boundary_1M_plus_1_parquet(self):
        fmt, warning = _resolve_output_format('auto', 1_000_001, 'compounds_final')
        assert fmt == 'parquet'
        assert warning is None

    def test_csv_override_on_large_data_warns(self):
        fmt, warning = _resolve_output_format('csv', 1_100_000, 'compounds_final')
        assert fmt == 'csv'
        assert warning is not None
        assert 'CSV' in warning
        assert 'parquet' in warning
        assert '1,100,000' in warning

    def test_parquet_on_any_size_honored_no_warning(self):
        for n_rows in (100, 100_000, 1_000_000, 5_000_000):
            fmt, warning = _resolve_output_format('parquet', n_rows, 'compounds_final')
            assert fmt == 'parquet'
            assert warning is None

    def test_cycle_metrics_always_csv_regardless_of_format(self):
        for fmt_in in ('auto', 'csv', 'parquet'):
            fmt, warning = _resolve_output_format(fmt_in, 5_000_000, 'cycle_metrics')
            assert fmt == 'csv'
            assert warning is None

    def test_selection_history_resolves_parquet_above_1M(self):
        fmt, warning = _resolve_output_format('auto', 1_500_000, 'selection_history')
        assert fmt == 'parquet'
        assert warning is None

    def test_selection_history_csv_at_or_below_1M(self):
        for n_rows in (100, 1_000_000):
            fmt, warning = _resolve_output_format('auto', n_rows, 'selection_history')
            assert fmt == 'csv'
            assert warning is None


class TestSaveResultsOutputFormat:
    def test_auto_csv_at_100k_writes_csv(self, tmp_path):
        n_rows = 100_000
        compounds_df = _make_compounds_df(n_rows)
        validation_result = _make_validation_result(compounds_df)
        cycle_metrics = _make_cycle_metrics()
        config = {'target_col': 'Activity'}
        output_dir = tmp_path / 'output'

        saved_files = save_results(
            compounds_df,
            cycle_metrics,
            validation_result,
            config,
            output_dir,
            output_format='auto',
        )

        assert saved_files['compounds_final'].suffix == '.csv'
        assert saved_files['compounds_final'].exists()

    def test_output_format_csv_override_on_large_data_with_warning(self, tmp_path):
        n_rows = 1_000_001
        compounds_df = _make_compounds_df(n_rows)
        validation_result = _make_validation_result(compounds_df)
        cycle_metrics = _make_cycle_metrics()
        config = {'target_col': 'Activity'}
        output_dir = tmp_path / 'output'

        with pytest.warns(LearnM8Warning, match='Writing.*CSV'):
            saved_files = save_results(
                compounds_df,
                cycle_metrics,
                validation_result,
                config,
                output_dir,
                output_format='csv',
            )

        assert saved_files['compounds_final'].suffix == '.csv'
        assert saved_files['compounds_final'].exists()

    def test_output_format_parquet_on_small_data_honored_no_warning(self, tmp_path):
        compounds_df = _make_compounds_df(100)
        validation_result = _make_validation_result(compounds_df)
        cycle_metrics = _make_cycle_metrics()
        config = {'target_col': 'Activity'}
        output_dir = tmp_path / 'output'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            saved_files = save_results(
                compounds_df,
                cycle_metrics,
                validation_result,
                config,
                output_dir,
                output_format='parquet',
            )

        learnm8_warnings = [x for x in w if issubclass(x.category, LearnM8Warning)]
        assert len(learnm8_warnings) == 0

        assert saved_files['compounds_final'].suffix == '.parquet'
        assert saved_files['compounds_final'].exists()

    def test_cycle_metrics_always_csv(self, tmp_path):
        compounds_df = _make_compounds_df(100)
        validation_result = _make_validation_result(compounds_df)
        cycle_metrics = _make_cycle_metrics()
        config = {'target_col': 'Activity'}
        output_dir = tmp_path / 'output'

        saved_files = save_results(
            compounds_df,
            cycle_metrics,
            validation_result,
            config,
            output_dir,
            output_format='auto',
        )

        assert saved_files['cycle_metrics'].suffix == '.csv'
        assert saved_files['cycle_metrics'].exists()

    def test_selection_history_parquet_above_1M(self, tmp_path):
        n_rows = 1_000_001
        compounds_df = _make_compounds_df(n_rows)
        validation_result = _make_validation_result(compounds_df)
        selected_preds = pl.DataFrame(
            {
                'ID': [f'CMP_{i:08d}' for i in range(n_rows)],
                'prediction': [float(i % 100) for i in range(n_rows)],
                'uncertainty': [0.1] * n_rows,
            }
        )
        cycle_metrics = [
            {
                'cycle': 0,
                'strategy': 'greedy',
                'batch_size': n_rows,
                'selected_predictions': selected_preds,
            }
        ]
        config = {'target_col': 'Activity'}
        output_dir = tmp_path / 'output'

        saved_files = save_results(
            compounds_df,
            cycle_metrics,
            validation_result,
            config,
            output_dir,
            output_format='auto',
        )

        assert saved_files['selection_history'].suffix == '.parquet'
        assert saved_files['selection_history'].exists()

import polars as pl
import pytest

from learnm8.visualization import animation

pytestmark = pytest.mark.unit


_SAMPLE_DF = pl.DataFrame(
    {
        'ID': ['cmp_1', 'cmp_2', 'cmp_3'],
        'SMILES': ['CC', 'CCO', 'CCN'],
        'score': [1.0, 2.5, 3.7],
        'label': ['A', 'B', 'C'],
    }
)


def test_read_results_file_auto_detects_parquet(tmp_path):
    parquet_path = tmp_path / 'data.parquet'
    _SAMPLE_DF.write_parquet(parquet_path)

    result = animation._read_results_file(tmp_path / 'data')

    assert isinstance(result, pl.DataFrame)
    assert result.shape == _SAMPLE_DF.shape
    assert result.columns == _SAMPLE_DF.columns
    for col in _SAMPLE_DF.columns:
        assert result[col].to_list() == _SAMPLE_DF[col].to_list()


def test_read_results_file_auto_detects_csv(tmp_path):
    csv_path = tmp_path / 'data.csv'
    _SAMPLE_DF.write_csv(csv_path)

    result = animation._read_results_file(tmp_path / 'data')

    assert isinstance(result, pl.DataFrame)
    assert result.shape == _SAMPLE_DF.shape
    assert result.columns == _SAMPLE_DF.columns


def test_read_results_file_missing_both_raises(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match=r'Results file not found.+tried \.parquet and \.csv extensions',
    ):
        animation._read_results_file(tmp_path / 'nonexistent')


def test_read_results_file_parquet_matches_csv_schema(tmp_path):
    parquet_path = tmp_path / 'data.parquet'
    csv_path = tmp_path / 'data.csv'
    _SAMPLE_DF.write_parquet(parquet_path)
    _SAMPLE_DF.write_csv(csv_path)

    result_parquet = animation._read_results_file(parquet_path)
    result_csv = animation._read_results_file(csv_path)

    assert result_parquet.columns == result_csv.columns
    assert result_parquet.shape == result_csv.shape
    for col in result_parquet.columns:
        parquet_vals = result_parquet[col].to_list()
        csv_vals = result_csv[col].to_list()
        if all(isinstance(v, (int, float)) or v is None for v in parquet_vals):
            assert parquet_vals == pytest.approx(csv_vals, nan_ok=True), (
                f'mismatch in column {col}'
            )
        else:
            assert parquet_vals == csv_vals, f'mismatch in column {col}'

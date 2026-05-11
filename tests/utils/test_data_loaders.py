import pytest
import polars as pl
from learnm8.utils.data_loaders import (
    validate_csv_columns,
    load_run_data,
    detect_score_direction_from_data,
)


@pytest.mark.unit
class TestValidateCsvColumns:

    def test_all_columns_present(self):
        df = pl.DataFrame({'ID': ['A'], 'SMILES': ['CCO'], 'Activity': [1.0]})
        validate_csv_columns(df, ['ID', 'SMILES', 'Activity'], "test file")

    def test_missing_column_raises(self):
        df = pl.DataFrame({'ID': ['A'], 'SMILES': ['CCO']})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_csv_columns(df, ['ID', 'SMILES', 'Activity'], "test file")

    def test_multiple_missing_columns_raises(self):
        df = pl.DataFrame({'Name': ['A']})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_csv_columns(df, ['ID', 'SMILES'], "test file")

    def test_error_message_includes_missing_columns(self):
        df = pl.DataFrame({'ID': ['A']})
        with pytest.raises(ValueError, match="SMILES"):
            validate_csv_columns(df, ['ID', 'SMILES'], "test file")

    def test_error_message_includes_available_columns(self):
        df = pl.DataFrame({'ID': ['A'], 'Name': ['mol']})
        with pytest.raises(ValueError, match="Name"):
            validate_csv_columns(df, ['ID', 'SMILES'], "test file")

    def test_empty_required_list(self):
        df = pl.DataFrame({'ID': ['A']})
        validate_csv_columns(df, [], "test file")

    def test_file_description_in_error(self):
        df = pl.DataFrame({'ID': ['A']})
        with pytest.raises(ValueError, match="my_description"):
            validate_csv_columns(df, ['SMILES'], "my_description")


@pytest.mark.unit
class TestLoadRunData:

    def test_returns_dataframe(self, tmp_path):
        csv_file = tmp_path / "compounds.csv"
        csv_file.write_text("ID,SMILES\nCOMP_001,CCO\nCOMP_002,CCC\n")
        oracle_file = tmp_path / "oracle.py"
        oracle_file.write_text("def oracle(ids): pass\n")
        result = load_run_data(str(csv_file), str(oracle_file))
        assert isinstance(result, pl.DataFrame)

    def test_has_required_columns(self, tmp_path):
        csv_file = tmp_path / "compounds.csv"
        csv_file.write_text("ID,SMILES\nCOMP_001,CCO\n")
        oracle_file = tmp_path / "oracle.py"
        oracle_file.write_text("def oracle(ids): pass\n")
        result = load_run_data(str(csv_file), str(oracle_file))
        assert 'ID' in result.columns
        assert 'SMILES' in result.columns

    def test_compounds_file_not_found(self, tmp_path):
        oracle_file = tmp_path / "oracle.py"
        oracle_file.write_text("def oracle(ids): pass\n")
        with pytest.raises(FileNotFoundError, match="Compound pool file not found"):
            load_run_data(str(tmp_path / "nonexistent.csv"), str(oracle_file))

    def test_oracle_file_not_found(self, tmp_path):
        csv_file = tmp_path / "compounds.csv"
        csv_file.write_text("ID,SMILES\nCOMP_001,CCO\n")
        with pytest.raises(FileNotFoundError, match="Oracle file not found"):
            load_run_data(str(csv_file), str(tmp_path / "nonexistent.py"))

    def test_missing_id_column_auto_generated(self, tmp_path):
        csv_file = tmp_path / "compounds.csv"
        csv_file.write_text("SMILES\nCCO\n")
        oracle_file = tmp_path / "oracle.py"
        oracle_file.write_text("def oracle(ids): pass\n")
        result = load_run_data(str(csv_file), str(oracle_file))
        assert 'ID' in result.columns

    def test_non_py_oracle_extension_warns(self, tmp_path, caplog):
        csv_file = tmp_path / "compounds.csv"
        csv_file.write_text("ID,SMILES\nCOMP_001,CCO\n")
        oracle_file = tmp_path / "oracle.txt"
        oracle_file.write_text("# not a python file\n")
        with caplog.at_level(logging.WARNING):
            load_run_data(str(csv_file), str(oracle_file))
        assert '.py' in caplog.text or 'extension' in caplog.text.lower()

    def test_correct_row_count(self, tmp_path):
        csv_file = tmp_path / "compounds.csv"
        rows = "\n".join(f"COMP_{i:03d},CCO" for i in range(10))
        csv_file.write_text(f"ID,SMILES\n{rows}\n")
        oracle_file = tmp_path / "oracle.py"
        oracle_file.write_text("def oracle(ids): pass\n")
        result = load_run_data(str(csv_file), str(oracle_file))
        assert result.height == 10


@pytest.mark.unit
class TestDetectScoreDirection:

    def test_activity_column_returns_higher(self):
        df = pl.DataFrame({'ID': ['A'], 'Activity': [0.5]})
        result = detect_score_direction_from_data(df, 'Activity')
        assert result == 'higher'

    def test_docking_column_returns_lower(self):
        df = pl.DataFrame({'ID': ['A'], 'dock_score': [-7.5]})
        result = detect_score_direction_from_data(df, 'dock_score')
        assert result == 'lower'

    def test_binding_energy_returns_lower(self):
        df = pl.DataFrame({'ID': ['A'], 'binding_energy': [-8.0]})
        result = detect_score_direction_from_data(df, 'binding_energy')
        assert result == 'lower'

    def test_score_column_returns_higher(self):
        df = pl.DataFrame({'ID': ['A'], 'score': [0.9]})
        result = detect_score_direction_from_data(df, 'score')
        assert result == 'higher'

    def test_affinity_column_returns_higher(self):
        df = pl.DataFrame({'ID': ['A'], 'affinity': [0.9]})
        result = detect_score_direction_from_data(df, 'affinity')
        assert result == 'higher'

    def test_rmsd_returns_lower(self):
        df = pl.DataFrame({'ID': ['A'], 'rmsd': [1.2]})
        result = detect_score_direction_from_data(df, 'rmsd')
        assert result == 'lower'

    def test_error_column_returns_lower(self):
        df = pl.DataFrame({'ID': ['A'], 'error': [0.1]})
        result = detect_score_direction_from_data(df, 'error')
        assert result == 'lower'

    def test_missing_column_returns_higher_default(self):
        df = pl.DataFrame({'ID': ['A'], 'Activity': [0.5]})
        result = detect_score_direction_from_data(df, 'NonExistent')
        assert result == 'higher'

    def test_mostly_negative_values_returns_lower(self):
        df = pl.DataFrame({'ID': [str(i) for i in range(10)], 'val': [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, 0.5]})
        result = detect_score_direction_from_data(df, 'val')
        assert result == 'lower'

    def test_mostly_positive_values_returns_higher_default(self):
        df = pl.DataFrame({'ID': [str(i) for i in range(5)], 'custom_prop': [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = detect_score_direction_from_data(df, 'custom_prop')
        assert result == 'higher'

    def test_accuracy_returns_higher(self):
        df = pl.DataFrame({'ID': ['A'], 'accuracy': [0.95]})
        result = detect_score_direction_from_data(df, 'accuracy')
        assert result == 'higher'

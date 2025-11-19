import pytest
from pathlib import Path
import polars as pl
import tempfile

from learnm8.utils.file_loaders import (
    load_compound_file,
    _detect_file_format,
    _load_csv_with_polars,
    _load_sdf_with_datamol,
    _load_smi_with_rdkit
)


class TestDetectFileFormat:
    def test_detect_csv(self):
        assert _detect_file_format(Path('test.csv')) == 'csv'

    def test_detect_tsv(self):
        assert _detect_file_format(Path('test.tsv')) == 'csv'

    def test_detect_txt(self):
        assert _detect_file_format(Path('test.txt')) == 'csv'

    def test_detect_sdf(self):
        assert _detect_file_format(Path('test.sdf')) == 'sdf'

    def test_detect_sd(self):
        assert _detect_file_format(Path('test.sd')) == 'sdf'

    def test_detect_smi(self):
        assert _detect_file_format(Path('test.smi')) == 'smi'

    def test_detect_smiles(self):
        assert _detect_file_format(Path('test.smiles')) == 'smi'

    def test_detect_sdf_gz(self):
        assert _detect_file_format(Path('test.sdf.gz')) == 'sdf'

    def test_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            _detect_file_format(Path('test.xlsx'))


class TestLoadCSV:
    @pytest.fixture
    def csv_file(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = pl.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCCC'],
            'Activity': [0.8, 0.5, 0.2]
        })
        df.write_csv(csv_path)
        return csv_path

    @pytest.fixture
    def csv_custom_columns(self, tmp_path):
        csv_path = tmp_path / "custom.csv"
        df = pl.DataFrame({
            'Compound_ID': ['COMP_001', 'COMP_002'],
            'Canonical_SMILES': ['CCO', 'CCC'],
            'Score': [0.8, 0.5]
        })
        df.write_csv(csv_path)
        return csv_path

    def test_load_csv_basic(self, csv_file):
        df = load_compound_file(csv_file)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert 'ID' in df.columns
        assert 'SMILES' in df.columns
        assert 'Activity' in df.columns

    def test_load_csv_custom_columns(self, csv_custom_columns):
        df = load_compound_file(
            csv_custom_columns,
            smiles_column='Canonical_SMILES',
            id_column='Compound_ID'
        )
        assert 'ID' in df.columns
        assert 'SMILES' in df.columns
        assert 'Score' in df.columns
        assert df['ID'].to_list() == ['COMP_001', 'COMP_002']

    def test_load_csv_generate_id(self, tmp_path):
        csv_path = tmp_path / "no_id.csv"
        df = pl.DataFrame({
            'SMILES': ['CCO', 'CCC', 'CCCC']
        })
        df.write_csv(csv_path)

        result = load_compound_file(csv_path)
        assert 'ID' in result.columns
        assert result['ID'].to_list() == ['compound_0', 'compound_1', 'compound_2']

    def test_load_csv_missing_smiles(self, tmp_path):
        csv_path = tmp_path / "no_smiles.csv"
        df = pl.DataFrame({
            'ID': ['COMP_001', 'COMP_002'],
            'Structure': ['CCO', 'CCC']
        })
        df.write_csv(csv_path)

        with pytest.raises(ValueError, match="SMILES column not found"):
            load_compound_file(csv_path)


class TestLoadSDF:
    @pytest.fixture
    def sdf_file(self):
        return Path(__file__).parent.parent / 'fixtures' / 'sample_compounds.sdf'

    @pytest.fixture
    def sdf_no_name(self):
        return Path(__file__).parent.parent / 'fixtures' / 'sample_compounds_no_name.sdf'

    def test_load_sdf_basic(self, sdf_file):
        if not sdf_file.exists():
            pytest.skip("Test fixture not found")

        df = load_compound_file(sdf_file, progress=False)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5
        assert 'ID' in df.columns
        assert 'SMILES' in df.columns
        assert 'Activity' in df.columns
        assert 'Category' in df.columns

    def test_load_sdf_with_properties(self, sdf_file):
        if not sdf_file.exists():
            pytest.skip("Test fixture not found")

        df = load_compound_file(sdf_file, progress=False)
        assert 'Activity' in df.columns
        assert 'Category' in df.columns
        assert df['ID'].to_list() == ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004', 'COMP_005']

    def test_load_sdf_without_name(self, sdf_no_name):
        if not sdf_no_name.exists():
            pytest.skip("Test fixture not found")

        df = load_compound_file(sdf_no_name, progress=False)
        assert 'ID' in df.columns
        assert all(mol_id.startswith('mol_') for mol_id in df['ID'].to_list())

    def test_load_sdf_custom_id_column(self, sdf_file):
        if not sdf_file.exists():
            pytest.skip("Test fixture not found")

        df = load_compound_file(sdf_file, id_column='Category', progress=False)
        assert 'ID' in df.columns


class TestLoadSMI:
    @pytest.fixture
    def smi_file(self):
        return Path(__file__).parent.parent / 'fixtures' / 'sample_compounds.smi'

    def test_load_smi_basic(self, smi_file):
        if not smi_file.exists():
            pytest.skip("Test fixture not found")

        df = load_compound_file(smi_file)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5
        assert 'ID' in df.columns
        assert 'SMILES' in df.columns

    def test_load_smi_with_properties(self, smi_file):
        if not smi_file.exists():
            pytest.skip("Test fixture not found")

        df = load_compound_file(smi_file)
        assert 'Activity' in df.columns

    def test_load_smi_invalid_format(self, tmp_path):
        smi_path = tmp_path / "invalid.smi"
        smi_path.write_text("Not a valid SMILES file\n")

        with pytest.raises(ValueError, match="Failed to load SMILES file|No valid molecules found"):
            load_compound_file(smi_path)


class TestErrorHandling:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Compound pool file not found"):
            load_compound_file('nonexistent.csv')

    def test_unsupported_format(self, tmp_path):
        xlsx_path = tmp_path / "test.xlsx"
        xlsx_path.touch()

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_compound_file(xlsx_path)

    def test_empty_sdf(self, tmp_path):
        sdf_path = tmp_path / "empty.sdf"
        sdf_path.write_text("$$$$\n")

        df = load_compound_file(sdf_path, progress=False)
        assert len(df) == 0


class TestIntegration:
    @pytest.fixture
    def csv_file(self):
        return Path(__file__).parent.parent / 'fixtures' / 'sample_compounds.csv'

    @pytest.fixture
    def sdf_file(self):
        return Path(__file__).parent.parent / 'fixtures' / 'sample_compounds.sdf'

    @pytest.fixture
    def smi_file(self):
        return Path(__file__).parent.parent / 'fixtures' / 'sample_compounds.smi'

    def test_csv_sdf_smi_consistency(self, csv_file, sdf_file, smi_file):
        if not all([csv_file.exists(), sdf_file.exists(), smi_file.exists()]):
            pytest.skip("Test fixtures not found")

        df_csv = load_compound_file(csv_file)
        df_sdf = load_compound_file(sdf_file, progress=False)
        df_smi = load_compound_file(smi_file)

        assert len(df_csv) == len(df_sdf) == len(df_smi) == 5
        assert 'ID' in df_csv.columns and 'ID' in df_sdf.columns and 'ID' in df_smi.columns
        assert 'SMILES' in df_csv.columns and 'SMILES' in df_sdf.columns and 'SMILES' in df_smi.columns

    def test_all_formats_have_required_columns(self, csv_file, sdf_file, smi_file):
        if not all([csv_file.exists(), sdf_file.exists(), smi_file.exists()]):
            pytest.skip("Test fixtures not found")

        for file_path in [csv_file, sdf_file, smi_file]:
            df = load_compound_file(file_path, progress=False)
            required = {'ID', 'SMILES'}
            assert required.issubset(set(df.columns)), f"Missing columns in {file_path.suffix}"


class TestBackwardCompatibility:
    def test_csv_loading_unchanged(self, tmp_path):
        csv_path = tmp_path / "compounds.csv"
        original_df = pl.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCCC'],
            'Activity': [0.8, 0.5, 0.2]
        })
        original_df.write_csv(csv_path)

        loaded_df = load_compound_file(csv_path)

        assert loaded_df.equals(original_df)

    def test_csv_behavior_matches_pl_read_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        test_df = pl.DataFrame({
            'ID': ['A', 'B', 'C'],
            'SMILES': ['CCO', 'CCC', 'CCCC']
        })
        test_df.write_csv(csv_path)

        df_old = pl.read_csv(csv_path)
        df_new = load_compound_file(csv_path)

        assert df_old.equals(df_new)

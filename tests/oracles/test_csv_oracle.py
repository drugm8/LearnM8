import pytest
import polars as pl
from pathlib import Path
from learnm8.oracles.csv_oracle import CSVOracle


@pytest.mark.unit
class TestCSVOracleInit:

    def test_init_valid_csv(self, tmp_path):
        csv_file = tmp_path / "oracle.csv"
        csv_file.write_text("ID,Activity\nCOMP_001,1.5\nCOMP_002,2.3\n")
        oracle = CSVOracle(str(csv_file))
        assert oracle.csv_path == csv_file
        assert oracle.ground_truth.height == 2

    def test_init_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CSVOracle(str(tmp_path / "nonexistent.csv"))

    def test_init_custom_id_column(self, tmp_path):
        csv_file = tmp_path / "oracle.csv"
        csv_file.write_text("CompoundID,Activity\nCOMP_001,1.5\nCOMP_002,2.3\n")
        oracle = CSVOracle(str(csv_file), id_column='CompoundID')
        assert 'ID' in oracle.ground_truth.columns

    def test_init_missing_id_column(self, tmp_path):
        csv_file = tmp_path / "oracle.csv"
        csv_file.write_text("Name,Activity\nCOMP_001,1.5\n")
        with pytest.raises(ValueError, match="ID column"):
            CSVOracle(str(csv_file))

    def test_init_filters_null_rows(self, tmp_path):
        csv_file = tmp_path / "oracle.csv"
        csv_file.write_text("ID,Activity\nCOMP_001,1.5\nCOMP_002,no_score\n")
        oracle = CSVOracle(str(csv_file))
        assert oracle.ground_truth.height == 1

    def test_init_with_smiles_column(self, tmp_path):
        csv_file = tmp_path / "oracle.csv"
        csv_file.write_text("ID,SMILES,Activity\nCOMP_001,CCO,1.5\nCOMP_002,CCC,2.3\n")
        oracle = CSVOracle(str(csv_file))
        assert 'SMILES' in oracle.ground_truth.columns


@pytest.mark.unit
class TestCSVOracleMeasure:

    @pytest.fixture
    def oracle_with_data(self, tmp_path):
        csv_file = tmp_path / "oracle.csv"
        csv_file.write_text(
            "ID,SMILES,Activity\n"
            "COMP_001,CCO,1.5\n"
            "COMP_002,CCC,2.3\n"
            "COMP_003,CCCC,0.8\n"
            "COMP_004,c1ccccc1,3.1\n"
            "COMP_005,CC(=O)O,1.2\n"
        )
        return CSVOracle(str(csv_file))

    def test_measure_returns_correct_columns(self, oracle_with_data):
        compounds = pl.DataFrame({
            'ID': ['COMP_001', 'COMP_002'],
            'SMILES': ['CCO', 'CCC']
        })
        result = oracle_with_data.measure(compounds, ['Activity'])
        assert 'ID' in result.columns
        assert 'Activity' in result.columns

    def test_measure_correct_values(self, oracle_with_data):
        compounds = pl.DataFrame({
            'ID': ['COMP_001', 'COMP_003'],
            'SMILES': ['CCO', 'CCCC']
        })
        result = oracle_with_data.measure(compounds, ['Activity'])
        assert result.height == 2
        activities = result.sort('ID')['Activity'].to_list()
        assert activities[0] == pytest.approx(1.5)
        assert activities[1] == pytest.approx(0.8)

    def test_measure_subset_of_compounds(self, oracle_with_data):
        compounds = pl.DataFrame({
            'ID': ['COMP_002'],
            'SMILES': ['CCC']
        })
        result = oracle_with_data.measure(compounds, ['Activity'])
        assert result.height == 1
        assert result['Activity'][0] == pytest.approx(2.3)

    def test_measure_preserves_order(self, oracle_with_data):
        compounds = pl.DataFrame({
            'ID': ['COMP_005', 'COMP_001', 'COMP_003'],
            'SMILES': ['CC(=O)O', 'CCO', 'CCCC']
        })
        result = oracle_with_data.measure(compounds, ['Activity'])
        assert result['ID'].to_list() == ['COMP_005', 'COMP_001', 'COMP_003']

    def test_measure_missing_compounds(self, oracle_with_data, caplog):
        compounds = pl.DataFrame({
            'ID': ['COMP_001', 'COMP_999'],
            'SMILES': ['CCO', 'CCNCC']
        })
        with caplog.at_level('WARNING', logger='learnm8.oracles.csv_oracle'):
            result = oracle_with_data.measure(compounds, ['Activity'])
        assert result.height == 1
        assert 'not found' in caplog.text

    def test_measure_wrong_property_name(self, oracle_with_data):
        compounds = pl.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO']
        })
        with pytest.raises(ValueError, match="Properties not found"):
            oracle_with_data.measure(compounds, ['NonExistentProp'])

    def test_measure_multiple_properties(self, tmp_path):
        csv_file = tmp_path / "oracle.csv"
        csv_file.write_text("ID,SMILES,Activity,Score\nCOMP_001,CCO,1.5,0.9\nCOMP_002,CCC,2.3,0.4\n")
        oracle = CSVOracle(str(csv_file))
        compounds = pl.DataFrame({'ID': ['COMP_001'], 'SMILES': ['CCO']})
        result = oracle.measure(compounds, ['Activity', 'Score'])
        assert 'Activity' in result.columns
        assert 'Score' in result.columns

    def test_measure_smiles_from_ground_truth(self, tmp_path):
        csv_file = tmp_path / "oracle.csv"
        csv_file.write_text("ID,SMILES,Activity\nCOMP_001,CCO,1.5\nCOMP_002,CCC,2.3\n")
        oracle = CSVOracle(str(csv_file))
        compounds = pl.DataFrame({'ID': ['COMP_001', 'COMP_002']})
        result = oracle.measure(compounds, ['Activity'])
        assert 'SMILES' in result.columns

    def test_measure_integration_with_real_data(self, small_real_compounds, tmp_path):
        if 'Activity' not in small_real_compounds.columns:
            pytest.skip("small_real_compounds fixture missing Activity column")
        csv_file = tmp_path / "oracle.csv"
        small_real_compounds.write_csv(str(csv_file))
        oracle = CSVOracle(str(csv_file))
        subset = small_real_compounds.head(5).select(['ID', 'SMILES'])
        result = oracle.measure(subset, ['Activity'])
        assert result.height == 5
        assert 'Activity' in result.columns
        assert result['Activity'].is_null().sum() == 0

import pytest
import polars as pl
import numpy as np
from pathlib import Path
import tempfile

from examples.oracles import SimilarityOracle, Pharmacophore2DOracle

try:
    from examples.oracles import CDPKitPharmacophoreOracle
    CDPKIT_AVAILABLE = True
except (ImportError, AttributeError):
    CDPKIT_AVAILABLE = False

try:
    from examples.oracles import VinaOracle
    VINA_AVAILABLE = True
except (ImportError, AttributeError):
    VINA_AVAILABLE = False


class TestSimilarityOracle:

    def test_initialization_valid(self):
        oracle = SimilarityOracle(
            reference_smiles='CCO',
            fingerprint_type='morgan'
        )
        assert oracle.reference_smiles == 'CCO'
        assert oracle.fingerprint_type == 'morgan'

    def test_initialization_invalid_smiles(self):
        with pytest.raises(ValueError, match="Invalid reference SMILES"):
            SimilarityOracle(reference_smiles='INVALID')

    def test_initialization_invalid_fingerprint(self):
        with pytest.raises(ValueError, match="fingerprint_type must be one of"):
            SimilarityOracle(
                reference_smiles='CCO',
                fingerprint_type='invalid'
            )

    def test_initialization_invalid_metric(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            SimilarityOracle(
                reference_smiles='CCO',
                metric='invalid'
            )

    def test_measure_basic(self, sample_compounds):
        oracle = SimilarityOracle(
            reference_smiles='CCO',
            fingerprint_type='morgan'
        )

        result = oracle.measure(sample_compounds, ['similarity'])

        assert result.height == 3
        assert 'ID' in result.columns
        assert 'similarity' in result.columns

        similarities = result['similarity'].to_list()
        assert all(isinstance(s, (float, type(None))) for s in similarities)
        assert similarities[0] == 1.0

    def test_measure_multiple_properties(self, sample_compounds):
        oracle = SimilarityOracle(
            reference_smiles='CCO',
            fingerprint_type='morgan'
        )

        result = oracle.measure(sample_compounds, ['sim1', 'sim2'])

        assert result.height == 3
        assert 'sim1' in result.columns
        assert 'sim2' in result.columns

    def test_measure_empty_compounds(self, empty_compounds):
        oracle = SimilarityOracle(
            reference_smiles='CCO',
            fingerprint_type='morgan'
        )

        result = oracle.measure(empty_compounds, ['similarity'])

        assert result.height == 0
        assert 'ID' in result.columns
        assert 'similarity' in result.columns

    def test_measure_preserves_order(self):
        compounds = pl.DataFrame({
            'ID': ['z', 'a', 'm'],
            'SMILES': ['CCO', 'CCC', 'CCCC']
        })

        oracle = SimilarityOracle(reference_smiles='CCO')
        result = oracle.measure(compounds, ['similarity'])

        assert result['ID'].to_list() == ['z', 'a', 'm']

    def test_measure_invalid_smiles(self):
        compounds = pl.DataFrame({
            'ID': ['comp1', 'comp2'],
            'SMILES': ['CCO', 'INVALID']
        })

        oracle = SimilarityOracle(reference_smiles='CCO')
        result = oracle.measure(compounds, ['similarity'])

        similarities = result['similarity'].to_list()
        assert similarities[0] == 1.0
        assert similarities[1] is None

    def test_different_fingerprints(self, sample_compounds):
        for fp_type in ['morgan', 'maccs', 'topological']:
            oracle = SimilarityOracle(
                reference_smiles='CCO',
                fingerprint_type=fp_type
            )
            result = oracle.measure(sample_compounds, ['similarity'])
            assert result.height == 3

    def test_different_metrics(self, sample_compounds):
        for metric in ['tanimoto', 'dice']:
            oracle = SimilarityOracle(
                reference_smiles='CCO',
                metric=metric
            )
            result = oracle.measure(sample_compounds, ['similarity'])
            assert result.height == 3


class TestPharmacophore2DOracle:

    def test_initialization_valid(self):
        oracle = Pharmacophore2DOracle(
            reference_smiles='CCO',
            metric='tanimoto'
        )
        assert oracle.reference_smiles == 'CCO'
        assert oracle.metric == 'tanimoto'

    def test_initialization_invalid_smiles(self):
        with pytest.raises(ValueError, match="Invalid reference SMILES"):
            Pharmacophore2DOracle(reference_smiles='INVALID')

    def test_initialization_invalid_metric(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            Pharmacophore2DOracle(
                reference_smiles='CCO',
                metric='invalid'
            )

    def test_measure_basic(self, sample_compounds):
        oracle = Pharmacophore2DOracle(
            reference_smiles='c1ccccc1O'
        )

        result = oracle.measure(sample_compounds, ['pharmacophore_similarity'])

        assert result.height == 3
        assert 'ID' in result.columns
        assert 'pharmacophore_similarity' in result.columns

    def test_measure_empty_compounds(self, empty_compounds):
        oracle = Pharmacophore2DOracle(
            reference_smiles='c1ccccc1O'
        )

        result = oracle.measure(empty_compounds, ['similarity'])

        assert result.height == 0
        assert 'ID' in result.columns
        assert 'similarity' in result.columns

    def test_measure_preserves_order(self):
        compounds = pl.DataFrame({
            'ID': ['z', 'a', 'm'],
            'SMILES': ['c1ccccc1O', 'c1ccccc1N', 'c1ccccc1']
        })

        oracle = Pharmacophore2DOracle(reference_smiles='c1ccccc1O')
        result = oracle.measure(compounds, ['similarity'])

        assert result['ID'].to_list() == ['z', 'a', 'm']

    def test_different_metrics(self):
        compounds = pl.DataFrame({
            'ID': ['comp1', 'comp2'],
            'SMILES': ['c1ccccc1O', 'c1ccccc1N']
        })

        for metric in ['tanimoto', 'dice']:
            oracle = Pharmacophore2DOracle(
                reference_smiles='c1ccccc1O',
                metric=metric
            )
            result = oracle.measure(compounds, ['similarity'])
            assert result.height == 2


@pytest.mark.skipif(not CDPKIT_AVAILABLE, reason="CDPKit not installed")
class TestCDPKitPharmacophoreOracle:

    def test_initialization_valid(self):
        oracle = CDPKitPharmacophoreOracle(
            reference_smiles='CCO',
            score_type='alignment_score'
        )
        assert oracle.reference_smiles == 'CCO'
        assert oracle.score_type == 'alignment_score'

    def test_initialization_invalid_smiles(self):
        with pytest.raises(ValueError, match="Failed to generate"):
            CDPKitPharmacophoreOracle(reference_smiles='INVALID')

    def test_initialization_invalid_score_type(self):
        with pytest.raises(ValueError, match="score_type must be one of"):
            CDPKitPharmacophoreOracle(
                reference_smiles='CCO',
                score_type='invalid'
            )

    def test_measure_basic(self, sample_compounds):
        oracle = CDPKitPharmacophoreOracle(
            reference_smiles='c1ccccc1O'
        )

        result = oracle.measure(sample_compounds, ['ph4_score'])

        assert result.height == 3
        assert 'ID' in result.columns
        assert 'ph4_score' in result.columns

    def test_measure_empty_compounds(self, empty_compounds):
        oracle = CDPKitPharmacophoreOracle(
            reference_smiles='c1ccccc1O'
        )

        result = oracle.measure(empty_compounds, ['score'])

        assert result.height == 0
        assert 'ID' in result.columns
        assert 'score' in result.columns

    def test_different_score_types(self):
        compounds = pl.DataFrame({
            'ID': ['comp1'],
            'SMILES': ['c1ccccc1O']
        })

        for score_type in ['alignment_score', 'feature_overlap', 'rmsd']:
            oracle = CDPKitPharmacophoreOracle(
                reference_smiles='c1ccccc1O',
                score_type=score_type
            )
            result = oracle.measure(compounds, ['score'])
            assert result.height == 1


@pytest.mark.skipif(not VINA_AVAILABLE, reason="Vina/Meeko not installed")
class TestVinaOracle:

    @pytest.fixture
    def mock_receptor_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False) as f:
            f.write("MOCK RECEPTOR PDBQT\n")
            receptor_path = f.name

        yield receptor_path

        Path(receptor_path).unlink(missing_ok=True)

    def test_initialization_valid(self, mock_receptor_file):
        oracle = VinaOracle(
            receptor_path=mock_receptor_file,
            center=(0.0, 0.0, 0.0),
            box_size=(20.0, 20.0, 20.0)
        )
        assert oracle.center == (0.0, 0.0, 0.0)
        assert oracle.box_size == (20.0, 20.0, 20.0)

    def test_initialization_missing_receptor(self):
        with pytest.raises(FileNotFoundError, match="Receptor file not found"):
            VinaOracle(
                receptor_path='/nonexistent/receptor.pdbqt',
                center=(0.0, 0.0, 0.0)
            )

    def test_measure_structure(self, mock_receptor_file):
        oracle = VinaOracle(
            receptor_path=mock_receptor_file,
            center=(0.0, 0.0, 0.0)
        )

        compounds = pl.DataFrame({
            'ID': ['comp1'],
            'SMILES': ['CCO']
        })

        result = oracle.measure(compounds, ['binding_affinity'])

        assert result.height == 1
        assert 'ID' in result.columns
        assert 'binding_affinity' in result.columns

    def test_measure_empty_compounds(self, mock_receptor_file, empty_compounds):
        oracle = VinaOracle(
            receptor_path=mock_receptor_file,
            center=(0.0, 0.0, 0.0)
        )

        result = oracle.measure(empty_compounds, ['affinity'])

        assert result.height == 0
        assert 'ID' in result.columns
        assert 'affinity' in result.columns


class TestOracleInterface:

    def test_missing_columns(self):
        oracle = SimilarityOracle(reference_smiles='CCO')

        bad_df = pl.DataFrame({
            'wrong_column': ['comp1', 'comp2']
        })

        with pytest.raises(ValueError, match="must have 'ID' and 'SMILES' columns"):
            oracle.measure(bad_df, ['similarity'])

    def test_missing_id_column(self):
        oracle = SimilarityOracle(reference_smiles='CCO')

        bad_df = pl.DataFrame({
            'SMILES': ['CCO', 'CCC']
        })

        with pytest.raises(ValueError, match="must have 'ID' and 'SMILES' columns"):
            oracle.measure(bad_df, ['similarity'])

    def test_missing_smiles_column(self):
        oracle = SimilarityOracle(reference_smiles='CCO')

        bad_df = pl.DataFrame({
            'ID': ['comp1', 'comp2']
        })

        with pytest.raises(ValueError, match="must have 'ID' and 'SMILES' columns"):
            oracle.measure(bad_df, ['similarity'])

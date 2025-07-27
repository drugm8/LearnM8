"""Pytest configuration and fixtures for LearnM8 tests."""

import pandas as pd
import pytest
from pathlib import Path
from learnm8.utils.featurizers import precompute_representations


@pytest.fixture
def sample_molecular_data():
    """Small subset of real molecular data from ADA dataset for testing."""
    data = {
        'ID': [
            'C39699189_decoy',
            'C63303102_decoy', 
            'C12603607_decoy',
            'C03209676_decoy',
            'C60324373_decoy',
            'C18113681_decoy',
            'C63907389_decoy',
            'C60692778_decoy',
            'C08497416_decoy',
            'C22107745_decoy'
        ],
        'SMILES': [
            'c1ccc2c(c1)c(c[nH]2)CCNC(=O)[C@@H]3C=C4N=C(C=C(N4N3)C(F)(F)F)c5ccco5',
            'c1ccc2=[NH+]CC(=c2c1)CCNC(=O)[C@@H]3NC(=NO3)C4=C[C@H]5C(=CC=N5)C=C4',
            'Cc1ccc(cc1)C(=O)NC[C@@H]2[C@H]([C@H]([C@@H](O2)CC(=O)N(C)Cc3ccccc3)O)O',
            'c1ccc(cc1)C(O)(C(=O)N/N=C/C=N/NC(=O)C(O)(c2ccccc2)c3ccccc3)c4ccccc4',
            'c1ccc(c(c1)C(=O)Nc2cccc(c2)C(F)(F)F)NC(=O)c3cccc(c3)OCC(=O)N',
            'c1ccc2c(c1)c(=O)[nH+]c(s2)N/C(=N/C(=O)Nc3ccc(cc3)F)/N',
            'C[C@H]1CCCC[C@H]1NC(=O)[C@@H](C)N2C(=O)[C@@H](NC2=O)CC3=c4ccccc4=[NH+]C3',
            'c1cc2c(cc1CNC(=O)c3ccc(c(c3)[N+](=O)[O-])N)CCC2',
            'C[C@@H](CCC(=O)NN)[C@H]1CC[C@@H]2[C@@]1(CC[C@@H]3[C@@H]2[C@@H](C[C@@H]4[C@@]3(CC[C@@H](C4)O)C)O)C',
            'COC(=O)[C@H](C)N1C(=O)c2ccccc2C1=O'
        ],
        'ESSENCE-Dock_Score': [
            -8.703801987097885,
            -10.02034149057392,
            -16.560806018792643,
            -9.116062809320225,
            -10.939132732453045,
            -8.512671934773307,
            -9.339555694525526,
            -11.771296774029446,
            -10.949264659868488,
            -9.2
        ],
        'Activity': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def training_data(sample_molecular_data):
    """Training subset of molecular data (first 6 compounds)."""
    return sample_molecular_data.iloc[:6].copy()


@pytest.fixture
def prediction_data(sample_molecular_data):
    """Prediction subset of molecular data (last 4 compounds)."""
    return sample_molecular_data.iloc[6:].copy()


@pytest.fixture
def test_results_dir():
    """Test directory for results and caching."""
    test_dir = Path(__file__).parent / "test_representations"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture(params=['morgan', 'descriptors'])
def featurizer_type(request):
    """Parametrized fixture for different featurizer types."""
    return request.param


@pytest.fixture(params=['ESSENCE-Dock_Score'])
def target_column(request):
    """Parametrized fixture for different target columns."""
    return request.param


@pytest.fixture(autouse=True)
def setup_molecular_representations(sample_molecular_data, test_results_dir):
    """Pre-compute molecular representations for all test data."""
    # Pre-compute Morgan fingerprints
    precompute_representations(
        sample_molecular_data, 
        "morgan", 
        test_results_dir
    )
    
    # Pre-compute Mordred descriptors
    precompute_representations(
        sample_molecular_data, 
        "descriptors", 
        test_results_dir
    )
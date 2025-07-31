"""Pytest configuration and fixtures for LearnM8 tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


@pytest.fixture
def sample_compounds() -> pd.DataFrame:
    """Create sample compounds DataFrame for testing."""
    np.random.seed(42)
    n_compounds = 100
    
    compounds = pd.DataFrame({
        'ID': [f'COMP_{i:04d}' for i in range(n_compounds)],
        'SMILES': [f'C1CCCCC1{"N" if i % 3 == 0 else "O"}{"Cl" if i % 7 == 0 else ""}' 
                  for i in range(n_compounds)],
    })
    
    return compounds


@pytest.fixture
def sample_predictions() -> np.ndarray:
    """Create sample predictions for testing."""
    np.random.seed(42)
    return np.random.normal(loc=5.0, scale=2.0, size=100)


@pytest.fixture
def sample_uncertainties() -> np.ndarray:
    """Create sample uncertainties for testing."""
    np.random.seed(42)
    return np.random.exponential(scale=0.5, size=100)


@pytest.fixture
def large_compound_pool() -> pd.DataFrame:
    """Create larger compound pool for performance testing."""
    np.random.seed(42)
    n_compounds = 1000
    
    compounds = pd.DataFrame({
        'ID': [f'LARGE_{i:06d}' for i in range(n_compounds)],
        'SMILES': [f'c1ccc{"n" if i % 5 == 0 else "c"}cc1{"Br" if i % 11 == 0 else ""}' 
                  for i in range(n_compounds)],
    })
    
    return compounds


@pytest.fixture
def large_predictions() -> np.ndarray:
    """Create predictions for large compound pool."""
    np.random.seed(42)
    return np.random.beta(a=2, b=5, size=1000) * 10


@pytest.fixture
def large_uncertainties() -> np.ndarray:
    """Create uncertainties for large compound pool."""
    np.random.seed(42)
    return np.random.gamma(shape=1.5, scale=0.3, size=1000)


@pytest.fixture
def performance_data_sequence() -> List[Dict[str, Any]]:
    """Create sequence of performance data for adaptation testing."""
    return [
        {
            'improvement_rate': 0.05,
            'selection_diversity': 0.7,
            'oracle_efficiency': 0.8,
            'cycle': 1
        },
        {
            'improvement_rate': 0.08,
            'selection_diversity': 0.75,
            'oracle_efficiency': 0.85,
            'cycle': 2
        },
        {
            'improvement_rate': 0.03,
            'selection_diversity': 0.6,
            'oracle_efficiency': 0.7,
            'cycle': 3
        },
        {
            'improvement_rate': -0.02,
            'selection_diversity': 0.5,
            'oracle_efficiency': 0.6,
            'cycle': 4
        },
        {
            'improvement_rate': 0.12,
            'selection_diversity': 0.9,
            'oracle_efficiency': 0.95,
            'cycle': 5
        }
    ]


@pytest.fixture
def declining_performance_sequence() -> List[Dict[str, Any]]:
    """Create declining performance sequence for edge case testing."""
    return [
        {
            'improvement_rate': 0.1,
            'selection_diversity': 0.8,
            'oracle_efficiency': 0.9,
            'cycle': 1
        },
        {
            'improvement_rate': 0.05,
            'selection_diversity': 0.7,
            'oracle_efficiency': 0.8,
            'cycle': 2
        },
        {
            'improvement_rate': -0.05,
            'selection_diversity': 0.5,
            'oracle_efficiency': 0.6,
            'cycle': 3
        },
        {
            'improvement_rate': -0.15,
            'selection_diversity': 0.3,
            'oracle_efficiency': 0.4,
            'cycle': 4
        }
    ]


@pytest.fixture
def rapid_change_performance() -> List[Dict[str, Any]]:
    """Create rapid performance changes for edge case testing."""
    return [
        {'improvement_rate': 0.1, 'selection_diversity': 0.8, 'oracle_efficiency': 0.9},
        {'improvement_rate': -0.2, 'selection_diversity': 0.2, 'oracle_efficiency': 0.3},
        {'improvement_rate': 0.3, 'selection_diversity': 0.9, 'oracle_efficiency': 0.95},
        {'improvement_rate': -0.1, 'selection_diversity': 0.4, 'oracle_efficiency': 0.5},
        {'improvement_rate': 0.25, 'selection_diversity': 0.85, 'oracle_efficiency': 0.9}
    ]


@pytest.fixture
def stagnant_performance_sequence() -> List[float]:
    """Create stagnant performance sequence for PerformanceBasedPruner."""
    return [0.5, 0.51, 0.49, 0.5, 0.52, 0.48, 0.5]


@pytest.fixture
def improving_performance_sequence() -> List[float]:
    """Create improving performance sequence for PerformanceBasedPruner."""
    return [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]


@pytest.fixture
def declining_performance_values() -> List[float]:
    """Create declining performance values for PerformanceBasedPruner."""
    return [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25]


@pytest.fixture
def empty_compounds() -> pd.DataFrame:
    """Create empty compounds DataFrame for edge case testing."""
    return pd.DataFrame(columns=['ID', 'SMILES'])


@pytest.fixture
def invalid_compounds() -> pd.DataFrame:
    """Create compounds DataFrame missing required columns."""
    return pd.DataFrame({
        'compound_id': ['COMP_001', 'COMP_002'],
        'structure': ['CCO', 'CCC']
    })


@pytest.fixture
def compounds_with_nan_predictions() -> tuple:
    """Create compounds with NaN predictions for error testing."""
    compounds = pd.DataFrame({
        'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
        'SMILES': ['CCO', 'CCC', 'CCN']
    })
    predictions = np.array([1.0, np.nan, 3.0])
    return compounds, predictions


@pytest.fixture
def mismatched_data() -> tuple:
    """Create mismatched compounds and predictions for error testing."""
    compounds = pd.DataFrame({
        'ID': ['COMP_001', 'COMP_002'],
        'SMILES': ['CCO', 'CCC']
    })
    predictions = np.array([1.0, 2.0, 3.0])  # Length mismatch
    return compounds, predictions


@pytest.fixture
def performance_metric_names() -> List[str]:
    """List of valid performance metric names."""
    return ['improvement_rate', 'diversity', 'efficiency']


@pytest.fixture
def adaptation_scenarios() -> Dict[str, Dict[str, Any]]:
    """Different adaptation scenarios for comprehensive testing."""
    return {
        'conservative': {
            'adaptation_rate': 0.05,
            'min_retention_fraction': 0.5,
            'max_retention_fraction': 0.9
        },
        'aggressive': {
            'adaptation_rate': 0.2,
            'min_retention_fraction': 0.1,
            'max_retention_fraction': 0.8
        },
        'balanced': {
            'adaptation_rate': 0.1,
            'min_retention_fraction': 0.3,
            'max_retention_fraction': 0.8
        }
    }


# ==============================================================================
# Real Molecular Data Fixtures
# ==============================================================================

def _load_test_data(filename: str) -> pd.DataFrame:
    """Helper function to load test data files with proper error handling."""
    test_data_dir = Path(__file__).parent / "data"
    data_file = test_data_dir / filename
    
    if not data_file.exists():
        # Fallback to empty DataFrame with proper columns
        return pd.DataFrame(columns=['ID', 'SMILES', 'Activity'])
    
    try:
        return pd.read_csv(data_file)
    except Exception as e:
        pytest.skip(f"Could not load test data {filename}: {e}")


@pytest.fixture
def small_real_compounds() -> pd.DataFrame:
    """50 real pharmaceutical compounds from ESSENCE ADA dataset for fast unit tests.
    
    Contains: ID, SMILES, Activity columns with diverse activity range.
    Use for basic functionality testing where realistic molecular structures matter.
    """
    return _load_test_data("small_molecules.csv")


@pytest.fixture
def medium_real_compounds() -> pd.DataFrame:
    """200 real compounds from MAPK1 dataset for integration tests.
    
    Contains: ID, SMILES, Activity, Consensus_Score columns.
    Use for testing acquisition functions, evaluation metrics, and workflow integration.
    """
    return _load_test_data("medium_molecules.csv")


@pytest.fixture
def diverse_real_compounds() -> pd.DataFrame:
    """100 structurally diverse compounds across multiple targets for diversity testing.
    
    Contains: ID, SMILES, Activity, Target columns from 5 different protein targets.
    Use for testing diversity-based acquisition and cross-target validation.
    """
    return _load_test_data("diverse_molecules.csv")


@pytest.fixture
def edge_case_compounds() -> pd.DataFrame:
    """20 real compounds with challenging molecular features for edge case testing.
    
    Contains: ID, SMILES, Activity, Edge_Case_Type columns.
    Includes salts, stereochemistry, charges, and other molecular edge cases.
    Use for error handling and robustness testing.
    """
    return _load_test_data("edge_case_molecules.csv")


@pytest.fixture
def multi_target_compounds() -> pd.DataFrame:
    """90 compounds from multiple targets for cross-validation testing.
    
    Contains: ID, SMILES, Activity, Target columns from ADA, CASP3, HIVPR.
    Use for testing generalization across different biological targets.
    """
    return _load_test_data("multi_target.csv")


@pytest.fixture
def real_compounds_with_predictions(small_real_compounds) -> tuple:
    """Real compounds with realistic predictions and uncertainties.
    
    Returns: (compounds_df, predictions, uncertainties)
    Use for testing acquisition functions that need both molecular data and ML outputs.
    """
    compounds = small_real_compounds.copy()
    
    if len(compounds) == 0:
        # Fallback to minimal synthetic data
        compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCN'],
            'Activity': [0.1, 0.5, 0.9]
        })
    
    # Generate realistic predictions based on actual activities if available
    if 'Activity' in compounds.columns:
        # Add some noise to actual activities for predictions
        np.random.seed(42)
        predictions = compounds['Activity'].values + np.random.normal(0, 0.1, len(compounds))
        predictions = np.clip(predictions, 0, 1)  # Keep in reasonable range
    else:
        np.random.seed(42)
        predictions = np.random.beta(2, 5, len(compounds))
    
    # Generate realistic uncertainties (higher for compounds with extreme predictions)
    uncertainties = 0.1 + 0.3 * np.abs(predictions - 0.5)  # Higher uncertainty for extreme values
    
    return compounds, predictions, uncertainties


@pytest.fixture
def regression_compounds() -> pd.DataFrame:
    """Real compounds with continuous activity values for regression testing.
    
    Uses medium dataset with normalized activity values suitable for regression tasks.
    """
    compounds = _load_test_data("medium_molecules.csv")
    
    if len(compounds) == 0:
        return pd.DataFrame(columns=['ID', 'SMILES', 'Activity'])
    
    # Ensure activity values are suitable for regression (continuous, reasonable range)
    if 'Activity' in compounds.columns:
        compounds = compounds.copy()
        compounds['Activity'] = pd.to_numeric(compounds['Activity'], errors='coerce')
        compounds = compounds.dropna(subset=['Activity'])
    
    return compounds


@pytest.fixture
def classification_compounds(diverse_real_compounds) -> pd.DataFrame:
    """Real compounds with binary activity classification for classification testing.
    
    Converts activity values to binary active/inactive labels.
    """
    compounds = diverse_real_compounds.copy()
    
    if len(compounds) == 0 or 'Activity' not in compounds.columns:
        return pd.DataFrame(columns=['ID', 'SMILES', 'Activity', 'Binary_Activity'])
    
    # Convert to binary classification (active/inactive)
    activity_median = compounds['Activity'].median()
    compounds['Binary_Activity'] = (compounds['Activity'] > activity_median).astype(int)
    
    return compounds


@pytest.fixture
def compounds_with_uncertainty(real_compounds_with_predictions) -> pd.DataFrame:
    """Real compounds with uncertainty estimates for uncertainty-based acquisition testing."""
    compounds, predictions, uncertainties = real_compounds_with_predictions
    
    compounds = compounds.copy()
    compounds['prediction'] = predictions
    compounds['uncertainty'] = uncertainties
    
    return compounds


@pytest.fixture  
def molecular_property_data(small_real_compounds) -> pd.DataFrame:
    """Real compounds with multiple molecular properties for property-based testing."""
    compounds = small_real_compounds.copy()
    
    if len(compounds) == 0:
        return pd.DataFrame(columns=['ID', 'SMILES', 'Activity'])
    
    # Add mock molecular properties based on SMILES length and composition
    # (In real usage, these would be computed by RDKit/Mordred)
    np.random.seed(42)
    n_compounds = len(compounds)
    
    compounds['molecular_weight'] = 150 + np.random.normal(100, 50, n_compounds)
    compounds['logp'] = np.random.normal(2.5, 1.5, n_compounds)
    compounds['num_rotatable_bonds'] = np.random.poisson(5, n_compounds)
    compounds['tpsa'] = 50 + np.random.exponential(50, n_compounds)
    
    return compounds
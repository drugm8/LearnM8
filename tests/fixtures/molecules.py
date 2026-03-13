"""Molecular data fixtures for LearnM8 tests.

This module provides fixtures for both synthetic and real molecular data:
- Synthetic data: Fast generation, simple structures for unit tests
- Real data: CSV-loaded pharmaceutical compounds for integration tests
- Derived data: Combinations of compounds with predictions/properties

Session-scoped fixtures are used for read-only CSV data (10-50x speedup).
"""

from pathlib import Path
from typing import Tuple
import pytest
import polars as pl
import numpy as np


def _load_test_data(filename: str) -> pl.DataFrame:
    """Helper function to load test data files with proper error handling."""
    test_data_dir = Path(__file__).parent.parent / "data"
    data_file = test_data_dir / filename

    if not data_file.exists():
        raise FileNotFoundError(
            f"Required test fixture not found: {data_file}\n"
            "Run: python3 tests/data/generate_fixtures.py"
        )

    try:
        return pl.read_csv(data_file)
    except Exception as e:
        raise RuntimeError(f"Could not load test data {filename}: {e}")


@pytest.fixture(scope='session')
def sample_compounds() -> pl.DataFrame:
    """~5 synthetic compounds with deterministic, structurally diverse SMILES.

    Session-scoped: loaded once per test session (or per xdist worker).
    Safe because Polars DataFrames are immutable.
    """
    return pl.DataFrame({
        'ID': ['COMP_000', 'COMP_001', 'COMP_002', 'COMP_003', 'COMP_004'],
        'SMILES': ['C1CCCCC1N', 'C1CCCCC1O', 'C1CCCCC1NCl', 'C1CCCCC1OCl', 'C1CCCCC1N'],
    })


@pytest.fixture
def large_compound_pool() -> pl.DataFrame:
    """Create larger compound pool for performance testing."""
    np.random.seed(42)
    n_compounds = 1000

    compounds = pl.DataFrame({
        'ID': [f'LARGE_{i:06d}' for i in range(n_compounds)],
        'SMILES': [f'c1ccc{"n" if i % 5 == 0 else "c"}cc1{"Br" if i % 11 == 0 else ""}'
                  for i in range(n_compounds)],
    })

    return compounds


@pytest.fixture
def empty_compounds() -> pl.DataFrame:
    """Create empty compounds DataFrame for edge case testing."""
    return pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8})


@pytest.fixture
def invalid_compounds() -> pl.DataFrame:
    """Create compounds DataFrame missing required columns."""
    return pl.DataFrame({
        'compound_id': ['COMP_001', 'COMP_002'],
        'structure': ['CCO', 'CCC']
    })


@pytest.fixture
def compounds_with_nan_predictions() -> tuple:
    """Create compounds with NaN predictions for error testing."""
    compounds = pl.DataFrame({
        'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
        'SMILES': ['CCO', 'CCC', 'CCN']
    })
    predictions = np.array([1.0, np.nan, 3.0])
    return compounds, predictions


@pytest.fixture
def mismatched_data() -> tuple:
    """Create mismatched compounds and predictions for error testing."""
    compounds = pl.DataFrame({
        'ID': ['COMP_001', 'COMP_002'],
        'SMILES': ['CCO', 'CCC']
    })
    predictions = np.array([1.0, 2.0, 3.0])  # Length mismatch
    return compounds, predictions


@pytest.fixture(scope='session')
def small_real_compounds() -> pl.DataFrame:
    """50 real pharmaceutical compounds from ESSENCE ADA dataset for fast unit tests.

    Contains: ID, SMILES, Activity columns with diverse activity range.
    Use for basic functionality testing where realistic molecular structures matter.

    Session-scoped for performance: Loaded once per test session, reused across tests.
    Safe because Polars DataFrames are immutable.
    """
    return _load_test_data("small_molecules.csv")


@pytest.fixture(scope='session')
def medium_real_compounds() -> pl.DataFrame:
    """200 real compounds from MAPK1 dataset for integration tests.

    Contains: ID, SMILES, Activity, Consensus_Score columns.
    Use for testing acquisition functions, evaluation metrics, and workflow integration.

    Session-scoped for performance optimization.
    """
    return _load_test_data("medium_molecules.csv")


@pytest.fixture(scope='session')
def diverse_real_compounds() -> pl.DataFrame:
    """100 structurally diverse compounds across multiple targets for diversity testing.

    Contains: ID, SMILES, Activity, Target columns from 5 different protein targets.
    Use for testing diversity-based acquisition and cross-target validation.

    Session-scoped for performance optimization.
    """
    return _load_test_data("diverse_molecules.csv")


@pytest.fixture
def edge_case_compounds() -> pl.DataFrame:
    """20 real compounds with challenging molecular features for edge case testing.

    Contains: ID, SMILES, Activity, Edge_Case_Type columns.
    Includes salts, stereochemistry, charges, and other molecular edge cases.
    Use for error handling and robustness testing.
    """
    return _load_test_data("edge_case_molecules.csv")


@pytest.fixture
def valid_edge_case_compounds() -> pl.DataFrame:
    """Valid edge case compounds only (filters out intentionally invalid SMILES).

    Contains valid compounds with unusual but chemically correct structures.
    Use for testing feature extraction and workflow with edge cases.
    """
    df = _load_test_data("edge_case_molecules.csv")
    return df.filter(~pl.col('Edge_Case_Type').str.starts_with('invalid_smiles'))


@pytest.fixture
def multi_target_compounds() -> pl.DataFrame:
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
    compounds = small_real_compounds.clone()

    if len(compounds) == 0:
        # Fallback to minimal synthetic data
        compounds = pl.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCN'],
            'Activity': [0.1, 0.5, 0.9]
        })

    # Generate realistic predictions based on actual activities if available
    if 'Activity' in compounds.columns:
        # Add some noise to actual activities for predictions
        np.random.seed(42)
        predictions = compounds.get_column('Activity').to_numpy() + np.random.normal(0, 0.1, len(compounds))
        predictions = np.clip(predictions, 0, 1)  # Keep in reasonable range
    else:
        np.random.seed(42)
        predictions = np.random.beta(2, 5, len(compounds))

    # Generate realistic uncertainties (higher for compounds with extreme predictions)
    uncertainties = 0.1 + 0.3 * np.abs(predictions - 0.5)  # Higher uncertainty for extreme values

    return compounds, predictions, uncertainties


@pytest.fixture
def regression_compounds() -> pl.DataFrame:
    """Real compounds with continuous activity values for regression testing.

    Uses medium dataset with normalized activity values suitable for regression tasks.
    """
    compounds = _load_test_data("medium_molecules.csv")

    if len(compounds) == 0:
        return pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8, 'Activity': pl.Float64})

    # Ensure activity values are suitable for regression (continuous, reasonable range)
    if 'Activity' in compounds.columns:
        compounds = compounds.with_columns(
            pl.col('Activity').cast(pl.Float64, strict=False)
        ).filter(pl.col('Activity').is_not_null())

    return compounds


@pytest.fixture
def classification_compounds(diverse_real_compounds) -> pl.DataFrame:
    """Real compounds with binary activity classification for classification testing.

    Converts activity values to binary active/inactive labels.
    """
    compounds = diverse_real_compounds.clone()

    if len(compounds) == 0 or 'Activity' not in compounds.columns:
        return pl.DataFrame(schema={
            'ID': pl.Utf8,
            'SMILES': pl.Utf8,
            'Activity': pl.Float64,
            'Binary_Activity': pl.Int64
        })

    # Convert to binary classification (active/inactive)
    activity_median = compounds.get_column('Activity').median()
    compounds = compounds.with_columns(
        (pl.col('Activity') > activity_median).cast(pl.Int64).alias('Binary_Activity')
    )

    return compounds


@pytest.fixture
def compounds_with_uncertainty(real_compounds_with_predictions) -> pl.DataFrame:
    """Real compounds with uncertainty estimates for uncertainty-based acquisition testing."""
    compounds, predictions, uncertainties = real_compounds_with_predictions

    compounds = compounds.with_columns([
        pl.lit(predictions).alias('prediction'),
        pl.lit(uncertainties).alias('uncertainty')
    ])

    return compounds


@pytest.fixture
def molecular_property_data(small_real_compounds) -> pl.DataFrame:
    """Real compounds with multiple molecular properties for property-based testing."""
    compounds = small_real_compounds.clone()

    if len(compounds) == 0:
        return pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8, 'Activity': pl.Float64})

    # Add mock molecular properties based on SMILES length and composition
    # (In real usage, these would be computed by RDKit/Mordred)
    np.random.seed(42)
    n_compounds = len(compounds)

    compounds = compounds.with_columns([
        pl.lit(150 + np.random.normal(100, 50, n_compounds)).alias('molecular_weight'),
        pl.lit(np.random.normal(2.5, 1.5, n_compounds)).alias('logp'),
        pl.lit(np.random.poisson(5, n_compounds)).alias('num_rotatable_bonds'),
        pl.lit(50 + np.random.exponential(50, n_compounds)).alias('tpsa')
    ])

    return compounds

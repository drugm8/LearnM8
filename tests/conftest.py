"""Pytest configuration and fixtures for LearnM8 tests."""

import logging
import pytest
import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from learnm8.core.interfaces import Learner, Oracle
from learnm8.features import extract_features


@pytest.fixture
def sample_compounds() -> pl.DataFrame:
    """Create sample compounds DataFrame for testing."""
    np.random.seed(42)
    n_compounds = 100

    compounds = pl.DataFrame({
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

def _load_test_data(filename: str) -> pl.DataFrame:
    """Helper function to load test data files with proper error handling."""
    test_data_dir = Path(__file__).parent / "data"
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


@pytest.fixture
def small_real_compounds() -> pl.DataFrame:
    """50 real pharmaceutical compounds from ESSENCE ADA dataset for fast unit tests.

    Contains: ID, SMILES, Activity columns with diverse activity range.
    Use for basic functionality testing where realistic molecular structures matter.
    """
    return _load_test_data("small_molecules.csv")


@pytest.fixture
def medium_real_compounds() -> pl.DataFrame:
    """200 real compounds from MAPK1 dataset for integration tests.

    Contains: ID, SMILES, Activity, Consensus_Score columns.
    Use for testing acquisition functions, evaluation metrics, and workflow integration.
    """
    return _load_test_data("medium_molecules.csv")


@pytest.fixture
def diverse_real_compounds() -> pl.DataFrame:
    """100 structurally diverse compounds across multiple targets for diversity testing.

    Contains: ID, SMILES, Activity, Target columns from 5 different protein targets.
    Use for testing diversity-based acquisition and cross-target validation.
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


# ==============================================================================
# Mock Classes for Testing
# ==============================================================================

class MockLearner(Learner):
    """Mock learner implementation for testing core interfaces."""

    def __init__(self, supports_uncertainty=False, fail_training=False, fail_prediction=False):
        self._trained = False
        self._supports_uncertainty = supports_uncertainty
        self._training_features = None
        self._fail_training = fail_training
        self._fail_prediction = fail_prediction

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Mock training implementation with featurizer-agnostic interface."""
        if self._fail_training:
            raise RuntimeError("Training failed (mock)")

        if len(features) == 0:
            raise ValueError("Cannot train on empty dataset")

        if features.shape[0] != targets.shape[0]:
            raise ValueError("Features and targets must have same length")

        self._trained = True
        self._training_features = features.copy()

    def predict(self, features: np.ndarray) -> tuple:
        """Mock prediction implementation with featurizer-agnostic interface."""
        if self._fail_prediction:
            raise RuntimeError("Prediction failed (mock)")

        if not self._trained:
            raise RuntimeError("Model must be trained before prediction")

        if len(features) == 0:
            return np.array([]), None

        np.random.seed(42)
        predictions = np.random.uniform(0, 1, len(features))

        if self._supports_uncertainty:
            uncertainties = np.random.uniform(0.1, 0.3, len(features))
            return predictions, uncertainties
        else:
            return predictions, None

    def supports_uncertainty(self) -> bool:
        """Return whether this learner supports uncertainty estimation."""
        return self._supports_uncertainty

    def get_name(self) -> str:
        """Return the name of this mock learner."""
        uncertainty_suffix = "_with_uncertainty" if self._supports_uncertainty else ""
        return f"MockLearner{uncertainty_suffix}"


class MockOracle(Oracle):
    """Mock oracle implementation for testing core interfaces."""

    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level
        self.call_count = 0

    def measure(self, compounds: pl.DataFrame, properties: list) -> pl.DataFrame:
        """Mock measurement implementation using pure Polars."""
        self.call_count += 1

        if len(compounds) == 0:
            schema = {'ID': pl.Utf8, **{prop: pl.Float64 for prop in properties}}
            return pl.DataFrame(schema=schema)

        result = compounds.select(['ID'])

        # Generate mock measurements for each property
        for prop in properties:
            np.random.seed(42 + hash(prop) % 1000)

            if prop == 'Activity':
                # Generate activity values based on SMILES hash for consistency
                activities = []
                for smiles in compounds['SMILES'].to_list():
                    base_value = (hash(smiles) % 1000) / 1000.0
                    noise = np.random.normal(0, self.noise_level)
                    activities.append(base_value + noise)
                result = result.with_columns(pl.Series(prop, activities))
            else:
                # Generic property
                result = result.with_columns(
                    pl.Series(prop, np.random.uniform(0, 1, len(compounds)))
                )

        return result


# ==============================================================================
# Mock Object Fixtures
# ==============================================================================

@pytest.fixture
def mock_learner():
    """Create a MockLearner instance without uncertainty support."""
    return MockLearner(supports_uncertainty=False)


@pytest.fixture
def mock_learner_with_uncertainty():
    """Create a MockLearner instance with uncertainty support."""
    return MockLearner(supports_uncertainty=True)


@pytest.fixture
def mock_oracle():
    """Create a MockOracle instance with default noise level."""
    return MockOracle(noise_level=0.1)


@pytest.fixture
def mock_oracle_low_noise():
    """Create a MockOracle instance with low noise level."""
    return MockOracle(noise_level=0.01)


@pytest.fixture
def mock_oracle_high_noise():
    """Create a MockOracle instance with high noise level."""
    return MockOracle(noise_level=0.5)


# ==============================================================================
# Master DataFrame Fixtures
# ==============================================================================

def create_initialized_master_df(
    valid_compounds: pl.DataFrame,
    target_col: str,
    initial_labeled_ids: Optional[List[str]] = None,
    initial_measurements: Optional[dict] = None
) -> pl.DataFrame:
    """Helper to create master DataFrame with initial labeled compounds.

    This helper mimics the old initialize_master_dataframe behavior for test compatibility.
    It creates an empty master DataFrame and then manually labels initial compounds.

    Args:
        valid_compounds: Polars DataFrame with ID and SMILES columns
        target_col: Name of target column
        initial_labeled_ids: Optional list of compound IDs to label initially
        initial_measurements: Optional dict mapping IDs to measurements

    Returns:
        Master DataFrame with initial compounds labeled
    """
    from learnm8.core.initialization import initialize_master_dataframe_empty
    from learnm8.core.dataframe_ops import update_status

    master_df = initialize_master_dataframe_empty(valid_compounds, target_col)

    if initial_labeled_ids and initial_measurements is not None and len(initial_labeled_ids) > 0:
        master_df = update_status(
            df=master_df,
            compound_ids=initial_labeled_ids,
            new_status='labeled',
            cycle=-1,
            target_col=target_col,
            target_values=initial_measurements
        )

    return master_df


@pytest.fixture
def sample_master_df(sample_compounds) -> pl.DataFrame:
    """Create master DataFrame with 3 labeled compounds (indices 0-2).

    Uses sample_compounds fixture as base.
    """
    compounds = sample_compounds.clone()

    initial_compounds = compounds.head(3)
    initial_ids = initial_compounds.get_column('ID').to_list()
    initial_values = dict(zip(initial_ids, [0.3, 0.6, 0.9]))

    return create_initialized_master_df(
        valid_compounds=compounds,
        target_col='Activity',
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values
    )


@pytest.fixture
def master_df_with_predictions(sample_master_df) -> pl.DataFrame:
    """Create master DataFrame with cycle 0 predictions.

    Adds predictions for unlabeled compounds (indices 3-10) with both
    predictions and uncertainties.
    """
    from learnm8.core.dataframe_ops import add_predictions

    master_df = sample_master_df.clone()

    # Add predictions for unlabeled compounds
    unlabeled = master_df.filter(pl.col('status') == 'unlabeled').head(8)
    unlabeled_ids = unlabeled.get_column('ID').to_list()

    np.random.seed(42)
    predictions = np.random.uniform(0.2, 0.8, len(unlabeled_ids))
    uncertainties = np.random.uniform(0.1, 0.3, len(unlabeled_ids))

    return add_predictions(
        df=master_df,
        cycle=0,
        compound_ids=unlabeled_ids,
        predictions=predictions,
        uncertainties=uncertainties
    )


@pytest.fixture
def master_df_multi_cycle(sample_master_df) -> pl.DataFrame:
    """Create master DataFrame with 3 cycles of predictions.

    Simulates 3 cycles with predictions, uncertainties, and status updates
    for compounds labeled in cycles 0, 1, 2.
    """
    from learnm8.core.dataframe_ops import add_predictions, update_status

    master_df = sample_master_df.clone()

    # Cycle 0
    unlabeled = master_df.filter(pl.col('status') == 'unlabeled').head(8)
    unlabeled_ids = unlabeled.get_column('ID').to_list()

    np.random.seed(42)
    predictions_0 = np.random.uniform(0.2, 0.8, len(unlabeled_ids))
    uncertainties_0 = np.random.uniform(0.1, 0.3, len(unlabeled_ids))

    master_df = add_predictions(
        df=master_df,
        cycle=0,
        compound_ids=unlabeled_ids,
        predictions=predictions_0,
        uncertainties=uncertainties_0
    )

    # Label 2 compounds in cycle 0
    selected_ids_0 = unlabeled_ids[:2]
    target_values_0 = dict(zip(selected_ids_0, [0.45, 0.55]))
    master_df = update_status(
        df=master_df,
        compound_ids=selected_ids_0,
        new_status='labeled',
        cycle=0,
        target_col='Activity',
        target_values=target_values_0
    )

    # Cycle 1
    unlabeled = master_df.filter(pl.col('status') == 'unlabeled').head(6)
    unlabeled_ids = unlabeled.get_column('ID').to_list()

    predictions_1 = np.random.uniform(0.3, 0.9, len(unlabeled_ids))
    uncertainties_1 = np.random.uniform(0.1, 0.25, len(unlabeled_ids))

    master_df = add_predictions(
        df=master_df,
        cycle=1,
        compound_ids=unlabeled_ids,
        predictions=predictions_1,
        uncertainties=uncertainties_1
    )

    # Label 2 compounds in cycle 1
    selected_ids_1 = unlabeled_ids[:2]
    target_values_1 = dict(zip(selected_ids_1, [0.65, 0.75]))
    master_df = update_status(
        df=master_df,
        compound_ids=selected_ids_1,
        new_status='labeled',
        cycle=1,
        target_col='Activity',
        target_values=target_values_1
    )

    # Cycle 2
    unlabeled = master_df.filter(pl.col('status') == 'unlabeled').head(4)
    unlabeled_ids = unlabeled.get_column('ID').to_list()

    predictions_2 = np.random.uniform(0.4, 0.85, len(unlabeled_ids))
    uncertainties_2 = np.random.uniform(0.1, 0.2, len(unlabeled_ids))

    master_df = add_predictions(
        df=master_df,
        cycle=2,
        compound_ids=unlabeled_ids,
        predictions=predictions_2,
        uncertainties=uncertainties_2
    )

    return master_df


@pytest.fixture
def caplog_info(caplog):
    """Capture INFO level logs."""
    caplog.set_level(logging.INFO)
    return caplog


@pytest.fixture
def caplog_debug(caplog):
    """Capture DEBUG level logs."""
    caplog.set_level(logging.DEBUG)
    return caplog
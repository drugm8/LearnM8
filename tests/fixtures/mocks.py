"""Mock object fixtures for LearnM8 tests.

Provides mock implementations of Learner and Oracle protocols for testing
without requiring actual ML libraries or expensive computations.

All fixtures are session-scoped (stateless objects safe to reuse).
"""

from typing import Tuple
import pytest
import polars as pl
import numpy as np
from learnm8.core.interfaces import Learner, Oracle


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


@pytest.fixture
def make_mock_learner():
    """Factory for fresh MockLearner instances. Each call returns clean state."""
    def _make(supports_uncertainty=False, fail_training=False, fail_prediction=False):
        return MockLearner(
            supports_uncertainty=supports_uncertainty,
            fail_training=fail_training,
            fail_prediction=fail_prediction,
        )
    return _make


@pytest.fixture
def make_mock_oracle():
    """Factory for fresh MockOracle instances. Each call returns call_count=0."""
    def _make(noise=0.1):
        return MockOracle(noise_level=noise)
    return _make


@pytest.fixture
def mock_learner(make_mock_learner):
    """DEPRECATED: Use make_mock_learner() directly. Will be removed in v0.12.0."""
    import warnings
    warnings.warn("mock_learner fixture is deprecated; use make_mock_learner()", DeprecationWarning, stacklevel=2)
    return make_mock_learner()


@pytest.fixture
def mock_learner_with_uncertainty(make_mock_learner):
    """DEPRECATED: Use make_mock_learner(supports_uncertainty=True). Will be removed in v0.12.0."""
    import warnings
    warnings.warn("mock_learner_with_uncertainty is deprecated; use make_mock_learner(supports_uncertainty=True)", DeprecationWarning, stacklevel=2)
    return make_mock_learner(supports_uncertainty=True)


@pytest.fixture
def mock_oracle(make_mock_oracle):
    """DEPRECATED: Use make_mock_oracle() directly. Will be removed in v0.12.0."""
    import warnings
    warnings.warn("mock_oracle fixture is deprecated; use make_mock_oracle()", DeprecationWarning, stacklevel=2)
    return make_mock_oracle()


@pytest.fixture
def mock_oracle_low_noise(make_mock_oracle):
    """DEPRECATED: Use make_mock_oracle(noise=0.01). Will be removed in v0.12.0."""
    import warnings
    warnings.warn("mock_oracle_low_noise is deprecated; use make_mock_oracle(noise=0.01)", DeprecationWarning, stacklevel=2)
    return make_mock_oracle(noise=0.01)


@pytest.fixture
def mock_oracle_high_noise(make_mock_oracle):
    """DEPRECATED: Use make_mock_oracle(noise=0.5). Will be removed in v0.12.0."""
    import warnings
    warnings.warn("mock_oracle_high_noise is deprecated; use make_mock_oracle(noise=0.5)", DeprecationWarning, stacklevel=2)
    return make_mock_oracle(noise=0.5)

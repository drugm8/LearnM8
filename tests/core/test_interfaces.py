"""
Interface compliance tests for core LearnM8 components.

Tests that all components properly implement their abstract interfaces.
"""

import pytest
import numpy as np
import polars as pl

from learnm8.core.interfaces import Learner, Oracle
from learnm8.features import extract_features
from learnm8.learners.sklearn import RandomForestLearner, GaussianProcessLearner
from learnm8.oracles.csv_oracle import CSVOracle


class MockLearner(Learner):
    """Mock learner for interface testing."""

    def __init__(self, supports_uncertainty=False):
        self._supports_uncertainty = supports_uncertainty
        self._is_trained = False

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Mock training implementation."""
        if len(features) == 0:
            raise ValueError("Cannot train on empty dataset")
        if len(targets) == 0:
            raise ValueError("Cannot train without targets")
        if len(features) != len(targets):
            raise ValueError("Features and targets must have same length")

        self._is_trained = True

    def predict(self, features: np.ndarray) -> tuple:
        """Mock prediction implementation."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")

        predictions = np.random.random(len(features))
        uncertainty = np.random.random(len(features)) if self._supports_uncertainty else None

        return predictions, uncertainty

    def supports_uncertainty(self) -> bool:
        """Return whether this learner supports uncertainty estimation."""
        return self._supports_uncertainty

    def get_name(self) -> str:
        """Return a descriptive name for this mock learner."""
        return f"MockLearner(uncertainty={self._supports_uncertainty})"


class MockOracle(Oracle):
    """Mock oracle for interface testing."""

    def __init__(self, fail_on_invalid=True):
        self.fail_on_invalid = fail_on_invalid
        self.measurement_count = 0

    def measure(self, compounds: pl.DataFrame, properties: list) -> pl.DataFrame:
        """Mock measurement implementation."""
        if len(compounds) == 0:
            return pl.DataFrame(schema={'ID': pl.Utf8, **{prop: pl.Float64 for prop in properties}})

        if 'ID' not in compounds.columns:
            raise KeyError("Compounds must have 'ID' column")

        result = compounds.select(['ID'])

        for prop in properties:
            # Simulate measurements
            result = result.with_columns(
                pl.Series(prop, np.random.uniform(-10, 0, len(compounds)))
            )

        self.measurement_count += len(compounds)
        return result


class TestLearnerInterface:
    """Test Learner interface compliance."""
    
    def test_learner_interface_methods(self, small_real_compounds, tmp_path):
        """Test that Learner interface methods are properly defined."""
        compounds = small_real_compounds.clone()
        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        targets = compounds['Activity'].to_numpy()

        learner = MockLearner(supports_uncertainty=True)

        assert hasattr(learner, 'train')
        assert hasattr(learner, 'predict')
        assert hasattr(learner, 'supports_uncertainty')

        learner.train(features, targets)

        predictions, uncertainty = learner.predict(features)

        assert len(predictions) == len(compounds)
        assert uncertainty is not None
        assert len(uncertainty) == len(compounds)

        assert learner.supports_uncertainty()
    
    def test_learner_without_uncertainty(self, small_real_compounds, tmp_path):
        """Test learner that doesn't support uncertainty."""
        compounds = small_real_compounds.clone()
        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        targets = compounds['Activity'].to_numpy()
        learner = MockLearner(supports_uncertainty=False)

        learner.train(features, targets)
        predictions, uncertainty = learner.predict(features)

        assert len(predictions) == len(compounds)
        assert uncertainty is None
        assert not learner.supports_uncertainty()
    
    def test_learner_error_conditions(self, tmp_path):
        """Test learner error handling."""
        learner = MockLearner()

        empty_features = np.array([]).reshape(0, 2048)
        empty_targets = np.array([])
        with pytest.raises(ValueError):
            learner.train(empty_features, empty_targets)

        smiles_list = ['CCO']
        features = extract_features(smiles_list, 'morgan', tmp_path)

        with pytest.raises(RuntimeError):
            learner.predict(features)
    
    def test_real_learner_interface_compliance(self, small_real_compounds, tmp_path):
        """Test that real learners comply with interface."""
        compounds = small_real_compounds.clone()
        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        targets = compounds['Activity'].to_numpy()

        rf_learner = RandomForestLearner()

        assert isinstance(rf_learner, Learner)
        assert hasattr(rf_learner, 'train')
        assert hasattr(rf_learner, 'predict')
        assert hasattr(rf_learner, 'supports_uncertainty')

        rf_learner.train(features, targets)
        predictions, uncertainty = rf_learner.predict(features)

        assert len(predictions) == len(compounds)
        assert isinstance(rf_learner.supports_uncertainty(), bool)

        gp_learner = GaussianProcessLearner()

        assert isinstance(gp_learner, Learner)
        gp_learner.train(features, targets)
        predictions, uncertainty = gp_learner.predict(features)

        assert len(predictions) == len(compounds)
        if gp_learner.supports_uncertainty():
            assert uncertainty is not None
            assert len(uncertainty) == len(compounds)


class TestOracleInterface:
    """Test Oracle interface compliance."""
    
    def test_oracle_interface_methods(self, small_real_compounds):
        """Test that Oracle interface methods are properly defined."""
        compounds = small_real_compounds.clone()
        oracle = MockOracle()

        assert hasattr(oracle, 'measure')

        properties = ['Activity', 'LogP']
        result = oracle.measure(compounds, properties)

        assert isinstance(result, pl.DataFrame)
        assert 'ID' in result.columns
        assert all(prop in result.columns for prop in properties)
        assert len(result) == len(compounds)
    
    def test_oracle_empty_input(self):
        """Test oracle with empty input."""
        oracle = MockOracle()
        empty_compounds = pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8})

        result = oracle.measure(empty_compounds, ['Activity'])

        assert isinstance(result, pl.DataFrame)
        assert 'ID' in result.columns
        assert 'Activity' in result.columns
        assert len(result) == 0
    
    def test_oracle_error_conditions(self, small_real_compounds):
        """Test oracle error handling."""
        compounds = small_real_compounds.clone()
        oracle = MockOracle()

        compounds_no_id = compounds.drop('ID')
        with pytest.raises(KeyError):
            oracle.measure(compounds_no_id, ['Activity'])
    
    def test_real_oracle_interface_compliance(self, small_real_compounds, tmp_path):
        """Test that real oracles comply with interface."""
        compounds = small_real_compounds.clone()
        test_csv = tmp_path / "test_data.csv"
        compounds.write_csv(test_csv)

        csv_oracle = CSVOracle(str(test_csv))

        assert isinstance(csv_oracle, Oracle)
        assert hasattr(csv_oracle, 'measure')

        result = csv_oracle.measure(compounds.head(5), ['Activity'])

        assert isinstance(result, pl.DataFrame)
        assert 'ID' in result.columns
        assert 'Activity' in result.columns
        assert len(result) == 5

    def test_csv_oracle_preserves_input_order(self, tmp_path):
        """Test that CSVOracle preserves input compound order."""
        oracle_data = pl.DataFrame({
            'ID': ['A', 'B', 'C', 'D', 'E'],
            'SMILES': ['CCO', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC'],
            'Activity': [0.1, 0.2, 0.3, 0.4, 0.5]
        })

        test_csv = tmp_path / "test_oracle.csv"
        oracle_data.write_csv(test_csv)

        oracle = CSVOracle(str(test_csv))

        request = pl.DataFrame({
            'ID': ['E', 'B', 'A', 'D', 'C'],
            'SMILES': ['CCCCCC', 'CCC', 'CCO', 'CCCCC', 'CCCC']
        })

        result = oracle.measure(request, ['Activity'])

        assert result['ID'].to_list() == ['E', 'B', 'A', 'D', 'C']
        assert result['Activity'].to_list() == [0.5, 0.2, 0.1, 0.4, 0.3]


class TestInterfaceIntegration:
    """Test integration between interface components."""
    
    def test_learner_oracle_integration(self, small_real_compounds, tmp_path):
        """Test learner and oracle working together."""
        compounds = small_real_compounds.clone()
        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        targets = compounds['Activity'].to_numpy()
        learner = MockLearner(supports_uncertainty=True)
        oracle = MockOracle()

        learner.train(features, targets)

        predictions, uncertainty = learner.predict(features)

        n_select = min(5, len(compounds))
        selected_indices = np.argsort(predictions)[-n_select:]
        selected_compounds = compounds[selected_indices]

        measurements = oracle.measure(selected_compounds, ['Activity'])

        assert len(measurements) == n_select
        assert 'ID' in measurements.columns
        assert 'Activity' in measurements.columns
    
    def test_interface_with_real_components(self, medium_real_compounds, tmp_path):
        """Test interface integration with real components."""
        compounds = medium_real_compounds.clone()

        if len(compounds) < 10:
            pytest.skip("Insufficient compounds for integration test")

        oracle_data = compounds.clone()
        test_csv = tmp_path / "oracle_data.csv"
        oracle_data.write_csv(test_csv)

        learner = RandomForestLearner()
        oracle = CSVOracle(str(test_csv))

        train_compounds = compounds.head(50)
        test_compounds = compounds.tail(20)

        train_features = extract_features(train_compounds['SMILES'].to_list(), 'morgan', tmp_path)
        train_targets = train_compounds['Activity'].to_numpy()
        learner.train(train_features, train_targets)

        test_features = extract_features(test_compounds['SMILES'].to_list(), 'morgan', tmp_path)
        predictions, uncertainty = learner.predict(test_features)

        assert len(predictions) == len(test_compounds)

        n_select = 5
        top_indices = np.argsort(predictions)[-n_select:]
        selected_compounds = test_compounds[top_indices]

        measurements = oracle.measure(selected_compounds, ['Activity'])

        assert len(measurements) == n_select
        assert set(measurements['ID'].to_list()) == set(selected_compounds['ID'].to_list())
    
    def test_interface_error_propagation(self, small_real_compounds, tmp_path):
        """Test error propagation through interface components."""
        compounds = small_real_compounds.clone()
        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        learner = MockLearner()
        oracle = MockOracle()

        targets_wrong_length = compounds['Activity'].to_numpy()[:len(compounds)//2]

        with pytest.raises(ValueError):
            learner.train(features, targets_wrong_length)

        targets = compounds['Activity'].to_numpy()
        learner.train(features, targets)

        compounds_no_id = compounds.drop('ID')

        with pytest.raises(KeyError):
            oracle.measure(compounds_no_id, ['Activity'])
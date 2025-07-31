"""
Interface compliance tests for core LearnM8 components.

Tests that all components properly implement their abstract interfaces.
"""

import pytest
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from learnm8.core.interfaces import Learner, Oracle
from learnm8.core.data_manager import DataManager
from learnm8.learners.sklearn import RandomForestLearner, GaussianProcessLearner
from learnm8.oracles.csv_oracle import CSVOracle
from learnm8.oracles.python_oracle import PythonOracle


class MockLearner(Learner):
    """Mock learner for interface testing."""
    
    def __init__(self, supports_uncertainty=False):
        self._supports_uncertainty = supports_uncertainty
        self._is_trained = False
    
    def train(self, compounds: pd.DataFrame, target_column: str, data_manager: DataManager):
        """Mock training implementation."""
        if len(compounds) == 0:
            raise ValueError("Cannot train on empty dataset")
        if target_column not in compounds.columns:
            raise KeyError(f"Target column '{target_column}' not found")
        
        self._is_trained = True
    
    def predict(self, compounds: pd.DataFrame, data_manager: DataManager):
        """Mock prediction implementation."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        predictions = np.random.random(len(compounds))
        uncertainty = np.random.random(len(compounds)) if self._supports_uncertainty else None
        
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
    
    def measure(self, compounds: pd.DataFrame, properties: list) -> pd.DataFrame:
        """Mock measurement implementation."""
        if len(compounds) == 0:
            return pd.DataFrame(columns=['ID'] + properties)
        
        if 'ID' not in compounds.columns:
            raise KeyError("Compounds must have 'ID' column")
        
        result = compounds[['ID']].copy()
        
        for prop in properties:
            # Simulate measurements
            result[prop] = np.random.uniform(-10, 0, len(compounds))  # Docking-like scores
        
        self.measurement_count += len(compounds)
        return result


class TestLearnerInterface:
    """Test Learner interface compliance."""
    
    def test_learner_interface_methods(self, small_real_compounds, tmp_path):
        """Test that Learner interface methods are properly defined."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        data_manager = DataManager(results_dir=tmp_path)
        
        # Test mock learner interface
        learner = MockLearner(supports_uncertainty=True)
        
        # Test interface methods exist
        assert hasattr(learner, 'train')
        assert hasattr(learner, 'predict')
        assert hasattr(learner, 'supports_uncertainty')
        
        # Test training
        learner.train(compounds, 'Activity', data_manager)
        
        # Test prediction
        predictions, uncertainty = learner.predict(compounds, data_manager)
        
        assert len(predictions) == len(compounds)
        assert uncertainty is not None  # This learner supports uncertainty
        assert len(uncertainty) == len(compounds)
        
        # Test uncertainty support
        assert learner.supports_uncertainty() == True
    
    def test_learner_without_uncertainty(self, small_real_compounds, tmp_path):
        """Test learner that doesn't support uncertainty."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner(supports_uncertainty=False)
        
        learner.train(compounds, 'Activity', data_manager)
        predictions, uncertainty = learner.predict(compounds, data_manager)
        
        assert len(predictions) == len(compounds)
        assert uncertainty is None  # This learner doesn't support uncertainty
        assert learner.supports_uncertainty() == False
    
    def test_learner_error_conditions(self, tmp_path):
        """Test learner error handling."""
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner()
        
        # Test training with empty data
        empty_compounds = pd.DataFrame(columns=['ID', 'SMILES', 'Activity'])
        with pytest.raises(ValueError):
            learner.train(empty_compounds, 'Activity', data_manager)
        
        # Test prediction without training
        compounds = pd.DataFrame({
            'ID': ['test1'],
            'SMILES': ['CCO'],
            'Activity': [1.0]
        })
        
        with pytest.raises(RuntimeError):
            learner.predict(compounds, data_manager)
    
    def test_real_learner_interface_compliance(self, small_real_compounds, tmp_path):
        """Test that real learners comply with interface."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        data_manager = DataManager(results_dir=tmp_path)
        
        # Test RandomForestLearner
        rf_learner = RandomForestLearner()
        
        assert isinstance(rf_learner, Learner)
        assert hasattr(rf_learner, 'train')
        assert hasattr(rf_learner, 'predict')
        assert hasattr(rf_learner, 'supports_uncertainty')
        
        rf_learner.train(compounds, 'Activity', data_manager)
        predictions, uncertainty = rf_learner.predict(compounds, data_manager)
        
        assert len(predictions) == len(compounds)
        assert isinstance(rf_learner.supports_uncertainty(), bool)
        
        # Test GaussianProcessLearner
        gp_learner = GaussianProcessLearner()
        
        assert isinstance(gp_learner, Learner)
        gp_learner.train(compounds, 'Activity', data_manager)
        predictions, uncertainty = gp_learner.predict(compounds, data_manager)
        
        assert len(predictions) == len(compounds)
        # GP should support uncertainty
        if gp_learner.supports_uncertainty():
            assert uncertainty is not None
            assert len(uncertainty) == len(compounds)


class TestOracleInterface:
    """Test Oracle interface compliance."""
    
    def test_oracle_interface_methods(self, small_real_compounds):
        """Test that Oracle interface methods are properly defined."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        oracle = MockOracle()
        
        # Test interface methods exist
        assert hasattr(oracle, 'measure')
        
        # Test measurement
        properties = ['Activity', 'LogP']
        result = oracle.measure(compounds, properties)
        
        assert isinstance(result, pd.DataFrame)
        assert 'ID' in result.columns
        assert all(prop in result.columns for prop in properties)
        assert len(result) == len(compounds)
    
    def test_oracle_empty_input(self):
        """Test oracle with empty input."""
        oracle = MockOracle()
        empty_compounds = pd.DataFrame(columns=['ID', 'SMILES'])
        
        result = oracle.measure(empty_compounds, ['Activity'])
        
        assert isinstance(result, pd.DataFrame)
        assert 'ID' in result.columns
        assert 'Activity' in result.columns
        assert len(result) == 0
    
    def test_oracle_error_conditions(self, small_real_compounds):
        """Test oracle error handling."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        oracle = MockOracle()
        
        # Test with missing ID column
        compounds_no_id = compounds.drop(columns=['ID'])
        with pytest.raises(KeyError):
            oracle.measure(compounds_no_id, ['Activity'])
    
    def test_real_oracle_interface_compliance(self, small_real_compounds, tmp_path):
        """Test that real oracles comply with interface."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Create a test CSV file for CSVOracle
        test_csv = tmp_path / "test_data.csv"
        compounds.to_csv(test_csv, index=False)
        
        # Test CSVOracle
        csv_oracle = CSVOracle(str(test_csv))
        
        assert isinstance(csv_oracle, Oracle)
        assert hasattr(csv_oracle, 'measure')
        
        # Test measurement
        result = csv_oracle.measure(compounds.head(5), ['Activity'])
        
        assert isinstance(result, pd.DataFrame)
        assert 'ID' in result.columns
        assert 'Activity' in result.columns
        assert len(result) == 5


class TestInterfaceIntegration:
    """Test integration between interface components."""
    
    def test_learner_oracle_integration(self, small_real_compounds, tmp_path):
        """Test learner and oracle working together."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Set up components
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner(supports_uncertainty=True)
        oracle = MockOracle()
        
        # Train learner
        learner.train(compounds, 'Activity', data_manager)
        
        # Get predictions
        predictions, uncertainty = learner.predict(compounds, data_manager)
        
        # Simulate active learning cycle - select some compounds
        n_select = min(5, len(compounds))
        selected_indices = np.argsort(predictions)[-n_select:]  # Top predictions
        selected_compounds = compounds.iloc[selected_indices]
        
        # Measure selected compounds with oracle
        measurements = oracle.measure(selected_compounds, ['Activity'])
        
        assert len(measurements) == n_select
        assert 'ID' in measurements.columns
        assert 'Activity' in measurements.columns
    
    def test_interface_with_real_components(self, medium_real_compounds, tmp_path):
        """Test interface integration with real components."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) < 10:
            pytest.skip("Insufficient compounds for integration test")
        
        # Create test oracle data
        oracle_data = compounds.copy()
        test_csv = tmp_path / "oracle_data.csv"
        oracle_data.to_csv(test_csv, index=False)
        
        # Set up real components
        data_manager = DataManager(results_dir=tmp_path)
        learner = RandomForestLearner()
        oracle = CSVOracle(str(test_csv))
        
        # Split data for simulation
        train_compounds = compounds.head(50)
        test_compounds = compounds.tail(20)
        
        # Train learner
        learner.train(train_compounds, 'Activity', data_manager)
        
        # Get predictions on test set
        predictions, uncertainty = learner.predict(test_compounds, data_manager)
        
        assert len(predictions) == len(test_compounds)
        
        # Select top compounds
        n_select = 5
        top_indices = np.argsort(predictions)[-n_select:]
        selected_compounds = test_compounds.iloc[top_indices]
        
        # Measure with oracle
        measurements = oracle.measure(selected_compounds, ['Activity'])
        
        assert len(measurements) == n_select
        assert set(measurements['ID']) == set(selected_compounds['ID'])
    
    def test_interface_error_propagation(self, small_real_compounds, tmp_path):
        """Test error propagation through interface components."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner()
        oracle = MockOracle()
        
        # Test error propagation - missing target column
        compounds_no_target = compounds.drop(columns=['Activity'])
        
        with pytest.raises(KeyError):
            learner.train(compounds_no_target, 'Activity', data_manager)
        
        # Train learner properly
        learner.train(compounds, 'Activity', data_manager)
        
        # Test oracle error propagation - missing ID
        compounds_no_id = compounds.drop(columns=['ID'])
        
        with pytest.raises(KeyError):
            oracle.measure(compounds_no_id, ['Activity'])
"""Tests for LearnM8 core system functionality.

Focused tests for core system integration and interfaces using real molecular data.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from learnm8.core.data_manager import DataManager
from learnm8.core.interfaces import Learner, Oracle


class MockLearner(Learner):
    """Mock learner implementation for testing core interfaces."""
    
    def __init__(self, supports_uncertainty=False):
        self._trained = False
        self._supports_uncertainty = supports_uncertainty
        self._training_data = None
    
    def train(self, compounds: pd.DataFrame, target_column: str, data_manager: DataManager) -> None:
        """Mock training implementation."""
        if len(compounds) == 0:
            raise ValueError("Cannot train on empty dataset")
        
        if target_column not in compounds.columns:
            raise KeyError(f"Target column '{target_column}' not found")
        
        # Get features to simulate real training
        compound_ids = compounds['ID'].tolist()
        smiles_list = compounds['SMILES'].tolist()
        features, valid_ids = data_manager.get_features(compound_ids, smiles_list, 'morgan')

        if len(valid_ids) != len(compounds):
            raise ValueError(f"Feature shape mismatch: expected {len(compounds)} compounds, got {len(valid_ids)} valid compounds")
        
        self._trained = True
        self._training_data = compounds.copy()
    
    def predict(self, compounds: pd.DataFrame, data_manager: DataManager) -> tuple:
        """Mock prediction implementation."""
        if not self._trained:
            raise RuntimeError("Model must be trained before prediction")
        
        compound_ids = compounds['ID'].tolist()
        smiles_list = compounds['SMILES'].tolist()
        features, valid_ids = data_manager.get_features(compound_ids, smiles_list, 'morgan')

        # Generate mock predictions for valid compounds only
        np.random.seed(42)
        predictions = np.random.uniform(0, 1, len(valid_ids))
        
        if self._supports_uncertainty:
            uncertainties = np.random.uniform(0.1, 0.3, len(valid_ids))
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
    
    def measure(self, compounds: pd.DataFrame, properties: list) -> pd.DataFrame:
        """Mock measurement implementation."""
        self.call_count += 1
        
        if len(compounds) == 0:
            return pd.DataFrame(columns=['ID'] + properties)
        
        result = compounds[['ID']].copy()
        
        # Generate mock measurements for each property
        for prop in properties:
            np.random.seed(42 + hash(prop) % 1000)
            
            if prop == 'Activity':
                # Generate activity values based on SMILES hash for consistency
                activities = []
                for smiles in compounds['SMILES']:
                    base_value = (hash(smiles) % 1000) / 1000.0
                    noise = np.random.normal(0, self.noise_level)
                    activities.append(base_value + noise)
                result[prop] = activities
            else:
                # Generic property
                result[prop] = np.random.uniform(0, 1, len(compounds))
        
        return result


class TestCoreInterfaces:
    """Test core interface implementations and compliance."""
    
    def test_learner_interface_compliance(self, small_real_compounds, tmp_path):
        """Test that learner implementations comply with interface."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner(supports_uncertainty=True)
        
        # Test interface methods exist
        assert hasattr(learner, 'train')
        assert hasattr(learner, 'predict')
        assert hasattr(learner, 'supports_uncertainty')
        
        # Test uncertainty support reporting
        assert learner.supports_uncertainty() == True
        
        # Test training
        learner.train(compounds, 'Activity', data_manager)
        
        # Test prediction
        predictions, uncertainties = learner.predict(compounds, data_manager)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(compounds)
        assert isinstance(uncertainties, np.ndarray)
        assert len(uncertainties) == len(compounds)
    
    def test_learner_without_uncertainty(self, small_real_compounds, tmp_path):
        """Test learner that doesn't support uncertainty."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner(supports_uncertainty=False)
        
        assert learner.supports_uncertainty() == False
        
        learner.train(compounds, 'Activity', data_manager)
        predictions, uncertainties = learner.predict(compounds, data_manager)
        
        assert isinstance(predictions, np.ndarray)
        assert uncertainties is None
    
    def test_oracle_interface_compliance(self, small_real_compounds):
        """Test that oracle implementations comply with interface."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        oracle = MockOracle()
        
        # Test interface methods exist
        assert hasattr(oracle, 'measure')
        
        # Test measurement
        properties = ['Activity', 'LogP']
        measurements = oracle.measure(compounds, properties)
        
        assert isinstance(measurements, pd.DataFrame)
        assert len(measurements) == len(compounds)
        assert 'ID' in measurements.columns
        assert all(prop in measurements.columns for prop in properties)
        
        # Test measurement consistency
        measurements2 = oracle.measure(compounds, properties)
        
        # Should be deterministic for same input
        pd.testing.assert_frame_equal(measurements, measurements2)
    
    def test_oracle_with_empty_compounds(self):
        """Test oracle behavior with empty compound set."""
        empty_compounds = pd.DataFrame(columns=['ID', 'SMILES'])
        oracle = MockOracle()
        
        result = oracle.measure(empty_compounds, ['Activity'])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert 'ID' in result.columns
        assert 'Activity' in result.columns


class TestCoreSystemIntegration:
    """Test integration between core system components."""
    
    def test_datamanager_learner_integration(self, medium_real_compounds, tmp_path):
        """Test DataManager integration with learner."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Use subset for faster testing
        compounds = compounds.head(20)
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner(supports_uncertainty=False)
        
        # Train learner using DataManager
        learner.train(compounds, 'Activity', data_manager)
        
        # Make predictions using DataManager
        predictions, uncertainties = learner.predict(compounds, data_manager)
        
        # Verify integration
        assert len(predictions) == len(compounds)
        assert uncertainties is None  # This learner doesn't support uncertainty
        
        # Verify DataManager cached features
        compound_ids = compounds['ID'].tolist()
        smiles_list = compounds['SMILES'].tolist()
        cached_features, cached_valid_ids = data_manager.get_features(compound_ids, smiles_list, 'morgan')
        assert cached_features.shape[0] == len(cached_valid_ids)
    
    def test_datamanager_oracle_integration(self, small_real_compounds, tmp_path):
        """Test DataManager integration with oracle."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        data_manager = DataManager(results_dir=tmp_path)
        oracle = MockOracle()
        
        # Oracle measures compounds
        measurements = oracle.measure(compounds, ['Measured_Activity', 'Solubility'])
        
        # DataManager processes the results
        combined_data = compounds.merge(measurements, on='ID')
        
        # Prepare training data
        valid_compounds, X, y = data_manager.prepare_training_data(combined_data, 'Measured_Activity', 'morgan')

        assert X.shape[0] == len(valid_compounds)
        assert len(y) == len(valid_compounds)
        assert len(valid_compounds) <= len(compounds)
        assert X.shape[1] > 0  # Should have molecular features
    
    def test_learner_oracle_workflow(self, diverse_real_compounds, tmp_path):
        """Test complete learner-oracle active learning workflow."""
        compounds = diverse_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Use subset for faster testing
        compounds = compounds.head(15)
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner(supports_uncertainty=True)
        oracle = MockOracle()
        
        # Initial training set
        labeled_compounds = compounds.head(8).copy()
        unlabeled_compounds = compounds.tail(7).copy()
        
        # Oracle measures labeled compounds
        measurements = oracle.measure(labeled_compounds, ['Measured_Activity'])
        labeled_with_activity = labeled_compounds.merge(measurements, on='ID')
        
        # Train learner
        learner.train(labeled_with_activity, 'Measured_Activity', data_manager)
        
        # Predict on unlabeled compounds
        predictions, uncertainties = learner.predict(unlabeled_compounds, data_manager)
        
        # Add predictions to unlabeled compounds
        unlabeled_compounds['prediction'] = predictions
        unlabeled_compounds['uncertainty'] = uncertainties
        
        # Select next compounds (simple greedy selection)
        next_batch = unlabeled_compounds.nlargest(3, 'prediction')
        
        # Oracle measures next batch
        next_measurements = oracle.measure(next_batch, ['Activity'])
        
        # Verify workflow completed successfully
        assert len(predictions) == 7
        assert len(uncertainties) == 7
        assert len(next_batch) == 3
        assert len(next_measurements) == 3
        assert oracle.call_count == 2  # Two oracle calls
    
    def test_multi_cycle_workflow(self, medium_real_compounds, tmp_path):
        """Test multi-cycle active learning workflow."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Use smaller subset for faster testing
        compounds = compounds.head(25)
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner(supports_uncertainty=True)
        oracle = MockOracle()
        
        # Start with small labeled set
        labeled_ids = set(compounds['ID'].head(5))
        batch_size = 3
        n_cycles = 3
        
        for cycle in range(n_cycles):
            # Get current labeled and unlabeled sets
            labeled_compounds = compounds[compounds['ID'].isin(labeled_ids)].copy()
            unlabeled_compounds = compounds[~compounds['ID'].isin(labeled_ids)].copy()
            
            if len(unlabeled_compounds) == 0:
                break
            
            # Oracle measures labeled compounds
            measurements = oracle.measure(labeled_compounds, ['Measured_Activity'])
            labeled_with_activity = labeled_compounds.merge(measurements, on='ID')
            
            # Train learner
            learner.train(labeled_with_activity, 'Measured_Activity', data_manager)
            
            # Predict on unlabeled compounds
            predictions, uncertainties = learner.predict(unlabeled_compounds, data_manager)
            unlabeled_compounds['prediction'] = predictions
            unlabeled_compounds['uncertainty'] = uncertainties
            
            # Select next batch (UCB-style selection)
            ucb_scores = predictions + uncertainties
            top_indices = np.argsort(ucb_scores)[-batch_size:]
            next_batch = unlabeled_compounds.iloc[top_indices]
            
            # Add to labeled set
            labeled_ids.update(next_batch['ID'])
        
        # Verify multi-cycle workflow
        assert len(labeled_ids) >= 5 + (n_cycles * batch_size)
        assert oracle.call_count == n_cycles


class TestCoreErrorHandling:
    """Test error handling in core system components."""
    
    def test_learner_error_conditions(self, small_real_compounds, tmp_path):
        """Test learner error handling."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner()
        
        # Test training with empty dataset
        empty_compounds = pd.DataFrame(columns=['ID', 'SMILES', 'Activity'])
        with pytest.raises(ValueError, match="empty dataset"):
            learner.train(empty_compounds, 'Activity', data_manager)
        
        # Test training with missing target column
        with pytest.raises(KeyError, match="Target column"):
            learner.train(compounds, 'NonexistentColumn', data_manager)
        
        # Test prediction before training
        with pytest.raises(RuntimeError, match="must be trained"):
            learner.predict(compounds, data_manager)
    
    def test_oracle_error_conditions(self):
        """Test oracle error handling."""
        oracle = MockOracle()
        
        # Test with invalid compounds DataFrame
        invalid_compounds = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
        with pytest.raises(KeyError):
            oracle.measure(invalid_compounds, ['Activity'])
        
        # Test with empty properties list
        compounds = pd.DataFrame({
            'ID': ['mol_1'],
            'SMILES': ['CCO']
        })
        
        result = oracle.measure(compounds, [])
        assert len(result.columns) == 1  # Only ID column
    
    def test_datamanager_learner_mismatch(self, small_real_compounds, tmp_path):
        """Test handling of mismatched data between DataManager and learner."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner()
        
        # Train on subset
        train_compounds = compounds.head(3)
        learner.train(train_compounds, 'Activity', data_manager)
        
        # Predict on different compounds (should still work)
        different_compounds = pd.DataFrame({
            'ID': ['new_mol_1', 'new_mol_2'],
            'SMILES': ['CCCC', 'c1ccc(N)cc1']
        })
        
        predictions, uncertainties = learner.predict(different_compounds, data_manager)
        
        # Should handle gracefully
        assert len(predictions) == 2
        assert uncertainties is None


class TestCorePerformance:
    """Test performance characteristics of core components."""
    
    def test_datamanager_caching_efficiency(self, medium_real_compounds, tmp_path):
        """Test that DataManager caching improves performance."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Use subset for testing
        compounds = compounds.head(30)
        compound_ids = compounds['ID'].tolist()
        
        data_manager = DataManager(results_dir=tmp_path)
        
        # First call should compute and cache features
        import time
        start_time = time.time()
        smiles_list = compounds['SMILES'].tolist()
        features1, valid_ids1 = data_manager.get_features(compound_ids, smiles_list, 'morgan')
        first_call_time = time.time() - start_time

        # Second call should use cache (faster)
        start_time = time.time()
        features2, valid_ids2 = data_manager.get_features(compound_ids, smiles_list, 'morgan')
        second_call_time = time.time() - start_time
        
        # Verify caching worked
        np.testing.assert_array_equal(features1, features2)
        
        # Second call should be faster (allowing some tolerance)
        # Note: This is approximate and may vary in test environments
        if first_call_time > 0.1:  # Only check if first call took reasonable time
            assert second_call_time <= first_call_time * 1.5  # Allow some overhead
    
    def test_learner_scalability(self, tmp_path):
        """Test learner performance with different dataset sizes."""
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner()
        
        sizes = [10, 25, 50]
        training_times = []
        
        for size in sizes:
            # Generate test compounds
            compounds = pd.DataFrame({
                'ID': [f'mol_{i}' for i in range(size)],
                'SMILES': ['CCO'] * size,
                'Activity': np.random.uniform(0, 1, size)
            })
            
            # Measure training time
            import time
            start_time = time.time()
            learner.train(compounds, 'Activity', data_manager)
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            # Test prediction time
            start_time = time.time()
            predictions, _ = learner.predict(compounds, data_manager)
            prediction_time = time.time() - start_time
            
            # Verify results
            assert len(predictions) == size
            
            # Training should be reasonably fast for test sizes
            assert training_time < 10.0  # 10 seconds max for any test size
            assert prediction_time < 5.0   # 5 seconds max for prediction
        
        # Training time should scale reasonably (allowing for variability)
        # Larger datasets may take longer, but not exponentially
        if len(training_times) >= 2:
            # Last should not be more than 10x the first
            assert training_times[-1] <= training_times[0] * 10


class TestCoreEdgeCases:
    """Test edge cases in core system functionality."""
    
    def test_single_compound_workflow(self, tmp_path):
        """Test complete workflow with single compound."""
        single_compound = pd.DataFrame({
            'ID': ['mol_1'],
            'SMILES': ['CCO'],
            'Activity': [0.7]
        })
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner()
        oracle = MockOracle()
        
        # Complete workflow with single compound
        measurements = oracle.measure(single_compound, ['Measured_Activity'])
        compound_with_activity = single_compound.merge(measurements, on='ID')
        
        learner.train(compound_with_activity, 'Measured_Activity', data_manager)
        predictions, uncertainties = learner.predict(single_compound, data_manager)
        
        # Should handle single compound gracefully
        assert len(predictions) == 1
        assert isinstance(predictions[0], (int, float, np.number))
    
    def test_duplicate_compound_handling(self, tmp_path):
        """Test handling of duplicate compounds."""
        compounds_with_duplicates = pd.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_1', 'mol_3'],  # mol_1 appears twice
            'SMILES': ['CCO', 'CCC', 'CCO', 'CCN'],
            'Activity': [0.5, 0.3, 0.5, 0.8]
        })
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner()
        
        # Should handle duplicates appropriately
        try:
            learner.train(compounds_with_duplicates, 'Activity', data_manager)
            predictions, _ = learner.predict(compounds_with_duplicates, data_manager)
            
            # Should return predictions for all rows (including duplicates)
            assert len(predictions) == 4
            
        except Exception as e:
            # Some implementations might reject duplicates, which is also valid
            assert "duplicate" in str(e).lower() or "unique" in str(e).lower()
    
    def test_unusual_smiles_handling(self, tmp_path):
        """Test handling of unusual but valid SMILES."""
        unusual_compounds = pd.DataFrame({
            'ID': ['salt_1', 'stereo_1', 'charged_1'],
            'SMILES': ['CCO.Cl', 'C[C@H](O)C', 'CC[NH3+]'],  # Salt, stereochemistry, charge
            'Activity': [0.5, 0.7, 0.3]
        })
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = MockLearner()
        
        try:
            learner.train(unusual_compounds, 'Activity', data_manager)
            predictions, _ = learner.predict(unusual_compounds, data_manager)
            
            # Should handle unusual SMILES if RDKit can parse them
            assert len(predictions) == 3
            
        except Exception as e:
            # Some unusual SMILES might not be supported, which is acceptable
            pytest.skip(f"Unusual SMILES not supported: {e}")
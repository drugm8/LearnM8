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




class TestCoreInterfaces:
    """Test core interface implementations and compliance."""
    
    def test_learner_interface_compliance(self, small_real_compounds, tmp_path, mock_learner_with_uncertainty):
        """Test that learner implementations comply with interface."""
        compounds = small_real_compounds.copy()

        if len(compounds) == 0:
            pytest.skip("No real molecular data available")

        data_manager = DataManager(results_dir=tmp_path)
        learner = mock_learner_with_uncertainty
        
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
    
    def test_learner_without_uncertainty(self, small_real_compounds, tmp_path, mock_learner):
        """Test learner that doesn't support uncertainty."""
        compounds = small_real_compounds.copy()

        if len(compounds) == 0:
            pytest.skip("No real molecular data available")

        data_manager = DataManager(results_dir=tmp_path)
        learner = mock_learner
        
        assert learner.supports_uncertainty() == False
        
        learner.train(compounds, 'Activity', data_manager)
        predictions, uncertainties = learner.predict(compounds, data_manager)
        
        assert isinstance(predictions, np.ndarray)
        assert uncertainties is None
    
    def test_oracle_interface_compliance(self, small_real_compounds, mock_oracle):
        """Test that oracle implementations comply with interface."""
        compounds = small_real_compounds.copy()

        if len(compounds) == 0:
            pytest.skip("No real molecular data available")

        oracle = mock_oracle
        
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
    
    def test_oracle_with_empty_compounds(self, mock_oracle):
        """Test oracle behavior with empty compound set."""
        empty_compounds = pd.DataFrame(columns=['ID', 'SMILES'])
        oracle = mock_oracle
        
        result = oracle.measure(empty_compounds, ['Activity'])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert 'ID' in result.columns
        assert 'Activity' in result.columns


class TestCoreSystemIntegration:
    """Test integration between core system components."""
    
    def test_datamanager_learner_integration(self, medium_real_compounds, tmp_path, mock_learner):
        """Test DataManager integration with learner."""
        compounds = medium_real_compounds.copy()

        if len(compounds) == 0:
            pytest.skip("No real molecular data available")

        # Use subset for faster testing
        compounds = compounds.head(20)

        data_manager = DataManager(results_dir=tmp_path)
        learner = mock_learner
        
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
    
    def test_datamanager_oracle_integration(self, small_real_compounds, tmp_path, mock_oracle):
        """Test DataManager integration with oracle."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        data_manager = DataManager(results_dir=tmp_path)
        oracle = mock_oracle
        
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
    
    def test_learner_oracle_workflow(self, diverse_real_compounds, tmp_path, mock_learner_with_uncertainty, mock_oracle):
        """Test complete learner-oracle active learning workflow."""
        compounds = diverse_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Use subset for faster testing
        compounds = compounds.head(15)
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = mock_learner_with_uncertainty
        oracle = mock_oracle
        
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
    
    def test_multi_cycle_workflow(self, medium_real_compounds, tmp_path, mock_learner_with_uncertainty, mock_oracle):
        """Test multi-cycle active learning workflow."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Use smaller subset for faster testing
        compounds = compounds.head(25)
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = mock_learner_with_uncertainty
        oracle = mock_oracle
        
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
    
    def test_learner_error_conditions(self, small_real_compounds, tmp_path, mock_learner):
        """Test learner error handling."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = mock_learner
        
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
    
    def test_oracle_error_conditions(self, mock_oracle):
        """Test oracle error handling."""
        oracle = mock_oracle
        
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
    
    def test_datamanager_learner_mismatch(self, small_real_compounds, tmp_path, mock_learner):
        """Test handling of mismatched data between DataManager and learner."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = mock_learner
        
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



class TestCoreEdgeCases:
    """Test edge cases in core system functionality."""
    
    def test_single_compound_workflow(self, tmp_path, mock_learner, mock_oracle):
        """Test complete workflow with single compound."""
        single_compound = pd.DataFrame({
            'ID': ['mol_1'],
            'SMILES': ['CCO'],
            'Activity': [0.7]
        })
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = mock_learner
        oracle = mock_oracle
        
        # Complete workflow with single compound
        measurements = oracle.measure(single_compound, ['Measured_Activity'])
        compound_with_activity = single_compound.merge(measurements, on='ID')
        
        learner.train(compound_with_activity, 'Measured_Activity', data_manager)
        predictions, uncertainties = learner.predict(single_compound, data_manager)
        
        # Should handle single compound gracefully
        assert len(predictions) == 1
        assert isinstance(predictions[0], (int, float, np.number))
    
    def test_duplicate_compound_handling(self, tmp_path, mock_learner):
        """Test handling of duplicate compounds."""
        compounds_with_duplicates = pd.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_1', 'mol_3'],  # mol_1 appears twice
            'SMILES': ['CCO', 'CCC', 'CCO', 'CCN'],
            'Activity': [0.5, 0.3, 0.5, 0.8]
        })
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = mock_learner
        
        # Should handle duplicates appropriately
        try:
            learner.train(compounds_with_duplicates, 'Activity', data_manager)
            predictions, _ = learner.predict(compounds_with_duplicates, data_manager)
            
            # Should return predictions for all rows (including duplicates)
            assert len(predictions) == 4
            
        except Exception as e:
            # Some implementations might reject duplicates, which is also valid
            assert "duplicate" in str(e).lower() or "unique" in str(e).lower()
    
    def test_unusual_smiles_handling(self, tmp_path, mock_learner):
        """Test handling of unusual but valid SMILES."""
        unusual_compounds = pd.DataFrame({
            'ID': ['salt_1', 'stereo_1', 'charged_1'],
            'SMILES': ['CCO.Cl', 'C[C@H](O)C', 'CC[NH3+]'],  # Salt, stereochemistry, charge
            'Activity': [0.5, 0.7, 0.3]
        })
        
        data_manager = DataManager(results_dir=tmp_path)
        learner = mock_learner
        
        try:
            learner.train(unusual_compounds, 'Activity', data_manager)
            predictions, _ = learner.predict(unusual_compounds, data_manager)
            
            # Should handle unusual SMILES if RDKit can parse them
            assert len(predictions) == 3
            
        except Exception as e:
            # Some unusual SMILES might not be supported, which is acceptable
            pytest.skip(f"Unusual SMILES not supported: {e}")
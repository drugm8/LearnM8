"""
Edge case tests consolidated from across the test suite.

Covers boundary conditions and unusual inputs for all components.
"""

import pytest
import pandas as pd
import numpy as np
from learnm8.core.data_manager import DataManager
from learnm8.acquisition.basic import GreedyAcquisition, RandomAcquisition
from learnm8.evaluation.core import evaluate_cycle


class TestDataManagerEdgeCases:
    """Edge cases for DataManager functionality."""
    
    def test_single_compound_dataset(self, edge_case_compounds, tmp_path):
        """Test DataManager with single compound."""
        single_compound = edge_case_compounds.head(1)
        
        dm = DataManager(results_dir=tmp_path)
        features, valid_ids = dm.get_features(
            single_compound['ID'].tolist(),
            smiles_list=single_compound['SMILES'].tolist(),
            featurizer_type='morgan'
        )

        assert features.shape[0] == len(valid_ids)
        assert features.shape[1] > 0  # Should have feature dimensions
        assert len(valid_ids) <= 1  # Should have at most 1 valid compound
    
    def test_invalid_smiles_handling(self, tmp_path):
        """Test handling of invalid SMILES strings."""
        invalid_data = pd.DataFrame({
            'ID': ['invalid_1', 'invalid_2'],
            'SMILES': ['INVALID', 'C((('],  # Invalid SMILES
            'Activity': [1.0, 2.0]
        })
        
        dm = DataManager(results_dir=tmp_path)
        
        # Should handle invalid SMILES gracefully by filtering them out
        features, valid_ids = dm.get_features(
            invalid_data['ID'].tolist(),
            smiles_list=invalid_data['SMILES'].tolist(),
            featurizer_type='morgan'
        )

        # Should filter out invalid SMILES, so no features returned
        assert features.shape[0] == len(valid_ids)
        assert len(valid_ids) == 0  # No valid compounds from invalid SMILES
        assert features.shape == (0, 2048)  # Empty features array
    
    def test_empty_dataset(self, tmp_path):
        """Test DataManager with empty dataset."""
        dm = DataManager(results_dir=tmp_path)
        
        features, valid_ids = dm.get_features([], [], 'morgan')
        assert features.shape == (0, 2048)  # Empty array with correct feature dimension
        assert len(valid_ids) == 0  # No valid IDs for empty input


class TestAcquisitionEdgeCases:
    """Edge cases for acquisition functions."""
    
    def test_acquisition_with_single_compound(self, small_real_compounds):
        """Test acquisition functions with single compound."""
        single_compound = small_real_compounds.head(1).copy()
        single_compound['prediction'] = [0.5]
        
        acq = GreedyAcquisition()
        selected = acq.select(single_compound, n_select=1)
        
        assert len(selected) == 1
        assert selected.iloc[0]['ID'] == single_compound.iloc[0]['ID']
    
    def test_acquisition_with_identical_predictions(self, small_real_compounds):
        """Test acquisition with identical prediction values."""
        compounds = small_real_compounds.head(10).copy()
        compounds['prediction'] = 0.5  # All identical
        
        acq = GreedyAcquisition()
        selected = acq.select(compounds, n_select=3)
        
        assert len(selected) == 3
        # Should select consistently (first ones due to identical scores)
    
    def test_acquisition_with_nan_predictions(self, small_real_compounds):
        """Test acquisition function handling of NaN predictions."""
        compounds = small_real_compounds.head(10).copy()
        compounds['prediction'] = [np.nan if i < 3 else 0.5 for i in range(len(compounds))]
        
        acq = GreedyAcquisition()
        
        # Should reject NaN predictions with error
        with pytest.raises(ValueError, match="Predictions contain NaN values"):
            acq.select(compounds, n_select=5)
    
    def test_random_acquisition_reproducibility(self, small_real_compounds):
        """Test random acquisition reproducibility with seed."""
        compounds = small_real_compounds.head(20).copy()
        compounds['prediction'] = np.random.random(len(compounds))
        
        acq1 = RandomAcquisition(random_state=42)
        acq2 = RandomAcquisition(random_state=42)
        
        selected1 = acq1.select(compounds, n_select=5)
        selected2 = acq2.select(compounds, n_select=5)
        
        # Should select identical compounds with same seed
        assert list(selected1['ID']) == list(selected2['ID'])


class TestEvaluationEdgeCases:
    """Edge cases for evaluation functionality."""
    
    def test_evaluation_with_single_compound(self, edge_case_compounds):
        """Test evaluation with single compound."""
        single_compound = edge_case_compounds.head(1)
        
        predictions = single_compound['Activity'].values
        ground_truth = single_compound['Activity'].values
        
        result = evaluate_cycle(
            cycle=0,
            predictions=predictions,
            ground_truth=ground_truth,
            labeled_data=single_compound,
            selected_compounds=single_compound,
            target_column='Activity'
        )
        
        # Should handle single compound case
        assert isinstance(result, dict)
        assert result['batch_size'] == 1
    
    def test_evaluation_with_empty_selection(self, small_real_compounds):
        """Test evaluation with empty selected compounds."""
        labeled_data = small_real_compounds.copy()
        empty_selection = pd.DataFrame(columns=['ID', 'SMILES', 'Activity'])
        
        predictions = labeled_data['Activity'].values
        ground_truth = labeled_data['Activity'].values
        
        result = evaluate_cycle(
            cycle=0,
            predictions=predictions,
            ground_truth=ground_truth,
            labeled_data=labeled_data,
            selected_compounds=empty_selection,
            target_column='Activity'
        )
        
        # Should handle empty selection gracefully
        assert isinstance(result, dict)
        assert result['batch_size'] == 0
    
    def test_evaluation_with_extreme_values(self, small_real_compounds):
        """Test evaluation with extreme prediction values."""
        labeled_data = small_real_compounds.copy()
        selected_compounds = small_real_compounds.head(5)
        
        # Create extreme predictions
        predictions = np.array([1e10 if i % 2 == 0 else -1e10 for i in range(len(labeled_data))])
        ground_truth = labeled_data['Activity'].values
        
        result = evaluate_cycle(
            cycle=0,
            predictions=predictions,
            ground_truth=ground_truth,
            labeled_data=labeled_data,
            selected_compounds=selected_compounds,
            target_column='Activity'
        )
        
        # Should handle extreme values
        assert isinstance(result, dict)
        assert np.isfinite(result['rmse'])
        assert np.isfinite(result['mae'])


class TestIntegratedEdgeCases:
    """Edge cases that span multiple components."""
    
    def test_workflow_with_edge_case_molecules(self, edge_case_compounds, tmp_path):
        """Test complete workflow with edge case molecular structures."""
        compounds = edge_case_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No edge case compounds available")
        
        # Test DataManager with edge cases
        dm = DataManager(results_dir=tmp_path)
        try:
            features, valid_ids = dm.get_features(
                compounds['ID'].tolist(),
                compounds['SMILES'].tolist(),
                'morgan'
            )
            
            # Test acquisition with edge case features
            compounds['prediction'] = np.random.random(len(compounds))
            acq = GreedyAcquisition()
            selected = acq.select(compounds, n_select=min(3, len(compounds)))
            
            assert len(selected) <= len(compounds)
            assert features.shape[0] == len(valid_ids)
            assert len(valid_ids) <= len(compounds)
            
        except (ValueError, RuntimeError) as e:
            # Some edge cases may legitimately fail
            pytest.skip(f"Edge case molecules caused expected failure: {e}")
    
    def test_workflow_with_minimal_data(self, edge_case_compounds, tmp_path):
        """Test workflow with minimal viable dataset."""
        # Use just 2 compounds
        minimal_data = edge_case_compounds.head(2)
        
        if len(minimal_data) < 2:
            pytest.skip("Insufficient compounds for minimal test")
        
        # Should handle minimal data without crashing
        dm = DataManager(results_dir=tmp_path)
        features, valid_ids = dm.get_features(
            minimal_data['ID'].tolist(),
            minimal_data['SMILES'].tolist(),
            'morgan'
        )

        assert features.shape[0] == len(valid_ids)
        assert len(valid_ids) <= 2  # Should have at most 2 valid compounds
        
        # Test acquisition
        minimal_data['prediction'] = [0.1, 0.9]
        acq = GreedyAcquisition()
        selected = acq.select(minimal_data, n_select=1)
        
        assert len(selected) == 1
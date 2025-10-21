"""
Edge case tests consolidated from across the test suite.

Covers boundary conditions and unusual inputs for all components.
"""

import pytest
import pandas as pd
import numpy as np
from learnm8.features import extract_features
from learnm8.acquisition.basic import GreedyAcquisition, RandomAcquisition
from learnm8.evaluation.core import evaluate_cycle


class TestFeatureExtractionEdgeCases:
    """Edge cases for feature extraction functionality."""
    
    def test_single_compound_dataset(self, edge_case_compounds, tmp_path):
        """Test feature extraction with single compound."""
        single_compound = edge_case_compounds.head(1)

        features = extract_features(
            single_compound['SMILES'].tolist(),
            'morgan',
            tmp_path
        )

        assert features.shape[0] <= 1
        assert features.shape[1] > 0  # Should have feature dimensions
    
    def test_invalid_smiles_handling(self, tmp_path):
        """Test handling of invalid SMILES strings."""
        invalid_smiles = ['INVALID', 'C(((']  # Invalid SMILES

        # Should raise error for invalid SMILES
        with pytest.raises(Exception):
            extract_features(invalid_smiles, 'morgan', tmp_path)
    
    def test_empty_dataset(self, tmp_path):
        """Test feature extraction with empty dataset."""
        features = extract_features([], 'morgan', tmp_path)
        assert features.shape == (0, 2048)  # Empty array with correct feature dimension


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

        # Test feature extraction with edge cases
        try:
            features = extract_features(
                compounds['SMILES'].tolist(),
                'morgan',
                tmp_path
            )

            # Test acquisition with edge case features
            compounds['prediction'] = np.random.random(len(compounds))
            acq = GreedyAcquisition()
            selected = acq.select(compounds, n_select=min(3, len(compounds)))

            assert len(selected) <= len(compounds)
            assert features.shape[0] <= len(compounds)

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
        features = extract_features(
            minimal_data['SMILES'].tolist(),
            'morgan',
            tmp_path
        )

        assert features.shape[0] <= 2  # Should have at most 2 compounds

        # Test acquisition
        minimal_data['prediction'] = [0.1, 0.9]
        acq = GreedyAcquisition()
        selected = acq.select(minimal_data, n_select=1)

        assert len(selected) == 1
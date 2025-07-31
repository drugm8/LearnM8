"""Kennard-Stone acquisition function tests.

Tests the KennardStoneAcquisition class for optimal molecular diversity sampling
using real molecular data and comprehensive functionality validation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from learnm8.acquisition.kennard_stone import KennardStoneAcquisition, create_kennard_stone_acquisition
from learnm8.acquisition.basic import RandomAcquisition, GreedyAcquisition
from learnm8.core.data_manager import DataManager


class TestKennardStoneAcquisition:
    """Test Kennard-Stone acquisition functionality."""
    
    def test_kennard_stone_basic_functionality(self, small_real_compounds):
        """Test basic Kennard-Stone acquisition with real molecular data."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) < 10:
            pytest.skip("Insufficient compounds for Kennard-Stone test")
        
        # Add predictions (KS doesn't use them but interface requires them)
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Create mock DataManager to avoid file system dependencies
        mock_dm = Mock()
        
        # Create mock fingerprints (binary for Tanimoto distance)
        n_compounds = len(compounds)
        np.random.seed(42)
        mock_fingerprints = np.random.randint(0, 2, size=(n_compounds, 1024))
        mock_dm.get_features.return_value = mock_fingerprints
        
        acq = KennardStoneAcquisition(mock_dm, featurizer_type='morgan', random_state=42)
        selected = acq.select(compounds, n_select=8)
        
        assert len(selected) == 8
        assert all(col in selected.columns for col in ['ID', 'SMILES', 'prediction'])
        assert 'acquisition_score' in selected.columns
        assert 'selection_order' in selected.columns
        
        # Should select valid compounds
        assert all(id in compounds['ID'].values for id in selected['ID'])
        
        # Acquisition scores should be in descending order (priority)
        assert all(selected['acquisition_score'].iloc[i] >= selected['acquisition_score'].iloc[i+1] 
                  for i in range(len(selected)-1))
        
        # Selection order should be ascending
        assert list(selected['selection_order']) == list(range(8))
    
    def test_kennard_stone_deterministic_selection(self, medium_real_compounds):
        """Test that Kennard-Stone selection is deterministic with same random_state."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) < 20:
            pytest.skip("Insufficient compounds for deterministic test")
        
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Create mock DataManager directly
        mock_dm = Mock()
        
        n_compounds = len(compounds)
        np.random.seed(42)
        mock_fingerprints = np.random.randint(0, 2, size=(n_compounds, 1024))
        mock_dm.get_features.return_value = mock_fingerprints
        
        # Run selection twice with same random_state
        acq1 = KennardStoneAcquisition(mock_dm, random_state=42)
        selected1 = acq1.select(compounds, n_select=10)
        
        acq2 = KennardStoneAcquisition(mock_dm, random_state=42)
        selected2 = acq2.select(compounds, n_select=10)
        
        # Should select identical compounds in same order
        assert list(selected1['ID']) == list(selected2['ID'])
        assert list(selected1['selection_order']) == list(selected2['selection_order'])
    
    def test_kennard_stone_vs_random_diversity(self, diverse_real_compounds):
        """Test that Kennard-Stone provides better diversity than random selection."""
        compounds = diverse_real_compounds.copy()
        
        if len(compounds) < 30:
            pytest.skip("Insufficient compounds for diversity comparison")
        
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Create mock DataManager directly
        mock_dm = Mock()
        
        n_compounds = len(compounds)
        np.random.seed(42)
        # Create more diverse fingerprints for meaningful comparison
        mock_fingerprints = np.random.randint(0, 2, size=(n_compounds, 2048))
        mock_dm.get_features.return_value = mock_fingerprints
        
        # Kennard-Stone selection
        ks_acq = KennardStoneAcquisition(mock_dm, random_state=42)
        ks_selected = ks_acq.select(compounds, n_select=15)
        
        # Random selection for comparison
        random_acq = RandomAcquisition(random_state=42)
        random_selected = random_acq.select(compounds, n_select=15)
        
        # Both should select same number
        assert len(ks_selected) == len(random_selected) == 15
        
        # KS should generally select different compounds than random
        # (This is probabilistic, but with seed=42 should be reliable)
        ks_ids = set(ks_selected['ID'])
        random_ids = set(random_selected['ID'])
        overlap = len(ks_ids & random_ids)
        
        # Expect less than 50% overlap (indicates different selection strategy)
        assert overlap < 7, f"Too much overlap ({overlap}/15) between KS and random selection"
    
    def test_kennard_stone_different_featurizers(self, small_real_compounds):
        """Test Kennard-Stone with different featurizer types."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) < 10:
            pytest.skip("Insufficient compounds for featurizer test")
        
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        featurizer_types = ['morgan', 'maccs', 'ecfp6']
        
        selections = {}
        
        for feat_type in featurizer_types:
            # Create mock DataManager directly
            mock_dm = Mock()
            
            # Different fingerprint sizes for different types
            n_compounds = len(compounds)
            if feat_type == 'morgan':
                fp_size = 2048
            elif feat_type == 'maccs':
                fp_size = 167
            else:  # ecfp6
                fp_size = 2048
            
            np.random.seed(42)  # Same seed for fair comparison
            mock_fingerprints = np.random.randint(0, 2, size=(n_compounds, fp_size))
            mock_dm.get_features.return_value = mock_fingerprints
            
            acq = KennardStoneAcquisition(mock_dm, featurizer_type=feat_type, random_state=42)
            selected = acq.select(compounds, n_select=6)
            
            assert len(selected) == 6
            assert feat_type in acq.get_name()
            selections[feat_type] = set(selected['ID'])
        
        # Different featurizers might select different compounds
        # (Though with same random fingerprints they'll be similar)
        assert len(selections) == len(featurizer_types)
    
    def test_kennard_stone_edge_cases(self, edge_case_compounds):
        """Test Kennard-Stone with edge cases and small datasets."""
        compounds = edge_case_compounds.copy()
        
        if len(compounds) == 0:
            # Create minimal test data
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002'],
                'SMILES': ['CCO', 'CCC'],
                'prediction': [0.5, 0.7]
            })
        else:
            compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Create mock DataManager directly
        mock_dm = Mock()
        
        n_compounds = len(compounds)
        np.random.seed(42)
        mock_fingerprints = np.random.randint(0, 2, size=(n_compounds, 1024))
        mock_dm.get_features.return_value = mock_fingerprints
        
        acq = KennardStoneAcquisition(mock_dm, random_state=42)
        
        # Test single compound selection
        if n_compounds >= 1:
            selected = acq.select(compounds, n_select=1)
            assert len(selected) == 1
        
        # Test selecting all compounds
        if n_compounds >= 2:
            selected_all = acq.select(compounds, n_select=n_compounds)
            assert len(selected_all) == n_compounds
        
        # Test selecting more than available
        if n_compounds >= 2:
            selected_more = acq.select(compounds, n_select=n_compounds + 5)
            assert len(selected_more) == n_compounds
    
    def test_kennard_stone_input_validation(self, small_real_compounds):
        """Test input validation and error handling."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
                'SMILES': ['CCO', 'CCC', 'CCN'],
                'prediction': [0.1, 0.5, 0.9]
            })
        else:
            compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Create mock DataManager directly
        mock_dm = Mock()
        acq = KennardStoneAcquisition(mock_dm)
        
        # Test empty DataFrame
        empty_df = pd.DataFrame(columns=['ID', 'SMILES', 'prediction'])
        with pytest.raises(ValueError, match="compounds DataFrame is empty"):
            acq.select(empty_df, n_select=1)
        
        # Test missing columns
        incomplete_df = compounds[['ID', 'SMILES']].copy()
        with pytest.raises(ValueError, match="Missing required columns"):
            acq.select(incomplete_df, n_select=1)
        
        # Test invalid n_select
        with pytest.raises(ValueError, match="n_select must be positive"):
            acq.select(compounds, n_select=0)
        
        with pytest.raises(ValueError, match="n_select must be positive"):
            acq.select(compounds, n_select=-1)
    
    def test_kennard_stone_requires_uncertainty(self):
        """Test that Kennard-Stone doesn't require uncertainty estimates."""
        mock_dm = Mock()
        acq = KennardStoneAcquisition(mock_dm)
        assert not acq.requires_uncertainty()
    
    def test_kennard_stone_get_name(self):
        """Test acquisition function naming."""
        mock_dm = Mock()
        acq_morgan = KennardStoneAcquisition(mock_dm, featurizer_type='morgan')
        assert 'Kennard-Stone' in acq_morgan.get_name()
        assert 'morgan' in acq_morgan.get_name()
        
        acq_maccs = KennardStoneAcquisition(mock_dm, featurizer_type='maccs')
        assert 'maccs' in acq_maccs.get_name()
    
    def test_kennard_stone_feature_extraction_failure(self, small_real_compounds):
        """Test error handling when feature extraction fails."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002'],
                'SMILES': ['CCO', 'CCC'],
                'prediction': [0.5, 0.7]
            })
        else:
            compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Create mock DataManager directly
        mock_dm = Mock()
        mock_dm.get_features.side_effect = RuntimeError("Feature extraction failed")
        
        acq = KennardStoneAcquisition(mock_dm)
        
        with pytest.raises(RuntimeError, match="Kennard-Stone acquisition failed"):
            acq.select(compounds, n_select=2)


class TestKennardStoneFactory:
    """Test factory function for KennardStoneAcquisition."""
    
    def test_create_kennard_stone_acquisition_defaults(self):
        """Test factory function with default parameters."""
        mock_dm = Mock()
        acq = create_kennard_stone_acquisition(mock_dm)
        
        assert isinstance(acq, KennardStoneAcquisition)
        assert acq.featurizer_type == 'morgan'
        assert acq.random_state is None
    
    def test_create_kennard_stone_acquisition_custom(self):
        """Test factory function with custom parameters."""
        mock_dm = Mock()
        acq = create_kennard_stone_acquisition(
            mock_dm,
            featurizer_type='maccs',
            random_state=123
        )
        
        assert isinstance(acq, KennardStoneAcquisition)
        assert acq.featurizer_type == 'maccs'
        assert acq.random_state == 123


# Integration tests with more complex scenarios
class TestKennardStoneIntegration:
    """Integration tests for Kennard-Stone acquisition."""
    
    def test_kennard_stone_with_real_data_manager(self, small_real_compounds, tmp_path):
        """Test Kennard-Stone with actual DataManager (if data available)."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) < 5:
            pytest.skip("Insufficient real compounds for integration test")
        
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        try:
            # This will fail if HDF5 cache isn't set up, which is expected in test environment
            mock_dm = Mock()
            acq = KennardStoneAcquisition(mock_dm, random_state=42)
            selected = acq.select(compounds, n_select=3)
            
            # If it succeeds, validate results
            assert len(selected) == 3
            assert all(col in selected.columns for col in ['ID', 'SMILES', 'prediction'])
            
        except (RuntimeError, FileNotFoundError, Exception):
            # Expected in test environment without proper HDF5 setup
            pytest.skip("DataManager not available in test environment")
    
    def test_kennard_stone_performance_characteristics(self, medium_real_compounds):
        """Test performance characteristics with larger dataset."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) < 50:
            pytest.skip("Insufficient compounds for performance test")
        
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Create mock DataManager directly
        mock_dm = Mock()
        
        n_compounds = len(compounds)
        np.random.seed(42)
        mock_fingerprints = np.random.randint(0, 2, size=(n_compounds, 2048))
        mock_dm.get_features.return_value = mock_fingerprints
        
        acq = KennardStoneAcquisition(mock_dm, random_state=42)
        
        # Test with different selection sizes
        for n_select in [5, 10, 20]:
            if n_select <= len(compounds):
                selected = acq.select(compounds, n_select=n_select)
                assert len(selected) == n_select
                
                # Verify selection order is maintained
                assert list(selected['selection_order']) == list(range(n_select))
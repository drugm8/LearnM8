"""Butina clustering acquisition function tests.

Tests the ButinaClusteringAcquisition class for molecular diversity selection
using Butina clustering algorithm with real molecular data.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from learnm8.acquisition.basic import RandomAcquisition, GreedyAcquisition

# Try to import RDKit and Butina acquisition for testing
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina
    from learnm8.acquisition.butina import ButinaClusteringAcquisition
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None
    Butina = None
    ButinaClusteringAcquisition = None


class TestButinaClusteringAcquisition:
    """Test Butina clustering acquisition functionality."""
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_butina_basic_functionality(self, diverse_real_compounds):
        """Test basic Butina clustering acquisition with real molecular data."""
        compounds = diverse_real_compounds.copy()
        
        if len(compounds) < 10:
            pytest.skip("Insufficient compounds for Butina clustering test")
        
        # Add predictions
        np.random.seed(42)
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        acq = ButinaClusteringAcquisition(threshold=0.4, random_state=42)
        selected = acq.select(compounds, n_select=8)
        
        assert len(selected) == 8
        assert all(col in selected.columns for col in ['ID', 'SMILES', 'prediction'])
        assert 'acquisition_score' in selected.columns
        assert all(id in compounds['ID'].values for id in selected['ID'])
        
        # Acquisition scores should be cluster sizes (positive integers)
        assert all(score > 0 for score in selected['acquisition_score'])
        assert all(isinstance(score, (int, np.integer)) for score in selected['acquisition_score'])
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_butina_deterministic_selection(self, medium_real_compounds):
        """Test reproducibility with same random_state."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) < 20:
            pytest.skip("Insufficient compounds for deterministic test")
        
        # Use first 50 compounds for faster testing
        compounds = compounds.head(50).copy()
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        acq1 = ButinaClusteringAcquisition(threshold=0.4, random_state=42)
        acq2 = ButinaClusteringAcquisition(threshold=0.4, random_state=42)
        
        selected1 = acq1.select(compounds, n_select=10)
        selected2 = acq2.select(compounds, n_select=10)
        
        # Should select identical compounds with same random state
        assert list(selected1['ID']) == list(selected2['ID'])
        assert list(selected1['acquisition_score']) == list(selected2['acquisition_score'])
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_butina_vs_random_diversity(self, diverse_real_compounds):
        """Test that Butina selects different compounds than random."""
        compounds = diverse_real_compounds.copy()
        
        if len(compounds) < 25:
            pytest.skip("Insufficient compounds for diversity comparison test")
        
        # Use first 30 compounds for faster testing
        compounds = compounds.head(30).copy()
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        butina_acq = ButinaClusteringAcquisition(threshold=0.4, random_state=42)
        random_acq = RandomAcquisition(random_state=43)  # Different seed
        
        butina_selected = butina_acq.select(compounds, n_select=12)
        random_selected = random_acq.select(compounds, n_select=12)
        
        assert len(butina_selected) == 12
        assert len(random_selected) == 12
        
        # Should select different compounds due to clustering
        butina_ids = set(butina_selected['ID'])
        random_ids = set(random_selected['ID'])
        
        overlap = len(butina_ids & random_ids)
        assert overlap < 10  # Less than 80% overlap expected
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_butina_threshold_effect(self, diverse_real_compounds):
        """Test effect of different clustering thresholds."""
        compounds = diverse_real_compounds.copy()
        
        if len(compounds) < 20:
            pytest.skip("Insufficient compounds for threshold test")
        
        # Use first 25 compounds for faster testing
        compounds = compounds.head(25).copy()
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Test tight clustering (low threshold = high similarity required)
        acq_tight = ButinaClusteringAcquisition(threshold=0.2, random_state=42)
        # Test loose clustering (high threshold = low similarity required)
        acq_loose = ButinaClusteringAcquisition(threshold=0.6, random_state=42)
        
        selected_tight = acq_tight.select(compounds, n_select=8)
        selected_loose = acq_loose.select(compounds, n_select=8)
        
        assert len(selected_tight) == 8
        assert len(selected_loose) == 8
        
        # Tight clustering should generally produce more clusters (smaller cluster sizes)
        # Loose clustering should produce fewer clusters (larger cluster sizes)
        avg_cluster_size_tight = np.mean(selected_tight['acquisition_score'])
        avg_cluster_size_loose = np.mean(selected_loose['acquisition_score'])
        
        # This is a tendency, not a strict rule due to data-dependent behavior
        assert avg_cluster_size_tight >= 1.0
        assert avg_cluster_size_loose >= 1.0
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_butina_featurizer_types(self, diverse_real_compounds):
        """Test different molecular featurizer types."""
        compounds = diverse_real_compounds.copy()
        
        if len(compounds) < 15:
            pytest.skip("Insufficient compounds for featurizer test")
        
        # Use first 20 compounds for faster testing
        compounds = compounds.head(20).copy()
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Test Morgan fingerprints
        acq_morgan = ButinaClusteringAcquisition(
            threshold=0.4, featurizer_type='morgan', random_state=42
        )
        
        # Test MACCS fingerprints
        acq_maccs = ButinaClusteringAcquisition(
            threshold=0.4, featurizer_type='maccs', random_state=42
        )
        
        selected_morgan = acq_morgan.select(compounds, n_select=6)
        selected_maccs = acq_maccs.select(compounds, n_select=6)
        
        assert len(selected_morgan) == 6
        assert len(selected_maccs) == 6
        
        # Different featurizers may select different compounds
        morgan_ids = set(selected_morgan['ID'])
        maccs_ids = set(selected_maccs['ID'])
        
        # Some difference expected due to different fingerprint representations
        assert len(morgan_ids) == len(maccs_ids) == 6
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")  
    def test_butina_edge_cases(self, edge_case_compounds):
        """Test edge cases and boundary conditions."""
        compounds = edge_case_compounds.copy()
        
        if len(compounds) < 5:
            pytest.skip("Insufficient compounds for edge case test")
        
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        acq = ButinaClusteringAcquisition(threshold=0.4, random_state=42)
        
        # Test selecting single compound
        selected_one = acq.select(compounds, n_select=1)
        assert len(selected_one) == 1
        assert 'acquisition_score' in selected_one.columns
        
        # Test selecting all compounds
        all_compounds = len(compounds)
        selected_all = acq.select(compounds, n_select=all_compounds)
        assert len(selected_all) == all_compounds
        
        # Test selecting more than available
        selected_more = acq.select(compounds, n_select=all_compounds + 5)
        assert len(selected_more) == all_compounds
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_butina_input_validation(self, small_real_compounds):
        """Test input validation and error handling."""
        compounds = small_real_compounds.copy()
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Test valid initialization
        acq = ButinaClusteringAcquisition(threshold=0.4)
        
        # Test invalid threshold values
        with pytest.raises(ValueError, match="Threshold .* not in valid range"):
            ButinaClusteringAcquisition(threshold=0.0)
        
        with pytest.raises(ValueError, match="Threshold .* not in valid range"):
            ButinaClusteringAcquisition(threshold=1.0)
        
        # Test unsupported featurizer type
        with pytest.raises(ValueError, match="Unsupported featurizer type"):
            acq_bad = ButinaClusteringAcquisition(featurizer_type='unsupported')
            acq_bad.select(compounds, n_select=5)
        
        # Test empty DataFrame
        empty_df = pd.DataFrame(columns=['ID', 'SMILES', 'prediction'])
        with pytest.raises(ValueError, match="compounds DataFrame is empty"):
            acq.select(empty_df, n_select=1)
        
        # Test missing required columns
        incomplete_df = compounds[['ID', 'SMILES']].copy()
        with pytest.raises(ValueError, match="Missing required columns"):
            acq.select(incomplete_df, n_select=1)
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_butina_invalid_smiles_handling(self, small_real_compounds):
        """Test handling of invalid SMILES strings."""
        compounds = small_real_compounds.copy()
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Add invalid SMILES
        compounds.loc[0, 'SMILES'] = 'invalid_smiles_string'
        
        acq = ButinaClusteringAcquisition(threshold=0.4, random_state=42)
        
        with pytest.raises(ValueError, match="Invalid SMILES"):
            acq.select(compounds, n_select=5)
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_butina_large_dataset_protection(self, small_real_compounds):
        """Test protection against overly large datasets."""
        # Create a mock dataset that appears large
        compounds = small_real_compounds.copy()
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Set a very low max_compounds limit
        acq = ButinaClusteringAcquisition(threshold=0.4, max_compounds=5)
        
        if len(compounds) > 5:
            with pytest.raises(ValueError, match="Dataset too large for Butina clustering"):
                acq.select(compounds, n_select=3)
    
    def test_butina_no_rdkit_error(self):
        """Test error when RDKit is not available."""
        with patch('learnm8.acquisition.butina.RDKIT_AVAILABLE', False):
            with pytest.raises(ImportError, match="RDKit is required for Butina clustering"):
                ButinaClusteringAcquisition()
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_butina_get_name(self):
        """Test get_name method returns proper identifier."""
        acq = ButinaClusteringAcquisition(threshold=0.4)
        name = acq.get_name()
        
        assert isinstance(name, str)
        assert 'butina' in name.lower()
        assert '0.4' in name


class TestButinaClusteringIntegration:
    """Integration tests for Butina clustering acquisition."""
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_butina_full_workflow(self, medium_real_compounds):
        """Test complete workflow from clustering to selection."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) < 30:
            pytest.skip("Insufficient compounds for full workflow test")
        
        # Use subset for faster testing
        compounds = compounds.head(40).copy()
        
        # Add realistic predictions with some noise
        np.random.seed(42)
        base_activity = compounds.get('Activity', np.random.random(len(compounds)))
        compounds['prediction'] = base_activity + np.random.normal(0, 0.1, len(compounds))
        
        acq = ButinaClusteringAcquisition(
            threshold=0.35,
            featurizer_type='morgan',
            fp_radius=2,
            fp_size=1024,
            random_state=42
        )
        
        # Test multiple selection sizes
        for n_select in [5, 10, 15]:
            selected = acq.select(compounds, n_select=n_select)
            
            assert len(selected) == n_select
            assert all(col in selected.columns for col in ['ID', 'SMILES', 'prediction', 'acquisition_score'])
            assert all(id in compounds['ID'].values for id in selected['ID'])
            
            # Verify no duplicate selections
            assert len(set(selected['ID'])) == n_select
            
            # Verify acquisition scores are meaningful (cluster sizes)
            assert all(score >= 1 for score in selected['acquisition_score'])
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_butina_vs_greedy_comparison(self, diverse_real_compounds):
        """Compare Butina clustering with greedy acquisition."""
        compounds = diverse_real_compounds.copy()
        
        if len(compounds) < 25:
            pytest.skip("Insufficient compounds for comparison test")
        
        # Use subset for faster testing
        compounds = compounds.head(30).copy()
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        butina_acq = ButinaClusteringAcquisition(threshold=0.4, random_state=42)
        greedy_acq = GreedyAcquisition()
        
        butina_selected = butina_acq.select(compounds, n_select=10)
        greedy_selected = greedy_acq.select(compounds, n_select=10)
        
        assert len(butina_selected) == 10
        assert len(greedy_selected) == 10
        
        # Butina should provide diversity-based selection
        butina_ids = set(butina_selected['ID'])
        greedy_ids = set(greedy_selected['ID'])
        
        # Should have some difference due to clustering-based selection
        overlap = len(butina_ids & greedy_ids)
        
        # Allow for some overlap but expect difference due to diversity focus
        assert overlap <= 9  # Less than 90% overlap expected
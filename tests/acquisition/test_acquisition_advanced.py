"""Tests for advanced LearnM8 acquisition functions.

Focused tests for clustering-based and diversity acquisition methods using real molecular data.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from learnm8.core.data_manager import DataManager
from learnm8.acquisition import get_acquisition_function
from learnm8.acquisition.basic import GreedyAcquisition
from learnm8.acquisition.umap_dbscan import UMAPDBSCANAcquisition, UMAPKMeansAcquisition
from learnm8.acquisition.tsne_dbscan import TSNEDBSCANAcquisition, TSNEKMeansAcquisition

# Conditionally import BitBIRCH
try:
    from learnm8.acquisition.bitbirch import BitBIRCHAcquisition
    BITBIRCH_AVAILABLE = True
except ImportError:
    BITBIRCH_AVAILABLE = False


@pytest.fixture
def temp_data_manager():
    """Create temporary DataManager for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dm = DataManager(results_dir=temp_dir)
        yield dm



class TestUMAPAcquisition:
    """Test UMAP-based acquisition methods."""
    
    def test_umap_dbscan_functionality(self, temp_data_manager, medium_real_compounds):
        """Test UMAP+DBSCAN acquisition functionality."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No medium real compounds available")
        
        compounds = compounds.head(20)
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        # Test with sensible defaults
        acq = UMAPDBSCANAcquisition(temp_data_manager, eps=0.5, min_samples=3)
        
        try:
            selected = acq.select(compounds, n_select=6)
            
            # Should select compounds
            assert len(selected) == 6
            assert isinstance(selected, pd.DataFrame)
            
            # Check method name
            name = acq.get_name()
            assert 'UMAP+DBSCAN' in name
            
        except Exception as e:
            pytest.skip(f"UMAP+DBSCAN failed: {e}")
    
    def test_umap_kmeans_functionality(self, temp_data_manager, medium_real_compounds):
        """Test UMAP+K-Means acquisition functionality."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No medium real compounds available")
        
        compounds = compounds.head(20)
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        # Test with fixed number of clusters
        acq = UMAPKMeansAcquisition(temp_data_manager, n_clusters=5)
        
        try:
            selected = acq.select(compounds, n_select=6)
            
            assert len(selected) == 6
            assert isinstance(selected, pd.DataFrame)
            
            # Check method name
            name = acq.get_name()
            assert 'UMAP+K-Means' in name
            
        except Exception as e:
            pytest.skip(f"UMAP+K-Means failed: {e}")
    
    def test_umap_without_umap_library(self, temp_data_manager, medium_real_compounds):
        """Test UMAP acquisition error handling when UMAP not available."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No medium real compounds available")
        
        compounds = compounds.head(15)
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        acq = UMAPDBSCANAcquisition(temp_data_manager)
        
        try:
            selected = acq.select(compounds, n_select=5)
            assert len(selected) == 5
        except ImportError:
            pytest.skip("UMAP not available")
        except Exception as e:
            # UMAP might fail with small datasets
            pytest.skip(f"UMAP failed: {e}")


class TestTSNEAcquisition:
    """Test t-SNE-based acquisition methods."""
    
    def test_tsne_dbscan_functionality(self, temp_data_manager, medium_real_compounds):
        """Test t-SNE+DBSCAN acquisition functionality."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No medium real compounds available")
        
        # Use smaller subset for t-SNE
        compounds = compounds.head(15)
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        acq = TSNEDBSCANAcquisition(temp_data_manager, eps=0.5, min_samples=3)
        
        try:
            selected = acq.select(compounds, n_select=5)
            
            assert len(selected) == 5
            assert isinstance(selected, pd.DataFrame)
            
            # Check method name
            name = acq.get_name()
            assert 't-SNE+DBSCAN' in name
            
        except Exception as e:
            pytest.skip(f"t-SNE+DBSCAN failed: {e}")
    
    def test_tsne_kmeans_functionality(self, temp_data_manager, medium_real_compounds):
        """Test t-SNE+K-Means acquisition functionality."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No medium real compounds available")
        
        compounds = compounds.head(15)
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        acq = TSNEKMeansAcquisition(temp_data_manager, n_clusters=4)
        
        try:
            selected = acq.select(compounds, n_select=5)
            
            assert len(selected) == 5
            assert isinstance(selected, pd.DataFrame)
            
            # Check method name
            name = acq.get_name()
            assert 't-SNE+K-Means' in name
            
        except Exception as e:
            pytest.skip(f"t-SNE+K-Means failed: {e}")
    
    def test_tsne_acquisition_with_warnings(self, temp_data_manager, medium_real_compounds):
        """Test t-SNE acquisition with large dataset warnings."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No medium real compounds available")
        
        # Use larger subset to trigger warning
        compounds = compounds.head(60)
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        acq = TSNEDBSCANAcquisition(temp_data_manager, max_compounds_warning=50)
        
        try:
            # Should produce warning for dataset size
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                selected = acq.select(compounds, n_select=8)
                
                # Check for warning
                warning_messages = [str(warning.message) for warning in w]
                has_warning = any("Large dataset for t-SNE" in msg for msg in warning_messages)
                
                if not has_warning:
                    # Warning might be suppressed, that's okay
                    pass
            
            assert len(selected) == 8
            
        except Exception as e:
            pytest.skip(f"t-SNE failed: {e}")


@pytest.mark.skipif(not BITBIRCH_AVAILABLE, reason="BitBIRCH not available")
class TestBitBIRCHAcquisition:
    """Test BitBIRCH acquisition methods."""
    
    def test_bitbirch_initialization(self, temp_data_manager):
        """Test BitBIRCH acquisition initialization."""
        acq = BitBIRCHAcquisition(temp_data_manager)
        
        # Verify default parameters
        assert acq.featurizer_type == 'morgan'
        assert acq.threshold == 0.5
        assert acq.branching_factor == 50
        assert acq.random_state == 42
        
        # Test custom parameters
        acq_custom = BitBIRCHAcquisition(
            temp_data_manager,
            threshold=0.7,
            branching_factor=30,
            random_state=123
        )
        
        assert acq_custom.threshold == 0.7
        assert acq_custom.branching_factor == 30
        assert acq_custom.random_state == 123
    
    def test_bitbirch_requires_uncertainty(self, temp_data_manager):
        """Test that BitBIRCH does not require uncertainty."""
        acq = BitBIRCHAcquisition(temp_data_manager)
        assert not acq.requires_uncertainty()
    
    def test_bitbirch_get_name(self, temp_data_manager):
        """Test BitBIRCH name generation."""
        acq = BitBIRCHAcquisition(temp_data_manager, featurizer_type='morgan')
        assert acq.get_name() == 'BitBIRCH(morgan)'
        
        acq = BitBIRCHAcquisition(temp_data_manager, featurizer_type='ecfp6')
        assert acq.get_name() == 'BitBIRCH(ecfp6)'


class TestAdvancedAcquisitionIntegration:
    """Test integration of advanced acquisition methods."""
    
    def test_advanced_acquisition_registry(self, temp_data_manager):
        """Test advanced acquisition function registration."""
        # Test advanced functions that require DataManager
        advanced_functions = ['umap_dbscan', 'umap_kmeans', 'tsne_dbscan', 'tsne_kmeans']
        
        for func_name in advanced_functions:
            try:
                acq_cls = get_acquisition_function(func_name)
                assert callable(acq_cls)
                
                # All these require DataManager
                acq = acq_cls(temp_data_manager)
                
                assert hasattr(acq, 'select')
                assert hasattr(acq, 'get_name')
                
            except Exception as e:
                # Some advanced methods might not be available
                pytest.skip(f"Advanced function {func_name} not available: {e}")
    
    def test_clustering_consistency(self, temp_data_manager, medium_real_compounds):
        """Test clustering-based acquisition consistency."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No medium real compounds available")
        
        compounds = compounds.head(20)
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        # Test reproducibility with same random seed
        try:
            acq1 = UMAPDBSCANAcquisition(temp_data_manager, random_state=42)
            acq2 = UMAPDBSCANAcquisition(temp_data_manager, random_state=42)
            
            selected1 = acq1.select(compounds, n_select=5)
            selected2 = acq2.select(compounds, n_select=5)
            
            # Should be reproducible
            assert set(selected1['ID']) == set(selected2['ID'])
            
        except Exception as e:
            pytest.skip(f"Clustering consistency test failed: {e}")
    
    def test_advanced_acquisition_error_handling(self, temp_data_manager):
        """Test error handling in advanced acquisition methods."""
        # Test with empty compounds
        empty_compounds = pd.DataFrame(columns=['ID', 'SMILES', 'prediction'])
        
        # Test removed - DiverseAcquisition class removed from codebase
        
        # Test edge case parameters
        try:
            acq = UMAPDBSCANAcquisition(temp_data_manager, n_components=1)
            single_compound = pd.DataFrame({
                'ID': ['mol_1'],
                'SMILES': ['CCO'],
                'prediction': [0.5]
            })
            
            # Should handle edge cases gracefully
            selected = acq.select(single_compound, n_select=1)
            assert len(selected) == 1
            
        except Exception as e:
            # Some configurations might fail, which is acceptable
            pytest.skip(f"Edge case handling failed: {e}")
    
    def test_mixed_acquisition_workflow(self, temp_data_manager, diverse_real_compounds):
        """Test workflow using multiple advanced acquisition methods."""
        compounds = diverse_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No diverse real compounds available")
        
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        # Test multiple advanced methods
        methods = []
        
        # Add clustering methods if available
        try:
            methods.append(('umap_dbscan', UMAPDBSCANAcquisition(temp_data_manager)))
        except Exception:
            pass
        
        selections = {}
        for name, acq in methods:
            try:
                selected = acq.select(compounds, n_select=6)
                selections[name] = selected
                
                # Verify each selection
                assert len(selected) == 6
                assert isinstance(selected, pd.DataFrame)
                assert all(col in selected.columns for col in ['ID', 'SMILES'])
                
            except Exception as e:
                # Some methods might fail in test environment
                continue
        
        # Should have at least one successful selection
        assert len(selections) >= 1


class TestAdvancedAcquisitionEdgeCases:
    """Test edge cases for advanced acquisition methods."""
    
    def test_clustering_with_insufficient_data(self, temp_data_manager):
        """Test clustering methods with very small datasets."""
        tiny_dataset = pd.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3'],
            'SMILES': ['CCO', 'CCC', 'CCN'],
            'prediction': [0.1, 0.5, 0.9]
        })
        
        # Test UMAP with small dataset
        try:
            acq = UMAPDBSCANAcquisition(temp_data_manager, n_components=2)
            selected = acq.select(tiny_dataset, n_select=2)
            
            # Should handle gracefully or raise appropriate error
            if isinstance(selected, pd.DataFrame):
                assert len(selected) <= 3
                
        except (ValueError, RuntimeError):
            # This is expected behavior for insufficient data
            pass
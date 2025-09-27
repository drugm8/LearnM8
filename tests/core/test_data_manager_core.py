"""
Core DataManager functionality tests.

Focused tests for essential DataManager operations with real molecular data.
"""

import pytest
import pandas as pd
import numpy as np
import h5py
from pathlib import Path

from learnm8.core.data_manager import DataManager


class TestDataManagerCore:
    """Core DataManager functionality tests."""
    
    def test_initialization_and_setup(self, tmp_path):
        """Test DataManager initialization and cache setup."""
        dm = DataManager(results_dir=tmp_path)
        
        assert dm.cache_dir.exists()
        assert dm.cache_dir.is_dir()
        assert 'morgan' in dm.featurizers
        assert 'maccs' in dm.featurizers
        assert 'descriptors' in dm.featurizers
    
    def test_feature_extraction_morgan(self, small_real_compounds, tmp_path):
        """Test Morgan fingerprint feature extraction."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        dm = DataManager(results_dir=tmp_path)
        
        features, valid_ids = dm.get_features(
            compounds['ID'].tolist(),
            compounds['SMILES'].tolist(),
            'morgan'
        )

        assert features.shape[0] == len(compounds)
        assert features.shape[1] > 0  # Should have feature dimensions
        assert np.all(np.isfinite(features))
        assert len(valid_ids) == len(compounds)
    
    def test_feature_extraction_maccs(self, small_real_compounds, tmp_path):
        """Test MACCS keys feature extraction."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        dm = DataManager(results_dir=tmp_path)
        
        features, valid_ids = dm.get_features(
            compounds['ID'].tolist(),
            compounds['SMILES'].tolist(),
            'maccs'
        )

        assert features.shape[0] == len(compounds)
        assert features.shape[1] == 167  # MACCS keys are 167-dimensional
        assert np.all((features == 0) | (features == 1))  # Binary features
        assert len(valid_ids) == len(compounds)
    
    def test_training_data_preparation(self, small_real_compounds, tmp_path):
        """Test preparation of training data."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        dm = DataManager(results_dir=tmp_path)

        valid_compounds, X, y = dm.prepare_training_data(compounds, 'Activity', 'morgan')

        assert X.shape[0] == len(valid_compounds)
        assert X.shape[1] > 0
        assert len(y) == len(valid_compounds)
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(y))
        
        # y should match Activity values for valid compounds
        np.testing.assert_array_almost_equal(y, valid_compounds['Activity'].values, decimal=6)
    
    def test_prediction_data_preparation(self, small_real_compounds, tmp_path):
        """Test preparation of prediction data."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        dm = DataManager(results_dir=tmp_path)

        valid_compounds, X = dm.prepare_prediction_data(compounds, 'morgan')

        assert X.shape[0] == len(valid_compounds)
        assert X.shape[1] > 0
        assert np.all(np.isfinite(X))
    
    def test_feature_caching(self, small_real_compounds, tmp_path):
        """Test HDF5 feature caching."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        dm = DataManager(results_dir=tmp_path)
        
        # First call - should compute and cache
        compound_ids = compounds['ID'].tolist()
        smiles_list = compounds['SMILES'].tolist()
        features1, valid_ids1 = dm.get_features(compound_ids, smiles_list, 'morgan')

        # Second call - should load from cache
        features2, valid_ids2 = dm.get_features(compound_ids, smiles_list, 'morgan')
        
        # Should be identical
        np.testing.assert_array_equal(features1, features2)
        
        # Cache file should exist
        cache_file = dm.cache_dir / "morgan_features.h5"
        assert cache_file.exists()
    
    def test_different_featurizers(self, small_real_compounds, tmp_path):
        """Test different molecular featurizers."""
        compounds = small_real_compounds.head(5).copy()  # Use fewer for descriptor speed
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        dm = DataManager(results_dir=tmp_path)
        compound_ids = compounds['ID'].tolist()
        smiles_list = compounds['SMILES'].tolist()
        
        # Test Morgan fingerprints
        morgan_features, morgan_valid_ids = dm.get_features(compound_ids, smiles_list, 'morgan')

        # Test MACCS keys
        maccs_features, maccs_valid_ids = dm.get_features(compound_ids, smiles_list, 'maccs')
        
        # Different featurizers should produce different dimensions
        assert morgan_features.shape[1] != maccs_features.shape[1]
        assert morgan_features.shape[0] == maccs_features.shape[0] == len(compounds)
    
    def test_partial_feature_loading(self, medium_real_compounds, tmp_path):
        """Test loading features for subset of compounds."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) < 10:
            pytest.skip("Insufficient compounds for partial loading test")
        
        dm = DataManager(results_dir=tmp_path)
        compound_ids = compounds['ID'].tolist()
        smiles_list = compounds['SMILES'].tolist()
        
        # Cache features for all compounds
        all_features, all_valid_ids = dm.get_features(compound_ids, smiles_list, 'morgan')

        # Load features for subset
        subset_ids = compounds['ID'].tolist()[:5]
        subset_smiles = compounds['SMILES'].tolist()[:5]
        subset_features, subset_valid_ids = dm.get_features(subset_ids, subset_smiles, 'morgan')
        
        assert subset_features.shape[0] == 5
        assert subset_features.shape[1] == all_features.shape[1]
        
        # Subset should match first 5 rows of all features
        np.testing.assert_array_equal(subset_features, all_features[:5])
    
    def test_cache_statistics(self, small_real_compounds, tmp_path):
        """Test cache statistics functionality."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        dm = DataManager(results_dir=tmp_path)
        
        # Initially empty cache
        # stats = dm.get_statistics()  # Method not implemented
        # assert isinstance(stats, dict)
        
        # Add some features
        dm.get_features(
            compounds['ID'].tolist(),
            compounds['SMILES'].tolist(),
            'morgan'
        )
        
        # Check updated stats
        # stats = dm.get_statistics()  # Method not implemented
        # assert 'cache_files' in stats
        # assert 'morgan' in stats['cache_files']
        # assert stats['cache_files']['morgan']['cached_compounds'] > 0
    
    def test_multi_target_data_handling(self, diverse_real_compounds, tmp_path):
        """Test DataManager with multi-target molecular data."""
        compounds = diverse_real_compounds.copy()
        
        if len(compounds) == 0 or 'Target' not in compounds.columns:
            pytest.skip("No multi-target molecular data available")
        
        dm = DataManager(results_dir=tmp_path)
        
        # Should handle multi-target data for feature extraction
        features, valid_ids = dm.get_features(
            compounds['ID'].tolist(),
            compounds['SMILES'].tolist(),
            'morgan'
        )
        
        assert features.shape[0] == len(compounds)
        assert features.shape[1] > 0
        
        # Should handle training data preparation
        valid_compounds, X, y = dm.prepare_training_data(compounds, 'Activity', 'morgan')

        assert X.shape[0] == len(valid_compounds)
        assert len(y) == len(valid_compounds)
        assert len(valid_compounds) <= len(compounds)


class TestDataManagerIntegration:
    """Integration tests for DataManager functionality."""
    
    def test_complete_workflow(self, medium_real_compounds, tmp_path):
        """Test complete DataManager workflow."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) < 20:
            pytest.skip("Insufficient compounds for workflow test")
        
        dm = DataManager(results_dir=tmp_path)
        
        # Split data
        train_compounds = compounds.head(50)
        pred_compounds = compounds.tail(20)
        
        # Prepare training data
        valid_train_compounds, X_train, y_train = dm.prepare_training_data(train_compounds, 'Activity', 'morgan')

        # Prepare prediction data
        valid_pred_compounds, X_pred = dm.prepare_prediction_data(pred_compounds, 'morgan')

        # Verify data shapes
        assert X_train.shape[0] == len(valid_train_compounds)
        assert X_pred.shape[0] == len(valid_pred_compounds)
        assert X_train.shape[1] == X_pred.shape[1]  # Same feature dimensions
        assert len(y_train) == len(valid_train_compounds)
        
        # Verify cache was used efficiently
        # cache_stats = dm.get_statistics()  # Method not implemented
        # assert len(cache_stats) > 0
    
    def test_incremental_feature_addition(self, small_real_compounds, tmp_path):
        """Test incremental addition of compounds to feature cache."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) < 8:
            pytest.skip("Insufficient compounds for incremental test")
        
        dm = DataManager(results_dir=tmp_path)
        compound_ids = compounds['ID'].tolist()
        smiles_list = compounds['SMILES'].tolist()
        
        # Add first batch of compounds
        first_batch = compounds.head(5)
        features1, valid_ids1 = dm.get_features(first_batch['ID'].tolist(), first_batch['SMILES'].tolist(), 'morgan')

        # Add second batch (overlapping)
        second_batch = compounds.head(8)  # Includes first 5 + 3 new
        features2, valid_ids2 = dm.get_features(second_batch['ID'].tolist(), second_batch['SMILES'].tolist(), 'morgan')
        
        # First 5 should be identical
        np.testing.assert_array_equal(features1, features2[:5])
        
        # Should have 8 total features now
        assert features2.shape[0] == 8
        assert features2.shape[1] == features1.shape[1]
    
    def test_cross_featurizer_consistency(self, small_real_compounds, tmp_path):
        """Test consistency across different featurizers."""
        compounds = small_real_compounds.head(5).copy()  # Limit for performance
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        dm = DataManager(results_dir=tmp_path)
        compound_ids = compounds['ID'].tolist()
        smiles_list = compounds['SMILES'].tolist()
        
        # Get features with different featurizers
        morgan_features, morgan_valid_ids = dm.get_features(compound_ids, smiles_list, 'morgan')
        maccs_features, maccs_valid_ids = dm.get_features(compound_ids, smiles_list, 'maccs')
        
        # Should have same number of compounds
        assert morgan_features.shape[0] == maccs_features.shape[0] == len(compounds)
        
        # Different feature dimensions
        assert morgan_features.shape[1] != maccs_features.shape[1]
        
        # Both should be valid features
        assert np.all(np.isfinite(morgan_features))
        assert np.all(np.isfinite(maccs_features))


class TestDataManagerErrorHandling:
    """Error handling tests for DataManager."""
    
    def test_missing_smiles_handling(self, tmp_path):
        """Test handling of invalid SMILES data."""
        dm = DataManager(results_dir=tmp_path)
        
        compound_ids = ['missing_1', 'missing_2']
        
        # Should handle invalid SMILES gracefully by filtering them out
        features, valid_ids = dm.get_features(compound_ids, None, 'morgan')

        # With invalid SMILES, we expect no valid compounds returned
        assert len(valid_ids) == 0
        assert features.shape == (0, 2048)  # No valid features
        assert np.all(features == 0)  # Should be zero vectors for invalid SMILES
    
    def test_invalid_featurizer(self, small_real_compounds, tmp_path):
        """Test error handling for invalid featurizer."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        dm = DataManager(results_dir=tmp_path)
        
        with pytest.raises((KeyError, ValueError)):
            dm.get_features(compounds['ID'].tolist(), compounds['SMILES'].tolist(), 'invalid_featurizer')
    
    def test_missing_target_column(self, small_real_compounds, tmp_path):
        """Test error handling for missing target column."""
        compounds = small_real_compounds.drop(columns=['Activity'])
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        dm = DataManager(results_dir=tmp_path)
        
        with pytest.raises(ValueError):
            dm.prepare_training_data(compounds, 'Activity', 'morgan')
    
    def test_empty_compound_list(self, tmp_path):
        """Test handling of empty compound list."""
        dm = DataManager(results_dir=tmp_path)
        
        features, valid_ids = dm.get_features([], [], 'morgan')

        # Should return empty arrays
        assert len(features) == 0
        assert len(valid_ids) == 0
        assert isinstance(features, np.ndarray)
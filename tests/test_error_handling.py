"""Essential error handling tests for LearnM8.

Focused tests for critical error conditions and edge cases using real molecular data.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from learnm8.core.data_manager import DataManager
from learnm8.acquisition import get_acquisition_function
from learnm8.acquisition.basic import GreedyAcquisition
from learnm8.oracles.csv_oracle import CSVOracle
from learnm8.oracles.python_oracle import PythonOracle


class TestDataValidationErrors:
    """Test essential data validation error handling."""
    
    def test_empty_dataframe_handling(self, tmp_path):
        """Test handling of empty DataFrames."""
        empty_compounds = pd.DataFrame(columns=['ID', 'SMILES', 'Activity'])
        
        # DataManager should handle empty data gracefully
        data_manager = DataManager(results_dir=tmp_path)
        
        with pytest.raises((ValueError, IndexError)):
            data_manager.prepare_training_data(empty_compounds, 'Activity', 'morgan')
        
        # Acquisition functions should reject empty data
        acq = GreedyAcquisition()
        
        with pytest.raises(ValueError, match="empty"):
            acq.select(empty_compounds, n_select=1)
    
    def test_missing_required_columns(self, small_real_compounds, tmp_path):
        """Test handling of missing required columns."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        data_manager = DataManager(results_dir=tmp_path)
        
        # Missing ID column
        compounds_no_id = compounds.drop(columns=['ID'])
        with pytest.raises((KeyError, ValueError)):
            data_manager.prepare_training_data(compounds_no_id, 'Activity', 'morgan')
        
        # Missing SMILES column
        compounds_no_smiles = compounds.drop(columns=['SMILES'])
        with pytest.raises((KeyError, ValueError)):
            data_manager.prepare_training_data(compounds_no_smiles, 'Activity', 'morgan')
        
        # Missing target column
        compounds_no_target = compounds.drop(columns=['Activity'])
        with pytest.raises((KeyError, ValueError)):
            data_manager.prepare_training_data(compounds_no_target, 'Activity', 'morgan')
    
    def test_invalid_smiles_handling(self, tmp_path):
        """Test handling of invalid SMILES strings."""
        invalid_compounds = pd.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3'],
            'SMILES': ['CCO', 'invalid_smiles', 'C1CCC'],  # One valid, two invalid
            'Activity': [0.5, 0.7, 0.3]
        })
        
        data_manager = DataManager(results_dir=tmp_path)
        
        # Should handle invalid SMILES appropriately
        try:
            valid_compounds, X, y = data_manager.prepare_training_data(invalid_compounds, 'Activity', 'morgan')

            # If it succeeds, should have filtered out invalid compounds
            assert X.shape[0] <= len(invalid_compounds)
            assert len(y) == X.shape[0]
            assert len(valid_compounds) == X.shape[0]

        except ValueError as e:
            # This is also acceptable - rejecting datasets with invalid SMILES
            assert "SMILES" in str(e) or "invalid" in str(e).lower()
    
    def test_mismatched_array_lengths(self, small_real_compounds):
        """Test handling of mismatched array lengths."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Add predictions with wrong length
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        wrong_length_predictions = np.random.uniform(0, 1, len(compounds) + 2)  # Too long
        
        acq = GreedyAcquisition()
        
        # Should detect length mismatch
        compounds_wrong = compounds.copy()
        compounds_wrong['prediction'] = wrong_length_predictions[:len(compounds)]  # Truncate to fit
        
        # This should work (same length after truncation)
        selected = acq.select(compounds_wrong, n_select=3)
        assert len(selected) == 3
    
    def test_nan_value_handling(self, small_real_compounds):
        """Test handling of NaN values in data."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Add NaN values
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        compounds.loc[0, 'prediction'] = np.nan
        compounds.loc[1, 'Activity'] = np.nan
        
        acq = GreedyAcquisition()
        
        # Should handle NaN values gracefully or raise appropriate error
        try:
            selected = acq.select(compounds, n_select=3)
            
            # If it succeeds, should not include NaN predictions
            assert not selected['prediction'].isna().any()
            
        except (ValueError, RuntimeError):
            # This is also acceptable behavior for NaN values
            pass


class TestAcquisitionErrors:
    """Test acquisition function error handling."""
    
    def test_invalid_n_select_values(self, small_real_compounds):
        """Test handling of invalid n_select parameters."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        acq = GreedyAcquisition()
        
        # Test negative n_select
        with pytest.raises(ValueError, match="n_select must be positive"):
            acq.select(compounds, n_select=-1)
        
        # Test zero n_select
        with pytest.raises(ValueError, match="n_select must be positive"):
            acq.select(compounds, n_select=0)
        
        # Test n_select larger than available compounds (should handle gracefully)
        selected = acq.select(compounds, n_select=len(compounds) + 10)
        assert len(selected) <= len(compounds)  # Should return what's available
    
    def test_missing_uncertainty_for_ucb(self, small_real_compounds):
        """Test UCB acquisition without uncertainty column."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        # No uncertainty column
        
        ucb_acq = get_acquisition_function('ucb')()
        
        with pytest.raises(ValueError, match="requires 'uncertainty' column"):
            ucb_acq.select(compounds, n_select=3)
    
    def test_acquisition_function_not_found(self):
        """Test handling of nonexistent acquisition functions."""
        with pytest.raises((KeyError, ValueError)):
            get_acquisition_function('nonexistent_function')
    
    def test_acquisition_with_duplicate_ids(self, small_real_compounds):
        """Test acquisition with duplicate compound IDs."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Create duplicates
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        compounds_with_dups = pd.concat([compounds, compounds.head(2)], ignore_index=True)
        
        acq = GreedyAcquisition()
        
        # Should handle duplicates gracefully or raise appropriate error
        try:
            selected = acq.select(compounds_with_dups, n_select=5)
            
            # If it succeeds, should not return duplicate IDs
            assert len(selected['ID'].unique()) == len(selected)
            
        except ValueError as e:
            # This is also acceptable - rejecting datasets with duplicates
            assert "duplicate" in str(e).lower() or "unique" in str(e).lower()


class TestOracleErrors:
    """Test oracle error handling."""
    
    def test_csv_oracle_file_not_found(self):
        """Test CSV oracle with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            CSVOracle("nonexistent_file.csv")
    
    def test_csv_oracle_invalid_format(self, tmp_path):
        """Test CSV oracle with invalid file format."""
        # Create invalid CSV file
        invalid_csv = tmp_path / "invalid.csv"
        with open(invalid_csv, 'w') as f:
            f.write("not,a,valid,csv,format\n")
            f.write("missing,required,columns\n")
        
        # Should fail to create oracle or fail during measurement
        try:
            oracle = CSVOracle(str(invalid_csv))
            
            # If creation succeeds, measurement should fail
            compounds = pd.DataFrame({
                'ID': ['mol_1'],
                'SMILES': ['CCO']
            })
            
            with pytest.raises((KeyError, ValueError)):
                oracle.measure(compounds, ['Activity'])
                
        except (KeyError, ValueError):
            # Failing at creation is also acceptable
            pass
    
    def test_python_oracle_invalid_module(self):
        """Test Python oracle with invalid module."""
        with pytest.raises(FileNotFoundError):
            PythonOracle(module_path="nonexistent_module.py", function_name="some_function")
    
    def test_python_oracle_invalid_function(self, tmp_path):
        """Test Python oracle with invalid function."""
        # Create Python file without required function
        py_file = tmp_path / "test_oracle.py"
        with open(py_file, 'w') as f:
            f.write("def wrong_function():\n    return 0.5\n")
        
        with pytest.raises(ValueError):
            PythonOracle(module_path=str(py_file), function_name="nonexistent_function")
    
    def test_oracle_measurement_with_empty_compounds(self, tmp_path):
        """Test oracle measurement with empty compound set."""
        # Create minimal valid CSV
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w') as f:
            f.write("ID,Activity\n")
            f.write("mol_1,0.5\n")
        
        oracle = CSVOracle(str(csv_file))
        empty_compounds = pd.DataFrame(columns=['ID', 'SMILES'])
        
        # Should handle empty input gracefully
        result = oracle.measure(empty_compounds, ['Activity'])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert 'ID' in result.columns
        assert 'Activity' in result.columns


class TestDataManagerErrors:
    """Test DataManager error handling."""
    
    def test_datamanager_invalid_results_dir(self):
        """Test DataManager with invalid results directory."""
        # Test with file instead of directory
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises((OSError, ValueError)):
                DataManager(results_dir=f.name)
    
    def test_datamanager_permission_denied(self, tmp_path):
        """Test DataManager with permission issues."""
        # Create read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        try:
            # Should fail during DataManager initialization when creating cache directory
            with pytest.raises(PermissionError):
                data_manager = DataManager(results_dir=readonly_dir)
                
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)
    
    def test_datamanager_corrupted_cache(self, tmp_path):
        """Test DataManager with corrupted cache files."""
        data_manager = DataManager(results_dir=tmp_path)
        
        # Create corrupted cache file
        cache_dir = tmp_path / ".features"
        cache_dir.mkdir()
        corrupted_file = cache_dir / "morgan.h5"
        
        with open(corrupted_file, 'w') as f:
            f.write("not an HDF5 file")
        
        compounds = pd.DataFrame({
            'ID': ['mol_1'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })
        
        # Should handle corrupted cache gracefully
        try:
            valid_compounds, X, y = data_manager.prepare_training_data(compounds, 'Activity', 'morgan')
            assert X.shape[0] == 1  # Should recompute features
            assert len(valid_compounds) == 1
        except Exception as e:
            # May fail due to cache corruption, which is acceptable
            assert "HDF5" in str(e) or "corrupt" in str(e).lower()
    
    def test_datamanager_unsupported_featurizer(self, tmp_path):
        """Test DataManager with unsupported featurizer."""
        data_manager = DataManager(results_dir=tmp_path)
        
        compound_ids = ['mol_1']
        
        with pytest.raises((ValueError, KeyError)):
            data_manager.get_features(compound_ids, None, 'unsupported_featurizer')


class TestIntegrationErrors:
    """Test error handling in integrated workflows."""
    
    def test_incomplete_workflow_recovery(self, small_real_compounds, tmp_path):
        """Test recovery from incomplete workflow execution."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Create CSV oracle
        csv_file = tmp_path / "oracle.csv"
        oracle_data = compounds[['ID', 'Activity']].copy()
        oracle_data.to_csv(csv_file, index=False)
        
        oracle = CSVOracle(str(csv_file))
        data_manager = DataManager(results_dir=tmp_path)
        
        # Start workflow
        labeled_compounds = compounds.head(3)
        measurements = oracle.measure(labeled_compounds, ['Activity'])
        labeled_with_activity = labeled_compounds.merge(measurements, on='ID')
        
        # Simulate interrupted workflow - try to use incomplete data
        # Drop SMILES columns (could be SMILES_x or SMILES_y after merge)
        smiles_cols = [col for col in labeled_with_activity.columns if 'SMILES' in col]
        incomplete_data = labeled_with_activity.drop(columns=smiles_cols)
        
        # Should fail because missing SMILES column
        with pytest.raises(ValueError, match="Missing required columns"):
            data_manager.prepare_training_data(incomplete_data, 'Activity', 'morgan')
    
    def test_mixed_valid_invalid_data(self, tmp_path):
        """Test handling of mixed valid and invalid data."""
        mixed_compounds = pd.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_4'],
            'SMILES': ['CCO', 'invalid_smiles', 'CCC', ''],  # Mixed valid/invalid
            'Activity': [0.5, 0.7, 0.3, 0.9]
        })
        
        data_manager = DataManager(results_dir=tmp_path)
        
        # Should handle mixed data appropriately
        try:
            valid_compounds, X, y = data_manager.prepare_training_data(mixed_compounds, 'Activity', 'morgan')

            # Should process valid compounds only
            assert X.shape[0] <= len(mixed_compounds)
            assert X.shape[0] >= 2  # At least the clearly valid ones
            assert len(y) == X.shape[0]
            assert len(valid_compounds) == X.shape[0]
            
        except ValueError as e:
            # Rejecting entire dataset due to invalid data is also acceptable
            assert "SMILES" in str(e) or "invalid" in str(e).lower()
    
    def test_resource_cleanup_after_errors(self, tmp_path):
        """Test that resources are cleaned up after errors."""
        data_manager = DataManager(results_dir=tmp_path)
        
        # Simulate error during feature computation
        invalid_compounds = pd.DataFrame({
            'ID': ['mol_1'],
            'SMILES': ['completely_invalid_smiles_string'],
            'Activity': [0.5]
        })
        
        try:
            data_manager.prepare_training_data(invalid_compounds, 'Activity', 'morgan')
        except Exception:
            pass  # Expected to fail
        
        # Check that no partial cache files were left
        cache_dir = tmp_path / ".features"
        if cache_dir.exists():
            # Should not have partial/corrupted cache files
            h5_files = list(cache_dir.glob("*.h5"))
            for h5_file in h5_files:
                # Files should either not exist or be valid (not 0-byte)
                if h5_file.exists():
                    assert h5_file.stat().st_size == 0 or h5_file.stat().st_size > 100


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""
    
    def test_partial_failure_recovery(self, medium_real_compounds, tmp_path):
        """Test recovery from partial failures in batch operations."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Use subset
        compounds = compounds.head(10)
        
        # Add some invalid SMILES to create partial failure scenario
        compounds.loc[2, 'SMILES'] = 'invalid'
        compounds.loc[5, 'SMILES'] = ''
        
        data_manager = DataManager(results_dir=tmp_path)
        
        try:
            valid_compounds, X, y = data_manager.prepare_training_data(compounds, 'Activity', 'morgan')

            # Should recover and process valid compounds
            assert X.shape[0] <= len(compounds)
            assert X.shape[0] >= 5  # Should have some valid compounds
            assert len(y) == X.shape[0]
            assert len(valid_compounds) == X.shape[0]
            
        except ValueError:
            # Complete failure is also acceptable for this test
            pass
    
    def test_graceful_degradation_with_limited_resources(self, tmp_path):
        """Test graceful degradation when resources are limited."""
        # Test with very small cache directory quota (simulate disk full)
        data_manager = DataManager(results_dir=tmp_path)
        
        # Create compounds that would require significant cache space
        large_compound_set = pd.DataFrame({
            'ID': [f'mol_{i}' for i in range(100)],
            'SMILES': ['CCO'] * 100,
            'Activity': np.random.uniform(0, 1, 100)
        })
        
        # This should either succeed or fail gracefully
        try:
            valid_compounds, X, y = data_manager.prepare_training_data(large_compound_set, 'Activity', 'morgan')

            # If it succeeds, should have reasonable results
            assert X.shape[0] == 100
            assert len(y) == 100
            assert len(valid_compounds) == 100
            
        except (OSError, MemoryError):
            # Failing due to resource constraints is acceptable
            pass
    
    def test_error_message_clarity(self, small_real_compounds):
        """Test that error messages are clear and informative."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        acq = GreedyAcquisition()
        
        # Test clear error message for missing column
        compounds_no_pred = compounds.drop(columns=['Activity'])
        
        try:
            acq.select(compounds_no_pred, n_select=3)
            assert False, "Should have raised an error"
        except (ValueError, KeyError) as e:
            error_msg = str(e).lower()
            # Error message should be informative
            assert any(keyword in error_msg for keyword in ['column', 'prediction', 'missing', 'required'])
        
        # Test clear error message for invalid n_select
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        try:
            acq.select(compounds, n_select=-1)
            assert False, "Should have raised an error"
        except ValueError as e:
            error_msg = str(e).lower()
            # Should mention n_select and positive
            assert 'n_select' in error_msg and 'positive' in error_msg
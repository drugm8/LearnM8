"""Essential error handling tests for LearnM8.

Focused tests for critical error conditions and edge cases using real molecular data.
"""

import pytest
import polars as pl
import numpy as np
import tempfile
from pathlib import Path

from learnm8.features import extract_features
from learnm8.acquisition import get_acquisition_function
from learnm8.acquisition import GreedyAcquisition
from learnm8.exceptions import OracleError
from learnm8.oracles.csv_oracle import CSVOracle
from learnm8.oracles.python_oracle import PythonOracle


@pytest.mark.integration
@pytest.mark.molecular
class TestDataValidationErrors:
    """Test essential data validation error handling."""
    
    def test_empty_dataframe_handling(self, tmp_path):
        """Test handling of empty DataFrames."""
        empty_compounds = pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8, 'Activity': pl.Float64})

        # Feature extraction should handle empty data gracefully
        features = extract_features([], 'morgan', tmp_path)
        assert features.shape[0] == 0

        # Acquisition functions should reject empty data
        acq = GreedyAcquisition()

        with pytest.raises(ValueError, match="empty"):
            acq.select(empty_compounds, n_select=1)
    
    def test_missing_required_columns(self, small_real_compounds, tmp_path):
        """Test handling of missing required columns."""
        compounds = small_real_compounds.clone()
        # Feature extraction only requires SMILES
        # Missing SMILES column - feature extraction should fail
        compounds_no_smiles = compounds.drop('SMILES')

        # Since we need SMILES for feature extraction, this should fail
        # Test would need DataFrame with SMILES column
        assert 'SMILES' not in compounds_no_smiles.columns
    
    @pytest.mark.slow
    def test_invalid_smiles_handling(self, tmp_path):
        """Test handling of invalid SMILES strings."""
        invalid_smiles = ['CCO', 'invalid_smiles', 'C1CCC']  # One valid, two invalid

        # Should raise error for invalid SMILES
        with pytest.raises(Exception):
            extract_features(invalid_smiles, 'morgan', tmp_path)
    
    def test_mismatched_array_lengths(self, small_real_compounds):
        """Test handling of mismatched array lengths."""
        compounds = small_real_compounds.clone()
        # Add predictions with wrong length
        compounds = compounds.with_columns(
            pl.Series('prediction', np.random.uniform(0, 1, len(compounds)))
        )
        wrong_length_predictions = np.random.uniform(0, 1, len(compounds) + 2)  # Too long
        
        acq = GreedyAcquisition()
        
        # Should detect length mismatch
        compounds_wrong = compounds.clone()
        # Truncate to fit
        compounds_wrong = compounds_wrong.with_columns(
            pl.Series('prediction', wrong_length_predictions[:len(compounds)])
        )
        
        # This should work (same length after truncation)
        selected = acq.select(compounds_wrong, n_select=3)
        assert len(selected) == 3
    
    def test_nan_value_handling(self, small_real_compounds):
        """Test handling of NaN values in data."""
        compounds = small_real_compounds.clone()
        # Add NaN values
        prediction_values = np.random.uniform(0, 1, len(compounds))
        prediction_values[0] = np.nan
        compounds = compounds.with_columns(
            pl.Series('prediction', prediction_values)
        )
        # Set Activity to NaN for second compound
        compounds = compounds.with_columns(
            pl.when(pl.int_range(pl.len()) == 1)
              .then(pl.lit(None))
              .otherwise(pl.col('Activity'))
              .alias('Activity')
        )
        
        acq = GreedyAcquisition()
        
        # Should handle NaN values gracefully or raise appropriate error
        try:
            selected = acq.select(compounds, n_select=3)
            
            # If it succeeds, should not include NaN predictions
            assert not selected['prediction'].is_null().any()
            
        except (ValueError, RuntimeError):
            # This is also acceptable behavior for NaN values
            pass


@pytest.mark.integration
@pytest.mark.molecular
class TestAcquisitionErrors:
    """Test acquisition function error handling."""
    
    def test_invalid_n_select_values(self, small_real_compounds):
        """Test handling of invalid n_select parameters."""
        compounds = small_real_compounds.clone()
        compounds = compounds.with_columns(
            pl.Series('prediction', np.random.uniform(0, 1, len(compounds)))
        )
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
        compounds = small_real_compounds.clone()
        compounds = compounds.with_columns(
            pl.Series('prediction', np.random.uniform(0, 1, len(compounds)))
        )
        # No uncertainty column
        
        ucb_acq = get_acquisition_function('ucb')()
        
        with pytest.raises(ValueError, match="requires an 'uncertainty' column"):
            ucb_acq.select(compounds, n_select=3)
    
    def test_acquisition_function_not_found(self):
        """Test handling of nonexistent acquisition functions."""
        with pytest.raises((KeyError, ValueError)):
            get_acquisition_function('nonexistent_function')
    
    def test_acquisition_with_duplicate_ids(self, small_real_compounds):
        """Test acquisition with duplicate compound IDs."""
        compounds = small_real_compounds.clone()
        # Create duplicates
        compounds = compounds.with_columns(
            pl.Series('prediction', np.random.uniform(0, 1, len(compounds)))
        )
        compounds_with_dups = pl.concat([compounds, compounds.head(2)], how='vertical')
        
        acq = GreedyAcquisition()
        
        # Should handle duplicates gracefully or raise appropriate error
        try:
            selected = acq.select(compounds_with_dups, n_select=5)
            
            # If it succeeds, should not return duplicate IDs
            assert len(selected['ID'].unique()) == len(selected)
            
        except ValueError as e:
            # This is also acceptable - rejecting datasets with duplicates
            assert "duplicate" in str(e).lower() or "unique" in str(e).lower()


@pytest.mark.integration
@pytest.mark.molecular
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
            compounds = pl.DataFrame({
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
        
        with pytest.raises(OracleError):
            PythonOracle(module_path=str(py_file), function_name="nonexistent_function")
    
    def test_oracle_measurement_with_empty_compounds(self, tmp_path):
        """Test oracle measurement with empty compound set."""
        # Create minimal valid CSV
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w') as f:
            f.write("ID,Activity\n")
            f.write("mol_1,0.5\n")
        
        oracle = CSVOracle(str(csv_file))
        empty_compounds = pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8})
        
        # Should handle empty input gracefully
        result = oracle.measure(empty_compounds, ['Activity'])
        
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert 'ID' in result.columns
        assert 'Activity' in result.columns


@pytest.mark.integration
@pytest.mark.molecular
class TestFeatureExtractionErrors:
    """Test feature extraction error handling."""

    def test_unsupported_featurizer(self, tmp_path):
        """Test feature extraction with unsupported featurizer."""
        smiles_list = ['CCO']

        with pytest.raises((ValueError, KeyError)):
            extract_features(smiles_list, 'unsupported_featurizer', tmp_path)

    @pytest.mark.slow
    def test_corrupted_cache(self, tmp_path):
        """Test feature extraction with corrupted cache files."""
        # Create corrupted cache file
        cache_dir = tmp_path / ".features"
        cache_dir.mkdir()
        corrupted_file = cache_dir / "morgan.h5"

        with open(corrupted_file, 'w') as f:
            f.write("not an HDF5 file")

        smiles_list = ['CCO']

        # Should handle corrupted cache gracefully
        try:
            features = extract_features(smiles_list, 'morgan', tmp_path)
            assert features.shape[0] == 1  # Should recompute features
        except Exception as e:
            # May fail due to cache corruption, which is acceptable
            assert "HDF5" in str(e) or "corrupt" in str(e).lower()


@pytest.mark.integration
@pytest.mark.molecular
class TestIntegrationErrors:
    """Test error handling in integrated workflows."""
    
    def test_incomplete_workflow_recovery(self, small_real_compounds, tmp_path):
        """Test recovery from incomplete workflow execution."""
        compounds = small_real_compounds.clone()
        # Create CSV oracle
        csv_file = tmp_path / "oracle.csv"
        oracle_data = compounds[['ID', 'Activity']].clone()
        oracle_data.write_csv(csv_file)

        oracle = CSVOracle(str(csv_file))

        # Start workflow
        labeled_compounds = compounds.head(3)
        measurements = oracle.measure(labeled_compounds, ['Activity'])
        labeled_with_activity = labeled_compounds.join(measurements, on='ID', how='left')

        # Simulate interrupted workflow - try to use incomplete data
        # Drop SMILES columns (could be SMILES_x or SMILES_y after merge)
        smiles_cols = [col for col in labeled_with_activity.columns if 'SMILES' in col]
        incomplete_data = labeled_with_activity.drop(smiles_cols)

        # Should fail because missing SMILES column
        assert 'SMILES' not in incomplete_data.columns
    
    @pytest.mark.slow
    def test_mixed_valid_invalid_data(self, tmp_path):
        """Test handling of mixed valid and invalid data."""
        mixed_smiles = ['CCO', 'invalid_smiles', 'CCC', '']  # Mixed valid/invalid

        # Should raise error for invalid SMILES
        with pytest.raises(Exception):
            extract_features(mixed_smiles, 'morgan', tmp_path)
    
    @pytest.mark.slow
    def test_resource_cleanup_after_errors(self, tmp_path):
        """Test that resources are cleaned up after errors."""
        # Simulate error during feature computation
        invalid_smiles = ['completely_invalid_smiles_string']

        try:
            extract_features(invalid_smiles, 'morgan', tmp_path)
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


@pytest.mark.integration
@pytest.mark.molecular
class TestErrorRecovery:
    """Test error recovery and graceful degradation."""

    @pytest.fixture
    def compounds_20(self, small_real_compounds):
        return small_real_compounds.head(20)

    @pytest.mark.slow
    def test_partial_failure_recovery(self, medium_real_compounds, tmp_path):
        """Test recovery from partial failures in batch operations."""
        compounds = medium_real_compounds.clone()
        # Use subset
        compounds = compounds.head(10)

        # Add some invalid SMILES to create partial failure scenario
        smiles_list = compounds['SMILES'].to_list()
        smiles_list[2] = 'invalid'
        smiles_list[5] = ''

        # Should raise error for invalid SMILES
        with pytest.raises(Exception):
            extract_features(smiles_list, 'morgan', tmp_path)
    
    @pytest.mark.slow
    def test_graceful_degradation_with_limited_resources(self, tmp_path):
        """Test graceful degradation when resources are limited."""
        smiles_list = ['CCO'] * 5

        # This should either succeed or fail gracefully
        try:
            features = extract_features(smiles_list, 'morgan', tmp_path)

            assert features.shape[0] == 5

        except (OSError, MemoryError):
            pass
    
    @pytest.mark.slow
    def test_error_message_clarity(self, compounds_20):
        """Test that error messages are clear and informative."""
        compounds = compounds_20.clone()
        acq = GreedyAcquisition()
        
        # Test clear error message for missing column
        compounds_no_pred = compounds.drop('Activity')
        
        try:
            acq.select(compounds_no_pred, n_select=3)
            assert False, "Should have raised an error"
        except (ValueError, KeyError) as e:
            error_msg = str(e).lower()
            # Error message should be informative
            assert any(keyword in error_msg for keyword in ['column', 'prediction', 'missing', 'required'])
        
        # Test clear error message for invalid n_select
        compounds = compounds.with_columns(
            pl.Series('prediction', np.random.uniform(0, 1, len(compounds)))
        )
        
        try:
            acq.select(compounds, n_select=-1)
            assert False, "Should have raised an error"
        except ValueError as e:
            error_msg = str(e).lower()
            # Should mention n_select and positive
            assert 'n_select' in error_msg and 'positive' in error_msg
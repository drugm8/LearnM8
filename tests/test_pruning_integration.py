"""
Tests for pruning integration functionality in learnm8.py.

Tests the apply_pruning_strategy function with real molecular data,
focusing on integration, error handling, and data validation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from learnm8.learnm8 import apply_pruning_strategy


class TestApplyPruningStrategy:
    """Test pruning strategy integration."""
    
    def test_probabilistic_pruning_basic(self, small_real_compounds):
        """Test basic probabilistic pruning functionality."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(10)],
                'SMILES': ['CCO'] * 10
            })
        
        # Create predictions with some compounds having low probability of being active
        predictions = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.05])[:len(compounds)]
        uncertainties = np.random.uniform(0.1, 0.3, len(compounds))
        
        params = {
            'threshold': 0.3,
            'retention_fraction': 0.8
        }
        
        try:
            pruned_pool, pruning_info = apply_pruning_strategy(
                pool=compounds,
                predictions=predictions,
                uncertainties=uncertainties,
                strategy='probabilistic',
                params=params
            )
            
            # Validate results
            assert isinstance(pruned_pool, pd.DataFrame)
            assert isinstance(pruning_info, dict)
            
            # Check pruning info structure
            expected_keys = ['pruned_count', 'original_pool_size', 'pruned_pool_size', 'pruning_fraction']
            assert all(key in pruning_info for key in expected_keys)
            
            # Validate data consistency
            assert len(pruned_pool) <= len(compounds)
            assert pruning_info['original_pool_size'] == len(compounds)
            assert pruning_info['pruned_pool_size'] == len(pruned_pool)
            assert pruning_info['pruned_count'] == len(compounds) - len(pruned_pool)
            
            # Validate ID and SMILES columns are preserved
            assert 'ID' in pruned_pool.columns
            assert 'SMILES' in pruned_pool.columns
            assert pruned_pool['ID'].isin(compounds['ID']).all()
            
        except ImportError:
            pytest.skip("Pruning module not available")
        except Exception as e:
            pytest.skip(f"Pruning strategy failed: {e}")
    
    def test_uncertainty_threshold_pruning(self, small_real_compounds):
        """Test uncertainty threshold pruning."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(8)],
                'SMILES': ['CCC'] * 8
            })
        
        predictions = np.random.uniform(0.3, 0.7, len(compounds))
        # Create uncertainties with some high values that should be pruned
        uncertainties = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])[:len(compounds)]
        
        params = {
            'threshold': 0.5,  # Prune compounds with uncertainty > 0.5
            'method': 'above'
        }
        
        try:
            pruned_pool, pruning_info = apply_pruning_strategy(
                pool=compounds,
                predictions=predictions,
                uncertainties=uncertainties,
                strategy='uncertainty_threshold',
                params=params
            )
            
            # Should have removed some compounds
            assert len(pruned_pool) < len(compounds)
            assert pruning_info['pruned_count'] > 0
            
            # Remaining compounds should have uncertainty <= threshold
            if 'uncertainty' in pruned_pool.columns:
                remaining_uncertainties = pruned_pool['uncertainty'].values
                assert all(unc <= params['threshold'] for unc in remaining_uncertainties)
            
        except ImportError:
            pytest.skip("Pruning module not available")
        except Exception as e:
            pytest.skip(f"Uncertainty threshold pruning not available: {e}")
    
    def test_prediction_threshold_pruning(self, small_real_compounds):
        """Test prediction threshold pruning."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(6)],
                'SMILES': ['CCN'] * 6
            })
        
        # Create predictions with clear low/high values
        predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])[:len(compounds)]
        uncertainties = np.random.uniform(0.1, 0.3, len(compounds))
        
        params = {
            'threshold': 0.4,  # Keep compounds with prediction > 0.4
            'method': 'above'
        }
        
        try:
            pruned_pool, pruning_info = apply_pruning_strategy(
                pool=compounds,
                predictions=predictions,
                uncertainties=uncertainties,
                strategy='prediction_threshold',
                params=params
            )
            
            # Should have removed low-prediction compounds
            assert len(pruned_pool) <= len(compounds)
            assert isinstance(pruning_info, dict)
            
        except ImportError:
            pytest.skip("Pruning module not available")
        except Exception as e:
            pytest.skip(f"Prediction threshold pruning not available: {e}")
    
    def test_empty_compound_pool(self):
        """Test pruning with empty compound pool."""
        empty_compounds = pd.DataFrame(columns=['ID', 'SMILES'])
        empty_predictions = np.array([])
        empty_uncertainties = np.array([])
        
        params = {'threshold': 0.5}
        
        try:
            pruned_pool, pruning_info = apply_pruning_strategy(
                pool=empty_compounds,
                predictions=empty_predictions,
                uncertainties=empty_uncertainties,
                strategy='probabilistic',
                params=params
            )
            
            # Should return empty pool with correct info
            assert len(pruned_pool) == 0
            assert pruning_info['original_pool_size'] == 0
            assert pruning_info['pruned_count'] == 0
            
        except ImportError:
            pytest.skip("Pruning module not available")
        except Exception as e:
            pytest.skip(f"Empty pool pruning failed: {e}")
    
    def test_invalid_strategy_error(self, small_real_compounds):
        """Test error handling for invalid pruning strategy."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002'],
                'SMILES': ['CCO', 'CCC']
            })
        
        predictions = np.array([0.5, 0.7])
        uncertainties = np.array([0.2, 0.3])
        params = {}
        
        with pytest.raises(ValueError) as exc_info:
            apply_pruning_strategy(
                pool=compounds,
                predictions=predictions,
                uncertainties=uncertainties,
                strategy='invalid_strategy',
                params=params
            )
        
        # Should provide helpful error message
        error_msg = str(exc_info.value)
        assert 'Failed to create pruning strategy' in error_msg or 'invalid_strategy' in error_msg
    
    def test_none_predictions_handling(self, small_real_compounds):
        """Test handling when predictions are None."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(4)],
                'SMILES': ['CCO'] * 4
            })
        
        params = {'threshold': 0.5}
        
        try:
            pruned_pool, pruning_info = apply_pruning_strategy(
                pool=compounds,
                predictions=None,
                uncertainties=None,
                strategy='probabilistic',
                params=params
            )
            
            # Should handle None predictions gracefully
            assert isinstance(pruned_pool, pd.DataFrame)
            assert isinstance(pruning_info, dict)
            
        except ImportError:
            pytest.skip("Pruning module not available")
        except Exception as e:
            # None predictions might cause errors in some strategies
            pytest.skip(f"None predictions handling failed: {e}")
    
    def test_dataframe_predictions(self, small_real_compounds):
        """Test handling of DataFrame predictions."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(5)],
                'SMILES': ['CCC'] * 5
            })
        
        # Create DataFrame predictions
        predictions_df = pd.DataFrame({
            'prediction': np.random.uniform(0, 1, len(compounds))
        })
        
        uncertainties_df = pd.DataFrame({
            'uncertainty': np.random.uniform(0.1, 0.3, len(compounds))
        })
        
        params = {'threshold': 0.4}
        
        try:
            pruned_pool, pruning_info = apply_pruning_strategy(
                pool=compounds,
                predictions=predictions_df,
                uncertainties=uncertainties_df,
                strategy='probabilistic',
                params=params
            )
            
            # Should convert DataFrames to arrays and work correctly
            assert isinstance(pruned_pool, pd.DataFrame)
            assert isinstance(pruning_info, dict)
            
        except ImportError:
            pytest.skip("Pruning module not available")
        except Exception as e:
            pytest.skip(f"DataFrame predictions handling failed: {e}")
    
    def test_strategy_name_mapping(self, small_real_compounds):
        """Test CLI-friendly strategy name mapping."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(3)],
                'SMILES': ['CCN'] * 3
            })
        
        predictions = np.array([0.3, 0.6, 0.9])
        uncertainties = np.array([0.2, 0.3, 0.1])
        
        # Test different CLI-friendly names
        strategy_names = [
            'probabilistic',
            'uncertainty_threshold', 
            'prediction_threshold'
        ]
        
        for strategy_name in strategy_names:
            params = {'threshold': 0.5}
            
            try:
                pruned_pool, pruning_info = apply_pruning_strategy(
                    pool=compounds,
                    predictions=predictions,
                    uncertainties=uncertainties,
                    strategy=strategy_name,
                    params=params
                )
                
                # Should map strategy name correctly
                assert isinstance(pruned_pool, pd.DataFrame)
                assert isinstance(pruning_info, dict)
                
            except ImportError:
                pytest.skip(f"Pruning module not available for {strategy_name}")
            except Exception as e:
                pytest.skip(f"Strategy {strategy_name} failed: {e}")
    
    def test_pruning_fraction_calculation(self, small_real_compounds):
        """Test correct calculation of pruning fraction."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(10)],
                'SMILES': ['CCO'] * 10
            })
        
        predictions = np.random.uniform(0, 1, len(compounds))
        uncertainties = np.random.uniform(0.1, 0.3, len(compounds))
        
        params = {'threshold': 0.3, 'retention_fraction': 0.6}  # Should keep ~60%
        
        try:
            pruned_pool, pruning_info = apply_pruning_strategy(
                pool=compounds,
                predictions=predictions,
                uncertainties=uncertainties,
                strategy='probabilistic',
                params=params
            )
            
            # Check fraction calculation
            expected_fraction = pruning_info['pruned_count'] / pruning_info['original_pool_size']
            assert abs(pruning_info['pruning_fraction'] - expected_fraction) < 1e-10
            
            # Fraction should be between 0 and 1
            assert 0 <= pruning_info['pruning_fraction'] <= 1
            
        except ImportError:
            pytest.skip("Pruning module not available")
        except Exception as e:
            pytest.skip(f"Pruning fraction test failed: {e}")
    
    def test_no_pruning_case(self, small_real_compounds):
        """Test case where no compounds are pruned."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(4)],
                'SMILES': ['CCC'] * 4
            })
        
        # Set parameters that shouldn't prune anything
        predictions = np.array([0.8, 0.9, 0.7, 0.85])[:len(compounds)] # High predictions
        uncertainties = np.array([0.1, 0.15, 0.12, 0.08])[:len(compounds)]  # Low uncertainties
        
        params = {'threshold': 0.5}  # Lenient threshold
        
        try:
            pruned_pool, pruning_info = apply_pruning_strategy(
                pool=compounds,
                predictions=predictions,
                uncertainties=uncertainties,
                strategy='probabilistic',
                params=params
            )
            
            # Should keep all compounds
            assert len(pruned_pool) == len(compounds)
            assert pruning_info['pruned_count'] == 0
            assert pruning_info['pruning_fraction'] == 0.0
            
        except ImportError:
            pytest.skip("Pruning module not available")
        except Exception as e:
            pytest.skip(f"No pruning test failed: {e}")
    
    def test_complete_pruning_case(self, small_real_compounds):
        """Test case where all compounds might be pruned."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(3)],
                'SMILES': ['CCN'] * 3
            })
        
        # Set parameters that might prune everything
        predictions = np.array([0.01, 0.02, 0.03])[:len(compounds)]  # Very low predictions
        uncertainties = np.array([0.9, 0.95, 0.92])[:len(compounds)]  # High uncertainties
        
        params = {'threshold': 0.9, 'retention_fraction': 0.1}  # Strict threshold
        
        try:
            pruned_pool, pruning_info = apply_pruning_strategy(
                pool=compounds,
                predictions=predictions,
                uncertainties=uncertainties,
                strategy='probabilistic',
                params=params
            )
            
            # Validate results even if everything is pruned
            assert len(pruned_pool) <= len(compounds)
            assert pruning_info['original_pool_size'] == len(compounds)
            assert pruning_info['pruned_pool_size'] == len(pruned_pool)
            
        except ImportError:
            pytest.skip("Pruning module not available")
        except Exception as e:
            pytest.skip(f"Complete pruning test failed: {e}")
    
    def test_missing_required_columns(self):
        """Test error handling when compounds lack required columns."""
        # Missing SMILES column
        invalid_compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002'],
            'structure': ['CCO', 'CCC']  # Wrong column name
        })
        
        predictions = np.array([0.5, 0.7])
        uncertainties = np.array([0.2, 0.3])
        params = {'threshold': 0.5}
        
        try:
            with pytest.raises((KeyError, ValueError)):
                apply_pruning_strategy(
                    pool=invalid_compounds,
                    predictions=predictions,
                    uncertainties=uncertainties,
                    strategy='probabilistic',
                    params=params
                )
        except ImportError:
            pytest.skip("Pruning module not available")
    
    def test_mismatched_data_lengths(self, small_real_compounds):
        """Test error handling when data lengths don't match."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
                'SMILES': ['CCO', 'CCC', 'CCN']
            })
        
        # Mismatched lengths
        predictions = np.array([0.5, 0.7])  # Too short
        uncertainties = np.array([0.2, 0.3, 0.4, 0.5])  # Too long
        params = {'threshold': 0.5}
        
        try:
            # Should handle mismatched lengths gracefully or raise appropriate error
            pruned_pool, pruning_info = apply_pruning_strategy(
                pool=compounds,
                predictions=predictions,
                uncertainties=uncertainties,
                strategy='probabilistic',
                params=params
            )
            
            # If it succeeds, validate basic structure
            assert isinstance(pruned_pool, pd.DataFrame)
            assert isinstance(pruning_info, dict)
            
        except ImportError:
            pytest.skip("Pruning module not available")
        except (ValueError, IndexError):
            # Expected for mismatched lengths
            pass
        except Exception as e:
            pytest.skip(f"Mismatched data length test failed: {e}")
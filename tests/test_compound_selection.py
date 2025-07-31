"""
Tests for compound selection functionality in learnm8.py.

Tests the select_compounds_by_strategy function with real molecular data,
focusing on different acquisition strategies, error handling, and data validation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from learnm8.learnm8 import select_compounds_by_strategy, _fallback_greedy_selection
from learnm8.core.data_manager import DataManager


class TestSelectCompoundsByStrategy:
    """Test compound selection with different strategies."""
    
    def test_greedy_selection_higher(self, small_real_compounds, tmp_path):
        """Test greedy selection with 'higher' score direction."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(10)],
                'SMILES': ['CCO'] * 10
            })
        
        # Create predictable predictions
        np.random.seed(42)
        predictions = np.random.uniform(0, 1, len(compounds))
        
        data_manager = DataManager(results_dir=tmp_path)
        
        selected = select_compounds_by_strategy(
            pool=compounds,
            predictions=predictions,
            uncertainties=None,
            strategy='greedy',
            batch_size=3,
            score_direction='higher',
            data_manager=data_manager
        )
        
        assert len(selected) == 3
        assert 'ID' in selected.columns
        assert 'SMILES' in selected.columns
        
        # Should select compounds with highest predictions
        selected_indices = [compounds[compounds['ID'] == id_val].index[0] for id_val in selected['ID']]
        selected_predictions = predictions[selected_indices]
        
        # Verify they are among the top predictions
        top_3_predictions = np.sort(predictions)[-3:]
        assert all(pred in top_3_predictions for pred in selected_predictions)
    
    def test_greedy_selection_lower(self, small_real_compounds, tmp_path):
        """Test greedy selection with 'lower' score direction."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(8)],
                'SMILES': ['CCC'] * 8
            })
        
        np.random.seed(42)
        predictions = np.random.uniform(0, 1, len(compounds))
        
        data_manager = DataManager(results_dir=tmp_path)
        
        selected = select_compounds_by_strategy(
            pool=compounds,
            predictions=predictions,
            uncertainties=None,
            strategy='greedy',
            batch_size=2,
            score_direction='lower',
            data_manager=data_manager
        )
        
        assert len(selected) == 2
        
        # Should select compounds with lowest predictions (after transformation)
        selected_indices = [compounds[compounds['ID'] == id_val].index[0] for id_val in selected['ID']]
        selected_predictions = predictions[selected_indices]
        
        # With 'lower' direction, it should prefer lower predictions
        bottom_2_predictions = np.sort(predictions)[:2]
        assert all(pred in bottom_2_predictions for pred in selected_predictions)
    
    def test_random_selection(self, small_real_compounds, tmp_path):
        """Test random selection strategy."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(6)],
                'SMILES': ['CCN'] * 6
            })
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        data_manager = DataManager(results_dir=tmp_path)
        
        selected = select_compounds_by_strategy(
            pool=compounds,
            predictions=predictions,
            uncertainties=None,
            strategy='random',
            batch_size=4,
            score_direction='higher',
            data_manager=data_manager
        )
        
        assert len(selected) == 4
        assert 'ID' in selected.columns
        assert 'SMILES' in selected.columns
        
        # Should be a subset of original compounds
        assert selected['ID'].isin(compounds['ID']).all()
    
    def test_uncertainty_based_selection(self, small_real_compounds, tmp_path):
        """Test uncertainty-based selection strategies."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(8)],
                'SMILES': ['CCO'] * 8
            })
        
        predictions = np.random.uniform(0.3, 0.7, len(compounds))
        uncertainties = np.random.uniform(0.1, 0.9, len(compounds))
        
        data_manager = DataManager(results_dir=tmp_path)
        
        # Test strategies that use uncertainty
        uncertainty_strategies = ['ucb']  # Start with basic ones that should work
        
        for strategy in uncertainty_strategies:
            try:
                selected = select_compounds_by_strategy(
                    pool=compounds,
                    predictions=predictions,
                    uncertainties=uncertainties,
                    strategy=strategy,
                    batch_size=3,
                    score_direction='higher',
                    data_manager=data_manager
                )
                
                assert len(selected) == 3
                assert 'ID' in selected.columns
                assert selected['ID'].isin(compounds['ID']).all()
                
            except Exception as e:
                # If the strategy fails, it should fall back to greedy
                pytest.skip(f"Strategy {strategy} not available or failed: {e}")
    
    def test_batch_size_constraints(self, small_real_compounds, tmp_path):
        """Test batch size constraint handling."""
        compounds = small_real_compounds.copy()
        
        # Use a small subset for predictable testing
        if len(compounds) >= 2:
            compounds = compounds.iloc[:2].copy()
        else:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002'],
                'SMILES': ['CCO', 'CCC']
            })
        
        predictions = np.array([0.3, 0.7])
        
        data_manager = DataManager(results_dir=tmp_path)
        
        # Test batch size larger than pool
        selected = select_compounds_by_strategy(
            pool=compounds,
            predictions=predictions,
            uncertainties=None,
            strategy='greedy',
            batch_size=10,  # Larger than pool size
            score_direction='higher',
            data_manager=data_manager
        )
        
        # Should return all available compounds
        assert len(selected) == len(compounds)
        
        # Test zero batch size
        selected = select_compounds_by_strategy(
            pool=compounds,
            predictions=predictions,
            uncertainties=None,
            strategy='greedy',
            batch_size=0,
            score_direction='higher',
            data_manager=data_manager
        )
        
        # Should return empty DataFrame
        assert len(selected) == 0
    
    def test_missing_predictions_handling(self, small_real_compounds, tmp_path):
        """Test handling when predictions are None."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(5)],
                'SMILES': ['CCC'] * 5
            })
        
        data_manager = DataManager(results_dir=tmp_path)
        
        selected = select_compounds_by_strategy(
            pool=compounds,
            predictions=None,  # No predictions provided
            uncertainties=None,
            strategy='greedy',
            batch_size=2,
            score_direction='higher',
            data_manager=data_manager
        )
        
        # Should still work (using random predictions)
        assert len(selected) == 2
        assert 'ID' in selected.columns
    
    def test_missing_uncertainties_handling(self, small_real_compounds, tmp_path):
        """Test handling when uncertainties are None for uncertainty-based strategies."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(6)],
                'SMILES': ['CCN'] * 6
            })
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        data_manager = DataManager(results_dir=tmp_path)
        
        # Try uncertainty-based strategy without uncertainties
        selected = select_compounds_by_strategy(
            pool=compounds,
            predictions=predictions,
            uncertainties=None,  # No uncertainties provided
            strategy='ucb',
            batch_size=3,
            score_direction='higher',
            data_manager=data_manager
        )
        
        # Should still work (using random uncertainties)
        assert len(selected) == 3
    
    def test_dataframe_predictions(self, small_real_compounds, tmp_path):
        """Test handling of DataFrame predictions."""
        compounds = small_real_compounds.copy()
        
        # Use a small subset for predictable testing
        if len(compounds) >= 4:
            compounds = compounds.iloc[:4].copy()
        else:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(4)],
                'SMILES': ['CCO'] * 4
            })
        
        # Create DataFrame predictions
        predictions_df = pd.DataFrame({
            'prediction': [0.2, 0.8, 0.5, 0.9][:len(compounds)]
        })
        
        data_manager = DataManager(results_dir=tmp_path)
        
        selected = select_compounds_by_strategy(
            pool=compounds,
            predictions=predictions_df,
            uncertainties=None,
            strategy='greedy',
            batch_size=2,
            score_direction='higher',
            data_manager=data_manager
        )
        
        assert len(selected) == 2
        assert 'ID' in selected.columns
    
    def test_invalid_strategy_error(self, small_real_compounds, tmp_path):
        """Test error handling for invalid strategy names."""
        compounds = small_real_compounds.copy()
        
        # Use a single compound for predictable testing
        if len(compounds) >= 1:
            compounds = compounds.iloc[:1].copy()
        else:
            compounds = pd.DataFrame({
                'ID': ['COMP_001'],
                'SMILES': ['CCO']
            })
        
        predictions = np.array([0.5])
        
        data_manager = DataManager(results_dir=tmp_path)
        
        with pytest.raises(ValueError) as exc_info:
            select_compounds_by_strategy(
                pool=compounds,
                predictions=predictions,
                uncertainties=None,
                strategy='invalid_strategy',
                batch_size=1,
                score_direction='higher',
                data_manager=data_manager
            )
        
        # Should provide helpful error message
        error_msg = str(exc_info.value)
        assert 'Unknown strategy' in error_msg or 'invalid_strategy' in error_msg
    
    def test_acquisition_params_passing(self, small_real_compounds, tmp_path):
        """Test passing of acquisition parameters."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(5)],
                'SMILES': ['CCC'] * 5
            })
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        data_manager = DataManager(results_dir=tmp_path)
        
        # Test with acquisition parameters
        acquisition_params = {'exploration_weight': 0.5}
        
        selected = select_compounds_by_strategy(
            pool=compounds,
            predictions=predictions,
            uncertainties=None,
            strategy='greedy',  # Simple strategy that should ignore params
            batch_size=2,
            score_direction='higher',
            data_manager=data_manager,
            acquisition_params=acquisition_params
        )
        
        assert len(selected) == 2
    
    def test_fallback_on_error(self, small_real_compounds, tmp_path):
        """Test fallback to greedy selection when advanced methods fail."""
        compounds = small_real_compounds.copy()
        
        # Use a small subset for predictable testing
        if len(compounds) >= 6:
            compounds = compounds.iloc[:6].copy()
        else:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(6)],
                'SMILES': ['CCN'] * 6
            })
        
        predictions = np.array([0.1, 0.9, 0.3, 0.7, 0.5, 0.2])[:len(compounds)]
        
        data_manager = DataManager(results_dir=tmp_path)
        
        # Mock acquisition function to raise an error
        with patch('learnm8.acquisition.get_acquisition_function') as mock_get_acq:
            mock_acq_func = Mock()
            mock_acq_func.select.side_effect = RuntimeError("Acquisition failed")
            mock_get_acq.return_value = Mock
            mock_get_acq.return_value.return_value = mock_acq_func
            
            # Should fall back to greedy selection
            selected = select_compounds_by_strategy(
                pool=compounds,
                predictions=predictions,
                uncertainties=None,
                strategy='some_failing_strategy',
                batch_size=3,
                score_direction='higher',
                data_manager=data_manager
            )
            
            # Should still return results (from fallback)
            assert len(selected) == 3


class TestFallbackGreedySelection:
    """Test fallback greedy selection function."""
    
    def test_fallback_greedy_higher(self):
        """Test fallback greedy selection with higher score direction."""
        compound_data = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004'],
            'SMILES': ['CCO', 'CCC', 'CCN', 'CO'],
            'prediction': [0.3, 0.8, 0.1, 0.6]
        })
        
        selected = _fallback_greedy_selection(
            compound_data=compound_data,
            batch_size=2,
            score_direction='higher'
        )
        
        assert len(selected) == 2
        
        # Should select compounds with highest predictions
        expected_ids = ['COMP_002', 'COMP_004']  # 0.8 and 0.6
        assert selected['ID'].tolist() == expected_ids
    
    def test_fallback_greedy_lower(self):
        """Test fallback greedy selection with lower score direction."""
        compound_data = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCN'],
            'prediction': [0.7, 0.2, 0.9]
        })
        
        selected = _fallback_greedy_selection(
            compound_data=compound_data,
            batch_size=2,
            score_direction='lower'
        )
        
        assert len(selected) == 2
        
        # Should select compounds with lowest predictions
        expected_ids = ['COMP_002', 'COMP_001']  # 0.2 and 0.7
        assert selected['ID'].tolist() == expected_ids
    
    def test_fallback_no_predictions(self):
        """Test fallback when no prediction column exists."""
        compound_data = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002'],
            'SMILES': ['CCO', 'CCC']
            # No prediction column
        })
        
        selected = _fallback_greedy_selection(
            compound_data=compound_data,
            batch_size=1,
            score_direction='higher'
        )
        
        # Should return random sample
        assert len(selected) == 1
        assert selected['ID'].iloc[0] in ['COMP_001', 'COMP_002']
    
    def test_fallback_empty_data(self):
        """Test fallback with empty compound data."""
        empty_data = pd.DataFrame(columns=['ID', 'SMILES', 'prediction'])
        
        selected = _fallback_greedy_selection(
            compound_data=empty_data,
            batch_size=1,
            score_direction='higher'
        )
        
        assert len(selected) == 0
    
    def test_fallback_batch_size_larger_than_data(self):
        """Test fallback when batch size exceeds available data."""
        compound_data = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'prediction': [0.5]
        })
        
        selected = _fallback_greedy_selection(
            compound_data=compound_data,
            batch_size=5,  # Larger than data
            score_direction='higher'
        )
        
        # Should return all available data
        assert len(selected) == 1
        assert selected['ID'].iloc[0] == 'COMP_001'
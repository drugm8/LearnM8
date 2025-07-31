"""
Tests for CSV export functionality in learnm8.py.

Tests the CSV export functions with real molecular data,
focusing on data integrity, file creation, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from learnm8.learnm8 import (
    _export_comprehensive_csv,
    _export_predictions_by_cycle_csv,
    _export_selection_history_csv,
    _export_best_compounds_csv
)


class TestCSVExportFunctions:
    """Test CSV export functionality."""
    
    def test_export_comprehensive_csv(self, small_real_compounds, tmp_path):
        """Test comprehensive CSV export functionality."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(8)],
                'SMILES': ['CCO'] * 8
            })
        
        # Create mock data for export
        labeled_data = compounds.iloc[:3].copy()
        labeled_data['Activity'] = [0.2, 0.7, 0.9]
        
        unlabeled_pool = compounds.iloc[3:].copy()
        
        all_metrics = [
            {
                'cycle': 0,
                'strategy': 'random',
                'selected_count': 2,
                'prediction_mean': 0.5,
                'uncertainty_mean': 0.2
            },
            {
                'cycle': 1,
                'strategy': 'greedy',
                'selected_count': 1,
                'prediction_mean': 0.7,
                'uncertainty_mean': 0.3
            }
        ]
        
        # Create prediction history
        prediction_history = {
            0: pd.DataFrame({
                'ID': unlabeled_pool['ID'].iloc[:3],
                'SMILES': unlabeled_pool['SMILES'].iloc[:3],
                'prediction_cycle_0': [0.4, 0.6, 0.8]
            })
        }
        
        # Create uncertainty history
        uncertainty_history = {
            0: pd.DataFrame({
                'ID': unlabeled_pool['ID'].iloc[:3],
                'SMILES': unlabeled_pool['SMILES'].iloc[:3],
                'uncertainty_cycle_0': [0.1, 0.2, 0.3]
            })
        }
        
        # Create selection history
        selection_history = [
            {
                'ID': unlabeled_pool['ID'].iloc[0],
                'SMILES': unlabeled_pool['SMILES'].iloc[0],
                'selected_cycle': 0,
                'strategy': 'random',
                'prediction_at_selection': 0.4,
                'uncertainty_at_selection': 0.1,
                'oracle_measured_value': 0.3
            }
        ]
        
        # Mock the evaluation module export function
        with patch('learnm8.evaluation.core.export_metrics_csv') as mock_export_metrics:
            csv_files = _export_comprehensive_csv(
                output_dir=str(tmp_path),
                labeled_data=labeled_data,
                unlabeled_pool=unlabeled_pool,
                all_metrics=all_metrics,
                prediction_history=prediction_history,
                uncertainty_history=uncertainty_history,
                selection_history=selection_history,
                compound_pool=compounds,
                target_column='Activity',
                oracle_type='run',
                score_direction='higher'
            )
            
            # Should return dictionary of created files
            assert isinstance(csv_files, dict)
            
            # Should have attempted to export metrics
            mock_export_metrics.assert_called_once()
    
    def test_export_predictions_by_cycle_csv(self, tmp_path):
        """Test export of predictions by cycle."""
        compounds = pd.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(6)],
            'SMILES': ['CCO'] * 6
        })
        
        labeled_data = compounds.iloc[:2].copy()
        labeled_data['Activity'] = [0.3, 0.8]
        
        # Create prediction history for multiple cycles
        prediction_history = {
            0: pd.DataFrame({
                'ID': compounds['ID'].iloc[2:5],
                'SMILES': compounds['SMILES'].iloc[2:5],
                'prediction_cycle_0': [0.4, 0.6, 0.7]
            }),
            1: pd.DataFrame({
                'ID': compounds['ID'].iloc[3:6],
                'SMILES': compounds['SMILES'].iloc[3:6],
                'prediction_cycle_1': [0.5, 0.8, 0.2]
            })
        }
        
        uncertainty_history = {
            0: pd.DataFrame({
                'ID': compounds['ID'].iloc[2:5],
                'SMILES': compounds['SMILES'].iloc[2:5],
                'uncertainty_cycle_0': [0.1, 0.2, 0.15]
            })
        }
        
        output_path = tmp_path / 'predictions_by_cycle.csv'
        
        _export_predictions_by_cycle_csv(
            prediction_history=prediction_history,
            uncertainty_history=uncertainty_history,
            compound_pool=compounds,
            labeled_data=labeled_data,
            target_column='Activity',
            output_path=output_path
        )
        
        # Check file was created
        assert output_path.exists()
        
        # Read and validate content
        result_df = pd.read_csv(output_path)
        
        # Should have all original compounds
        assert len(result_df) == len(compounds)
        assert 'ID' in result_df.columns
        assert 'SMILES' in result_df.columns
        
        # Should have prediction columns for each cycle
        assert 'prediction_cycle_0' in result_df.columns
        assert 'prediction_cycle_1' in result_df.columns
        
        # Should have uncertainty columns
        assert 'uncertainty_cycle_0' in result_df.columns
        
        # Should have oracle values for labeled compounds
        assert 'final_oracle_value' in result_df.columns
    
    def test_export_selection_history_csv(self, tmp_path):
        """Test export of selection history."""
        selection_history = [
            {
                'ID': 'COMP_001',
                'SMILES': 'CCO',
                'selected_cycle': 0,
                'strategy': 'random',
                'prediction_at_selection': 0.4,
                'uncertainty_at_selection': 0.1,
                'oracle_measured_value': 0.3
            },
            {
                'ID': 'COMP_002',
                'SMILES': 'CCC',
                'selected_cycle': 0,
                'strategy': 'random',
                'prediction_at_selection': 0.6,
                'uncertainty_at_selection': 0.2,
                'oracle_measured_value': 0.7
            },
            {
                'ID': 'COMP_003',
                'SMILES': 'CCN',
                'selected_cycle': 1,
                'strategy': 'greedy',
                'prediction_at_selection': 0.8,
                'uncertainty_at_selection': 0.15,
                'oracle_measured_value': 0.9
            }
        ]
        
        output_path = tmp_path / 'selection_history.csv'
        
        _export_selection_history_csv(
            selection_history=selection_history,
            output_path=output_path
        )
        
        # Check file was created
        assert output_path.exists()
        
        # Read and validate content
        result_df = pd.read_csv(output_path)
        
        assert len(result_df) == 3
        assert 'ID' in result_df.columns
        assert 'SMILES' in result_df.columns
        assert 'selected_cycle' in result_df.columns
        assert 'strategy' in result_df.columns
        assert 'prediction_at_selection' in result_df.columns
        assert 'oracle_measured_value' in result_df.columns
        
        # Validate data values
        assert result_df['ID'].tolist() == ['COMP_001', 'COMP_002', 'COMP_003']
        assert result_df['selected_cycle'].tolist() == [0, 0, 1]
        assert result_df['strategy'].tolist() == ['random', 'random', 'greedy']
    
    def test_export_best_compounds_csv_higher(self, tmp_path):
        """Test export of best compounds with 'higher' score direction."""
        labeled_data = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004'],
            'SMILES': ['CCO', 'CCC', 'CCN', 'CO'],
            'Activity': [0.3, 0.9, 0.1, 0.7]
        })
        
        output_path = tmp_path / 'best_compounds.csv'
        
        _export_best_compounds_csv(
            labeled_data=labeled_data,
            target_column='Activity',
            score_direction='higher',
            output_path=output_path
        )
        
        # Check file was created
        assert output_path.exists()
        
        # Read and validate content
        result_df = pd.read_csv(output_path)
        
        assert len(result_df) == 4
        assert 'rank' in result_df.columns
        assert 'ID' in result_df.columns
        assert 'Activity' in result_df.columns
        
        # Should be sorted by Activity (descending for 'higher')
        expected_order = ['COMP_002', 'COMP_004', 'COMP_001', 'COMP_003']  # 0.9, 0.7, 0.3, 0.1
        assert result_df['ID'].tolist() == expected_order
        assert result_df['rank'].tolist() == [1, 2, 3, 4]
    
    def test_export_best_compounds_csv_lower(self, tmp_path):
        """Test export of best compounds with 'lower' score direction."""
        labeled_data = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCN'],
            'Activity': [0.8, 0.2, 0.5]
        })
        
        output_path = tmp_path / 'best_compounds_lower.csv'
        
        _export_best_compounds_csv(
            labeled_data=labeled_data,
            target_column='Activity',
            score_direction='lower',
            output_path=output_path
        )
        
        # Check file was created
        assert output_path.exists()
        
        # Read and validate content
        result_df = pd.read_csv(output_path)
        
        # Should be sorted by Activity (ascending for 'lower')
        expected_order = ['COMP_002', 'COMP_003', 'COMP_001']  # 0.2, 0.5, 0.8
        assert result_df['ID'].tolist() == expected_order
        assert result_df['rank'].tolist() == [1, 2, 3]
    
    def test_empty_selection_history_export(self, tmp_path):
        """Test export with empty selection history."""
        output_path = tmp_path / 'empty_selection.csv'
        
        _export_selection_history_csv(
            selection_history=[],
            output_path=output_path
        )
        
        # Should not create file or create empty file
        # Either behavior is acceptable
        if output_path.exists():
            result_df = pd.read_csv(output_path)
            assert len(result_df) == 0
    
    def test_missing_target_column_error(self, tmp_path):
        """Test error handling when target column is missing."""
        labeled_data = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002'],
            'SMILES': ['CCO', 'CCC']
            # Missing 'Activity' column
        })
        
        output_path = tmp_path / 'invalid_best.csv'
        
        # Should handle missing target column gracefully
        _export_best_compounds_csv(
            labeled_data=labeled_data,
            target_column='Activity',
            score_direction='higher',
            output_path=output_path
        )
        
        # Should not create file or handle error gracefully
        # Function should not crash
    
    def test_predictions_export_with_missing_data(self, tmp_path):
        """Test predictions export with missing prediction/uncertainty data."""
        compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCN']
        })
        
        labeled_data = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })
        
        # Empty histories
        prediction_history = {}
        uncertainty_history = {}
        
        output_path = tmp_path / 'predictions_empty.csv'
        
        _export_predictions_by_cycle_csv(
            prediction_history=prediction_history,
            uncertainty_history=uncertainty_history,
            compound_pool=compounds,
            labeled_data=labeled_data,
            target_column='Activity',
            output_path=output_path
        )
        
        # Should still create basic file with compound info
        assert output_path.exists()
        
        result_df = pd.read_csv(output_path)
        assert len(result_df) == len(compounds)
        assert 'ID' in result_df.columns
        assert 'SMILES' in result_df.columns
    
    def test_comprehensive_export_error_handling(self, tmp_path):
        """Test error handling in comprehensive export."""
        compounds = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO']
        })
        
        labeled_data = compounds.copy()
        unlabeled_pool = pd.DataFrame(columns=['ID', 'SMILES'])
        
        # Create invalid metrics (missing required fields)
        all_metrics = [{'invalid': 'data'}]
        
        # Mock evaluation export to raise an error
        with patch('learnm8.evaluation.core.export_metrics_csv', side_effect=Exception("Export failed")):
            csv_files = _export_comprehensive_csv(
                output_dir=str(tmp_path),
                labeled_data=labeled_data,
                unlabeled_pool=unlabeled_pool,
                all_metrics=all_metrics,
                prediction_history={},
                uncertainty_history={},
                selection_history=[],
                compound_pool=compounds,
                target_column='Activity',
                oracle_type='run',
                score_direction='higher'
            )
            
            # Should return dictionary even if some exports fail
            assert isinstance(csv_files, dict)
    
    def test_file_path_creation(self, tmp_path):
        """Test that output directories are created properly."""
        # Create nested directory path
        nested_path = tmp_path / 'nested' / 'directory'
        
        compounds = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })
        
        # Should create directories as needed
        with patch('learnm8.evaluation.core.export_metrics_csv'):
            csv_files = _export_comprehensive_csv(
                output_dir=str(nested_path),
                labeled_data=compounds,
                unlabeled_pool=pd.DataFrame(columns=['ID', 'SMILES']),
                all_metrics=[],
                prediction_history={},
                uncertainty_history={},
                selection_history=[],
                compound_pool=compounds,
                target_column='Activity',
                oracle_type='run',
                score_direction='higher'
            )
            
            # Directory should be created
            assert nested_path.exists()
            assert nested_path.is_dir()
    
    def test_large_data_export(self, tmp_path):
        """Test export with larger datasets."""
        # Create larger dataset
        n_compounds = 100
        compounds = pd.DataFrame({
            'ID': [f'COMP_{i:04d}' for i in range(n_compounds)],
            'SMILES': ['CCO'] * n_compounds
        })
        
        labeled_data = compounds.iloc[:20].copy()
        labeled_data['Activity'] = np.random.uniform(0, 1, 20)
        
        # Create larger selection history
        selection_history = []
        for i in range(15):
            selection_history.append({
                'ID': labeled_data['ID'].iloc[i % len(labeled_data)],
                'SMILES': labeled_data['SMILES'].iloc[i % len(labeled_data)],
                'selected_cycle': i // 5,
                'strategy': 'random',
                'prediction_at_selection': np.random.uniform(0, 1),
                'uncertainty_at_selection': np.random.uniform(0.1, 0.3),
                'oracle_measured_value': np.random.uniform(0, 1)
            })
        
        output_path = tmp_path / 'large_selection_history.csv'
        
        _export_selection_history_csv(
            selection_history=selection_history,
            output_path=output_path
        )
        
        # Should handle large data without issues
        assert output_path.exists()
        
        result_df = pd.read_csv(output_path)
        assert len(result_df) == 15
        assert 'ID' in result_df.columns
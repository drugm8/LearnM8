"""
Tests for cycle execution functionality in learnm8.py.

Tests execute_single_cycle, execute_run_mode_cycle, and execute_benchmark_mode_cycle
functions with real molecular data, focusing on integration and data flow.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from learnm8.learnm8 import (
    execute_single_cycle, 
    execute_run_mode_cycle, 
    execute_benchmark_mode_cycle
)
from learnm8.core.interfaces import Learner, Oracle
from learnm8.core.data_manager import DataManager


class MockLearner(Learner):
    """Mock learner for cycle testing."""
    
    def __init__(self, supports_uncertainty=False, prediction_values=None):
        self.supports_uncertainty_flag = supports_uncertainty
        self.is_trained = False
        self.training_data = None
        self.prediction_values = prediction_values
    
    def train(self, compounds: pd.DataFrame, target_column: str, data_manager: DataManager):
        """Mock training implementation."""
        if len(compounds) == 0:
            raise ValueError("Cannot train on empty dataset")
        if target_column not in compounds.columns:
            raise KeyError(f"Target column '{target_column}' not found")
        
        self.is_trained = True
        self.training_data = compounds.copy()
    
    def predict(self, compounds: pd.DataFrame, data_manager: DataManager):
        """Mock prediction implementation."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        if self.prediction_values is not None and len(self.prediction_values) >= len(compounds):
            predictions = self.prediction_values[:len(compounds)]
        else:
            # Generate consistent predictions based on compound IDs
            np.random.seed(hash(''.join(compounds['ID'].astype(str))) % 2**32)
            predictions = np.random.uniform(0.1, 0.9, len(compounds))
        
        uncertainty = None
        if self.supports_uncertainty_flag:
            uncertainty = np.random.uniform(0.05, 0.3, len(compounds))
        
        return predictions, uncertainty
    
    def supports_uncertainty(self) -> bool:
        return self.supports_uncertainty_flag
    
    def get_name(self) -> str:
        return f"MockLearner(uncertainty={self.supports_uncertainty_flag})"


class MockOracle(Oracle):
    """Mock oracle for cycle testing."""
    
    def __init__(self, measurement_values=None, ground_truth=None):
        self.measurement_count = 0
        self.measurement_values = measurement_values
        self.ground_truth = ground_truth
    
    def measure(self, compounds: pd.DataFrame, properties: list) -> pd.DataFrame:
        """Mock measurement implementation."""
        if len(compounds) == 0:
            return pd.DataFrame(columns=['ID', 'SMILES'] + properties)
        
        result = compounds[['ID', 'SMILES']].copy()
        
        for prop in properties:
            if self.ground_truth is not None and prop in self.ground_truth.columns:
                # Use ground truth if available
                truth_mapping = dict(zip(self.ground_truth['ID'], self.ground_truth[prop]))
                result[prop] = result['ID'].map(lambda x: truth_mapping.get(x, np.random.uniform(0, 1)))
            elif self.measurement_values is not None and len(self.measurement_values) >= len(compounds):
                result[prop] = self.measurement_values[:len(compounds)]
            else:
                # Generate consistent values
                np.random.seed(hash(''.join(compounds['ID'].astype(str))) % 2**32)
                result[prop] = np.random.uniform(0.1, 0.9, len(compounds))
        
        self.measurement_count += len(compounds)
        return result


class TestCycleExecution:
    """Test cycle execution functions."""
    
    def test_execute_single_cycle_basic(self, small_real_compounds, tmp_path):
        """Test basic single cycle execution."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(10)],
                'SMILES': ['CCO'] * 10
            })
        
        # Create labeled and unlabeled data
        labeled_data = compounds.iloc[:3].copy()
        labeled_data['Activity'] = [0.1, 0.5, 0.9]
        unlabeled_pool = compounds.iloc[3:].copy()
        
        oracle = MockOracle(ground_truth=labeled_data)
        learner = MockLearner(supports_uncertainty=True)
        data_manager = DataManager(results_dir=tmp_path)
        
        # Execute single cycle
        result = execute_single_cycle(
            labeled_data=labeled_data,
            unlabeled_pool=unlabeled_pool,
            original_compound_pool=compounds,
            strategy='greedy',
            batch_fraction=0.2,
            cycle=0,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            data_manager=data_manager,
            original_pool_size=len(compounds),
            score_direction='higher'
        )
        
        # Unpack results
        new_labeled, new_unlabeled, metrics = result[:3]
        
        # Validate results
        assert len(new_labeled) > len(labeled_data)  # Should have added compounds
        assert len(new_unlabeled) < len(unlabeled_pool)  # Should have removed compounds
        assert len(new_labeled) + len(new_unlabeled) == len(compounds)
        
        # Validate metrics
        assert metrics['cycle'] == 0
        assert metrics['strategy'] == 'greedy'
        assert metrics['selected_count'] > 0
        assert 'Activity' in new_labeled.columns
    
    def test_execute_run_mode_cycle(self, small_real_compounds, tmp_path):
        """Test run mode cycle execution."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(8)],
                'SMILES': ['CCO'] * 8
            })
        
        labeled_data = compounds.iloc[:2].copy()
        labeled_data['Activity'] = [0.2, 0.8]
        unlabeled_pool = compounds.iloc[2:].copy()
        
        oracle = MockOracle(ground_truth=labeled_data)
        learner = MockLearner()
        data_manager = DataManager(results_dir=tmp_path)
        
        result = execute_run_mode_cycle(
            labeled_data=labeled_data,
            unlabeled_pool=unlabeled_pool,
            strategy='random',
            batch_fraction=0.04,  # 2 compounds for 50-compound pool: 50 * 0.04 = 2
            cycle=1,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            data_manager=data_manager,
            original_pool_size=len(compounds),
            score_direction='higher',
            enable_evaluation=False  # Disable for simpler testing
        )
        
        new_labeled, new_unlabeled, metrics = result[:3]
        
        # Calculate expected batch size based on original pool size
        expected_batch_size = max(1, int(len(compounds) * 0.04))
        assert len(new_labeled) == len(labeled_data) + expected_batch_size
        assert len(new_unlabeled) == len(unlabeled_pool) - expected_batch_size
        assert metrics['strategy'] == 'random'
        assert metrics['selected_count'] == expected_batch_size
    
    def test_execute_benchmark_mode_cycle(self, small_real_compounds, tmp_path):
        """Test benchmark mode cycle execution."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(6)],
                'SMILES': ['CCC'] * 6
            })
        
        # Add ground truth for all compounds
        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))
        
        labeled_data = compounds.iloc[:2].copy()
        unlabeled_pool = compounds.iloc[2:].copy()
        
        oracle = MockOracle(ground_truth=compounds)
        learner = MockLearner(supports_uncertainty=True)
        data_manager = DataManager(results_dir=tmp_path)
        
        result = execute_benchmark_mode_cycle(
            labeled_data=labeled_data,
            unlabeled_pool=unlabeled_pool,
            original_compound_pool=compounds,
            strategy='greedy',
            batch_fraction=0.04,  # 2 compounds for 50-compound pool: 50 * 0.04 = 2
            cycle=0,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            data_manager=data_manager,
            original_pool_size=len(compounds),
            score_direction='higher',
            ground_truth_data=compounds,
            enable_evaluation=False
        )
        
        new_labeled, new_unlabeled, metrics = result[:3]
        
        # Calculate expected batch size based on original pool size
        expected_batch_size = max(1, int(len(compounds) * 0.04))
        assert len(new_labeled) == len(labeled_data) + expected_batch_size
        assert len(new_unlabeled) == len(unlabeled_pool) - expected_batch_size
        assert metrics['strategy'] == 'greedy'
    
    def test_empty_unlabeled_pool_handling(self, tmp_path):
        """Test handling of empty unlabeled pool."""
        labeled_data = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })
        
        empty_unlabeled = pd.DataFrame(columns=['ID', 'SMILES'])
        
        oracle = MockOracle()
        learner = MockLearner()
        data_manager = DataManager(results_dir=tmp_path)
        
        result = execute_single_cycle(
            labeled_data=labeled_data,
            unlabeled_pool=empty_unlabeled,
            original_compound_pool=labeled_data,
            strategy='greedy',
            batch_fraction=0.5,
            cycle=0,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            data_manager=data_manager,
            original_pool_size=1,
            score_direction='higher'
        )
        
        new_labeled, new_unlabeled, metrics = result[:3]
        
        # Should return unchanged data with appropriate metrics
        assert len(new_labeled) == len(labeled_data)
        assert len(new_unlabeled) == 0
        assert metrics['selected_count'] == 0
        assert metrics['remaining_pool'] == 0
    
    def test_batch_size_calculation(self, tmp_path):
        """Test correct batch size calculation from batch_fraction."""
        compounds = pd.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(20)],
            'SMILES': ['CCO'] * 20
        })
        
        labeled_data = compounds.iloc[:5].copy()
        labeled_data['Activity'] = np.random.uniform(0, 1, 5)
        unlabeled_pool = compounds.iloc[5:].copy()
        
        oracle = MockOracle()
        learner = MockLearner()
        data_manager = DataManager(results_dir=tmp_path)
        
        # Test different batch fractions
        test_cases = [
            (0.1, 2),   # 10% of 20 = 2
            (0.25, 5),  # 25% of 20 = 5
            (0.05, 1),  # 5% of 20 = 1 (minimum 1)
        ]
        
        for batch_fraction, expected_selected in test_cases:
            result = execute_single_cycle(
                labeled_data=labeled_data,
                unlabeled_pool=unlabeled_pool,
                original_compound_pool=compounds,
                strategy='random',
                batch_fraction=batch_fraction,
                cycle=0,
                oracle=oracle,
                learner=learner,
                target_column='Activity',
                data_manager=data_manager,
                original_pool_size=len(compounds),
                score_direction='higher',
                enable_evaluation=False
            )
            
            new_labeled, new_unlabeled, metrics = result[:3]
            assert metrics['selected_count'] == expected_selected
    
    def test_score_direction_lower(self, tmp_path):
        """Test cycle execution with 'lower' score direction."""
        compounds = pd.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(8)],
            'SMILES': ['CCN'] * 8
        })
        
        labeled_data = compounds.iloc[:2].copy()
        labeled_data['Activity'] = [0.9, 0.1]  # Lower is better
        unlabeled_pool = compounds.iloc[2:].copy()
        
        oracle = MockOracle()
        learner = MockLearner()
        data_manager = DataManager(results_dir=tmp_path)
        
        result = execute_single_cycle(
            labeled_data=labeled_data,
            unlabeled_pool=unlabeled_pool,
            original_compound_pool=compounds,
            strategy='greedy',
            batch_fraction=0.25,
            cycle=0,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            data_manager=data_manager,
            original_pool_size=len(compounds),
            score_direction='lower',  # Lower is better
            enable_evaluation=False
        )
        
        new_labeled, new_unlabeled, metrics = result[:3]
        assert metrics['selected_count'] == 2
        assert len(new_labeled) == 4
    
    def test_csv_export_mode(self, tmp_path):
        """Test cycle execution with CSV export enabled."""
        compounds = pd.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(6)],
            'SMILES': ['CCO'] * 6
        })
        
        labeled_data = compounds.iloc[:2].copy()
        labeled_data['Activity'] = [0.3, 0.7]
        unlabeled_pool = compounds.iloc[2:].copy()
        
        oracle = MockOracle()
        learner = MockLearner(supports_uncertainty=True)
        data_manager = DataManager(results_dir=tmp_path)
        
        result = execute_single_cycle(
            labeled_data=labeled_data,
            unlabeled_pool=unlabeled_pool,
            original_compound_pool=compounds,
            strategy='random',
            batch_fraction=0.5,
            cycle=0,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            data_manager=data_manager,
            original_pool_size=len(compounds),
            score_direction='higher',
            export_csv=True,  # Enable CSV export
            enable_evaluation=False
        )
        
        # Should return additional data for CSV export
        assert len(result) == 6  # labeled, unlabeled, metrics, predictions, uncertainties, selections
        
        new_labeled, new_unlabeled, metrics, predictions, uncertainties, selections = result
        
        # Validate CSV export data
        assert predictions is not None
        assert uncertainties is not None  # Learner supports uncertainty
        assert isinstance(selections, list)
    
    def test_learner_training_failure(self, tmp_path):
        """Test handling of learner training failure."""
        compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002'],
            'SMILES': ['CCO', 'CCC']
        })
        
        # Empty labeled data should cause training failure
        empty_labeled = pd.DataFrame(columns=['ID', 'SMILES', 'Activity'])
        unlabeled_pool = compounds.copy()
        
        oracle = MockOracle()
        learner = MockLearner()
        data_manager = DataManager(results_dir=tmp_path)
        
        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            execute_single_cycle(
                labeled_data=empty_labeled,
                unlabeled_pool=unlabeled_pool,
                original_compound_pool=compounds,
                strategy='greedy',
                batch_fraction=0.5,
                cycle=0,
                oracle=oracle,
                learner=learner,
                target_column='Activity',
                data_manager=data_manager,
                original_pool_size=len(compounds),
                score_direction='higher'
            )
    
    def test_prediction_statistics_calculation(self, tmp_path):
        """Test calculation of prediction statistics in cycle metrics."""
        compounds = pd.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(10)],
            'SMILES': ['CCO'] * 10
        })
        
        labeled_data = compounds.iloc[:3].copy()
        labeled_data['Activity'] = [0.1, 0.5, 0.9]
        unlabeled_pool = compounds.iloc[3:].copy()
        
        # Create learner with predictable values
        prediction_values = np.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.7, 0.5])
        
        oracle = MockOracle()
        learner = MockLearner(prediction_values=prediction_values, supports_uncertainty=True)
        data_manager = DataManager(results_dir=tmp_path)
        
        result = execute_single_cycle(
            labeled_data=labeled_data,
            unlabeled_pool=unlabeled_pool,
            original_compound_pool=compounds,
            strategy='random',
            batch_fraction=0.2,
            cycle=0,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            data_manager=data_manager,
            original_pool_size=len(compounds),
            score_direction='higher',
            enable_evaluation=False
        )
        
        new_labeled, new_unlabeled, metrics = result[:3]
        
        # Should have prediction statistics
        assert 'prediction_mean' in metrics
        assert 'prediction_std' in metrics
        assert 'uncertainty_mean' in metrics  # Learner supports uncertainty
        assert 'uncertainty_std' in metrics
        
        # Values should be reasonable
        assert 0 <= metrics['prediction_mean'] <= 1
        assert metrics['prediction_std'] >= 0
    
    def test_multiple_strategies(self, tmp_path):
        """Test cycle execution with different acquisition strategies."""
        compounds = pd.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(12)],
            'SMILES': ['CCO'] * 12
        })
        
        labeled_data = compounds.iloc[:3].copy()
        labeled_data['Activity'] = [0.2, 0.5, 0.8]
        unlabeled_pool = compounds.iloc[3:].copy()
        
        oracle = MockOracle()
        learner = MockLearner(supports_uncertainty=True)
        data_manager = DataManager(results_dir=tmp_path)
        
        strategies_to_test = ['random', 'greedy']
        
        for strategy in strategies_to_test:
            result = execute_single_cycle(
                labeled_data=labeled_data,
                unlabeled_pool=unlabeled_pool,
                original_compound_pool=compounds,
                strategy=strategy,
                batch_fraction=0.25,
                cycle=0,
                oracle=oracle,
                learner=learner,
                target_column='Activity',
                data_manager=data_manager,
                original_pool_size=len(compounds),
                score_direction='higher',
                enable_evaluation=False
            )
            
            new_labeled, new_unlabeled, metrics = result[:3]
            
            assert metrics['strategy'] == strategy
            assert metrics['selected_count'] == 3  # 25% of 12
            assert len(new_labeled) == len(labeled_data) + 3
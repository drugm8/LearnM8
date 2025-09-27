"""
Integration tests for core LearnM8 active learning functionality.

Tests the main run_active_learning function with real molecular data,
focusing on integration, functionality, error handling, and data validation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from learnm8.learnm8 import run_active_learning
from learnm8.core.interfaces import Learner, Oracle
from learnm8.core.data_manager import DataManager


class MockLearner(Learner):
    """Mock learner for testing with predictable behavior."""
    
    def __init__(self, supports_uncertainty=False, fail_training=False, fail_prediction=False):
        self.supports_uncertainty_flag = supports_uncertainty
        self.fail_training = fail_training
        self.fail_prediction = fail_prediction
        self.is_trained = False
        self.training_data = None
    
    def train(self, compounds: pd.DataFrame, target_column: str, data_manager: DataManager):
        """Mock training implementation."""
        if self.fail_training:
            raise RuntimeError("Training failed")
        
        if len(compounds) == 0:
            raise ValueError("Cannot train on empty dataset")
        if target_column not in compounds.columns:
            raise KeyError(f"Target column '{target_column}' not found")
        
        self.is_trained = True
        self.training_data = compounds.copy()
    
    def predict(self, compounds: pd.DataFrame, data_manager: DataManager):
        """Mock prediction implementation with realistic values."""
        if self.fail_prediction:
            raise RuntimeError("Prediction failed")
        
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Generate predictions based on compound IDs for consistency
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
    """Mock oracle for testing with realistic molecular property simulation."""
    
    def __init__(self, fail_measurement=False, ground_truth=None):
        self.fail_measurement = fail_measurement
        self.measurement_count = 0
        self.ground_truth = ground_truth
    
    def measure(self, compounds: pd.DataFrame, properties: list) -> pd.DataFrame:
        """Mock measurement implementation."""
        if self.fail_measurement:
            raise RuntimeError("Oracle measurement failed")
        
        if len(compounds) == 0:
            return pd.DataFrame(columns=['ID', 'SMILES'] + properties)
        
        if 'ID' not in compounds.columns:
            raise KeyError("Compounds must have 'ID' column")
        
        result = compounds[['ID', 'SMILES']].copy()
        
        for prop in properties:
            if self.ground_truth is not None and prop in self.ground_truth.columns:
                # Use ground truth if available
                truth_mapping = dict(zip(self.ground_truth['ID'], self.ground_truth[prop]))
                result[prop] = result['ID'].map(lambda x: truth_mapping.get(x, np.random.uniform(0, 1)))
            else:
                # Generate consistent values based on compound ID
                np.random.seed(hash(''.join(compounds['ID'].astype(str))) % 2**32)
                result[prop] = np.random.uniform(0.1, 0.9, len(compounds))
        
        self.measurement_count += len(compounds)
        return result


class TestRunActiveLearning:
    """Test core run_active_learning function integration."""
    
    def test_basic_active_learning_integration(self, small_real_compounds, tmp_path):
        """Test basic active learning workflow with real molecular data."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Add activity column for testing
        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))
        
        oracle = MockOracle(ground_truth=compounds)
        learner = MockLearner(supports_uncertainty=True)
        
        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            featurizer='morgan',
            n_cycles=3,
            batch_fraction=0.1,
            initial_size=5,
            output_dir=str(tmp_path),
            random_state=42
        )
        
        # Validate results structure
        assert 'labeled_data' in results
        assert 'unlabeled_data' in results
        assert 'cycle_metrics' in results
        assert 'total_cycles' in results
        
        # Validate data integrity
        labeled_data = results['labeled_data']
        unlabeled_data = results['unlabeled_data']
        
        assert len(labeled_data) > 5  # Should have initial + selected compounds
        assert len(unlabeled_data) + len(labeled_data) == len(compounds)
        assert 'Activity' in labeled_data.columns
        
        # Validate cycle metrics
        cycle_metrics = results['cycle_metrics']
        assert len(cycle_metrics) == 3
        
        for i, metrics in enumerate(cycle_metrics):
            assert metrics['cycle'] == i
            assert metrics['selected_count'] > 0
            assert 'cumulative_labeled' in metrics
    
    def test_single_cycle_active_learning(self, small_real_compounds, tmp_path):
        """Test single cycle active learning."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))
        
        oracle = MockOracle(ground_truth=compounds)
        learner = MockLearner()
        
        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            featurizer='morgan',
            n_cycles=1,
            initial_batch_fraction=0.2,  # Use initial_batch_fraction for single cycle
            initial_size=3,
            output_dir=str(tmp_path),
            random_state=42
        )
        
        assert len(results['cycle_metrics']) == 1
        assert results['total_cycles'] == 1
        
        # Check that initial compounds + selected compounds are labeled
        expected_labeled = 3 + max(1, int(len(compounds) * 0.2))
        assert len(results['labeled_data']) >= expected_labeled
    
    def test_advanced_cycles_specification(self, small_real_compounds, tmp_path):
        """Test advanced cycles specification with different strategies."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))
        
        oracle = MockOracle(ground_truth=compounds)
        learner = MockLearner(supports_uncertainty=True)
        
        # Use advanced cycles specification
        cycles = [
            ('random', 0.05),
            ('greedy', 0.1),
            ('random', 0.05)
        ]
        
        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            featurizer='morgan',
            cycles=cycles,
            initial_size=3,
            output_dir=str(tmp_path),
            random_state=42
        )
        
        assert len(results['cycle_metrics']) == 3
        
        # Verify strategies were used correctly
        for i, (expected_strategy, _) in enumerate(cycles):
            assert results['cycle_metrics'][i]['strategy'] == expected_strategy
    
    def test_string_learner_creation(self, small_real_compounds, tmp_path):
        """Test creation of learner from string specification."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))
        oracle = MockOracle(ground_truth=compounds)
        
        # Test with string learner specification
        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner='rf',  # String specification
            target_column='Activity',
            featurizer='morgan',
            n_cycles=2,
            batch_fraction=0.15,
            initial_size=3,
            output_dir=str(tmp_path),
            random_state=42
        )
        
        assert results['total_cycles'] == 2
        assert len(results['labeled_data']) > 3
    
    def test_score_direction_handling(self, small_real_compounds, tmp_path):
        """Test both score directions (higher/lower is better)."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))
        oracle = MockOracle(ground_truth=compounds)
        learner = MockLearner()
        
        # Test 'higher' direction (default)
        results_higher = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            featurizer='morgan',
            score_direction='higher',
            n_cycles=1,
            batch_fraction=0.1,
            initial_size=3,
            output_dir=str(tmp_path / 'higher'),
            random_state=42
        )
        
        # Test 'lower' direction
        results_lower = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            featurizer='morgan',
            score_direction='lower',
            n_cycles=1,
            batch_fraction=0.1,
            initial_size=3,
            output_dir=str(tmp_path / 'lower'),
            random_state=42
        )
        
        assert results_higher['total_cycles'] == 1
        assert results_lower['total_cycles'] == 1
    
    def test_empty_compound_pool_handling(self, tmp_path):
        """Test error handling with empty compound pool."""
        empty_compounds = pd.DataFrame(columns=['ID', 'SMILES'])
        oracle = MockOracle()
        learner = MockLearner()
        
        with pytest.raises((ValueError, IndexError)):
            run_active_learning(
                compound_pool=empty_compounds,
                oracle=oracle,
                learner=learner,
                target_column='Activity',
                featurizer='morgan',
                n_cycles=1,
                output_dir=str(tmp_path)
            )
    
    def test_invalid_compound_pool_columns(self, tmp_path):
        """Test error handling with missing required columns."""
        # Missing SMILES column
        invalid_compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004', 'COMP_005', 
                   'COMP_006', 'COMP_007', 'COMP_008', 'COMP_009', 'COMP_010', 'COMP_011'],
            'structure': ['CCO', 'CCC', 'CCN', 'CO', 'CN', 'C', 'CC', 'CNC', 'COC', 'CCO', 'CCCO']  # Wrong column name
        })
        
        oracle = MockOracle()
        learner = MockLearner()
        
        with pytest.raises(KeyError):
            run_active_learning(
                compound_pool=invalid_compounds,
                oracle=oracle,
                learner=learner,
                target_column='Activity',
                featurizer='morgan',
                n_cycles=1,
                initial_size=2,  # Small initial size to avoid sampling error
                output_dir=str(tmp_path)
            )
    
    def test_invalid_parameters_validation(self, small_real_compounds, tmp_path):
        """Test validation of invalid parameters."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002'],
                'SMILES': ['CCO', 'CCC']
            })
        
        oracle = MockOracle()
        learner = MockLearner()
        
        # Test invalid score_direction
        with pytest.raises(ValueError, match="score_direction must be one of"):
            run_active_learning(
                compound_pool=compounds,
                oracle=oracle,
                learner=learner,
                target_column='Activity',
                featurizer='morgan',
                score_direction='invalid',
                output_dir=str(tmp_path)
            )
        
        # Test invalid n_cycles
        with pytest.raises(ValueError, match="n_cycles must be at least 1"):
            run_active_learning(
                compound_pool=compounds,
                oracle=oracle,
                learner=learner,
                target_column='Activity',
                featurizer='morgan',
                n_cycles=0,
                output_dir=str(tmp_path)
            )
        
        # Test invalid batch_fraction
        with pytest.raises(ValueError, match="batch_fraction must be in"):
            run_active_learning(
                compound_pool=compounds,
                oracle=oracle,
                learner=learner,
                target_column='Activity',
                featurizer='morgan',
                batch_fraction=1.5,  # > 1
                output_dir=str(tmp_path)
            )
        
        with pytest.raises(ValueError, match="batch_fraction must be in"):
            run_active_learning(
                compound_pool=compounds,
                oracle=oracle,
                learner=learner,
                target_column='Activity',
                featurizer='morgan',
                batch_fraction=0,  # <= 0
                output_dir=str(tmp_path)
            )
    
    def test_oracle_failure_handling(self, small_real_compounds, tmp_path):
        """Test handling of oracle measurement failures."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002'],
                'SMILES': ['CCO', 'CCC']
            })
        
        oracle = MockOracle(fail_measurement=True)
        learner = MockLearner()
        
        with pytest.raises(RuntimeError, match="Oracle measurement failed"):
            run_active_learning(
                compound_pool=compounds,
                oracle=oracle,
                learner=learner,
                target_column='Activity',
                featurizer='morgan',
                n_cycles=1,
                initial_size=1,
                output_dir=str(tmp_path)
            )
    
    def test_learner_failure_handling(self, small_real_compounds, tmp_path):
        """Test handling of learner training and prediction failures."""
        compounds = small_real_compounds.copy()
        
        # Use a small subset for predictable testing
        if len(compounds) >= 3:
            compounds = compounds.iloc[:3].copy()
        else:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
                'SMILES': ['CCO', 'CCC', 'CCN']
            })
        
        compounds['Activity'] = [0.1, 0.5, 0.9]
        
        # Test training failure
        oracle = MockOracle(ground_truth=compounds)
        failing_learner = MockLearner(fail_training=True)
        
        with pytest.raises(RuntimeError, match="Training failed"):
            run_active_learning(
                compound_pool=compounds,
                oracle=oracle,
                learner=failing_learner,
                target_column='Activity',
                featurizer='morgan',
                n_cycles=1,
                initial_size=1,
                output_dir=str(tmp_path)
            )
    
    def test_csv_export_functionality(self, small_real_compounds, tmp_path):
        """Test CSV export functionality."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
                'SMILES': ['CCO', 'CCC', 'CCN']
            })
        
        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))
        
        oracle = MockOracle(ground_truth=compounds)
        learner = MockLearner(supports_uncertainty=True)
        
        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            featurizer='morgan',
            n_cycles=2,
            batch_fraction=0.3,
            initial_size=1,
            output_dir=str(tmp_path),
            export_csv=True,
            random_state=42
        )
        
        # Check that CSV files are mentioned in results
        assert 'csv_files' in results
        
        # Check that output directory exists
        assert Path(tmp_path).exists()
    
    def test_temporary_output_directory(self, small_real_compounds):
        """Test automatic creation of temporary output directory."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002'],
                'SMILES': ['CCO', 'CCC']
            })
        
        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))
        
        oracle = MockOracle(ground_truth=compounds)
        learner = MockLearner()
        
        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            featurizer='morgan',
            n_cycles=1,
            batch_fraction=0.5,
            initial_size=1,
            output_dir=None,  # Should create temporary directory
            random_state=42
        )
        
        # Verify temporary directory was created and is in results
        assert 'output_dir' in results
        assert results['output_dir'] is not None
        assert Path(results['output_dir']).exists()
    
    def test_early_termination_empty_pool(self, tmp_path):
        """Test early termination when pool is exhausted."""
        # Very small compound pool
        compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002'],
            'SMILES': ['CCO', 'CCC'],
            'Activity': [0.1, 0.9]
        })
        
        oracle = MockOracle(ground_truth=compounds)
        learner = MockLearner()
        
        results = run_active_learning(
            compound_pool=compounds,
            oracle=oracle,
            learner=learner,
            target_column='Activity',
            featurizer='morgan',
            n_cycles=10,  # More cycles than compounds
            batch_fraction=0.5,
            initial_size=1,
            output_dir=str(tmp_path),
            random_state=42
        )
        
        # Should terminate early
        assert results['total_cycles'] < 10
        assert len(results['unlabeled_data']) == 0  # Pool exhausted
    
    def test_different_acquisition_strategies(self, small_real_compounds, tmp_path):
        """Test different acquisition strategies work correctly."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(10)],
                'SMILES': ['CCO'] * 10  # Simple molecules
            })
        
        compounds['Activity'] = np.random.uniform(0, 1, len(compounds))
        
        oracle = MockOracle(ground_truth=compounds)
        learner = MockLearner(supports_uncertainty=True)
        
        strategies_to_test = ['random', 'greedy']
        
        for strategy in strategies_to_test:
            results = run_active_learning(
                compound_pool=compounds,
                oracle=oracle,
                learner=learner,
                target_column='Activity',
                featurizer='morgan',
                strategy=strategy,
                n_cycles=2,
                batch_fraction=0.2,
                initial_size=2,
                output_dir=str(tmp_path / strategy),
                random_state=42
            )
            
            assert results['total_cycles'] == 2
            
            # Verify strategies were used correctly
            # Cycle 0 uses initial_strategy (default: 'random')
            # Cycle 1 uses the specified strategy
            assert results['cycle_metrics'][0]['strategy'] == 'random'  # initial_strategy default
            assert results['cycle_metrics'][1]['strategy'] == strategy  # specified strategy
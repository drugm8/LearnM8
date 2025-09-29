"""
Tests for learner creation functionality in learnm8.py.

Tests the create_learner_from_string function for proper learner instantiation,
error handling, and validation with real molecular data.
"""

import pytest
from unittest.mock import patch

from learnm8.learnm8 import create_learner_from_string
from learnm8.core.interfaces import Learner


class TestCreateLearnerFromString:
    """Test learner creation from string specifications."""
    
    def test_valid_learner_creation(self):
        """Test creation of all valid learner types."""
        valid_learners = [
            'rf', 'gp', 'xgb', 'mlp', 'mc_dropout',
            'ensemble', 'rf_ensemble', 'lr_ensemble', 
            'xgb_ensemble', 'dt_ensemble', 'mixed_ensemble'
        ]
        
        for learner_name in valid_learners:
            learner = create_learner_from_string(learner_name, featurizer_type='morgan')
            
            # Verify it's a Learner instance
            assert isinstance(learner, Learner)
            
            # Verify it has required interface methods
            assert hasattr(learner, 'train')
            assert hasattr(learner, 'predict')
            assert hasattr(learner, 'supports_uncertainty')
            assert callable(learner.train)
            assert callable(learner.predict)
            assert callable(learner.supports_uncertainty)
    
    def test_sklearn_learners_creation(self):
        """Test creation of sklearn-based learners."""
        sklearn_learners = ['rf', 'gp', 'xgb']
        
        for learner_name in sklearn_learners:
            learner = create_learner_from_string(learner_name, featurizer_type='morgan')
            assert isinstance(learner, Learner)
            
            # Test that these can be instantiated without errors
            assert learner is not None
    
    def test_torch_learners_creation(self):
        """Test creation of PyTorch-based learners."""
        torch_learners = ['mlp', 'mc_dropout']
        
        for learner_name in torch_learners:
            try:
                learner = create_learner_from_string(learner_name, featurizer_type='morgan')
                assert isinstance(learner, Learner)
                assert learner is not None
            except ImportError:
                pytest.skip(f"PyTorch not available for {learner_name}")
    
    def test_ensemble_learners_creation(self):
        """Test creation of ensemble learners."""
        ensemble_learners = [
            'ensemble', 'rf_ensemble', 'lr_ensemble', 
            'xgb_ensemble', 'dt_ensemble', 'mixed_ensemble'
        ]
        
        for learner_name in ensemble_learners:
            learner = create_learner_from_string(learner_name, featurizer_type='morgan')
            assert isinstance(learner, Learner)
            
            # Ensemble learners should support uncertainty
            assert learner.supports_uncertainty()
    
    def test_learner_creation_with_kwargs(self):
        """Test learner creation with additional keyword arguments."""
        # Test with random forest parameters
        learner = create_learner_from_string('rf', featurizer_type='morgan', n_estimators=50, random_state=42)
        assert isinstance(learner, Learner)
        
        # Test with gaussian process parameters
        learner = create_learner_from_string('gp', featurizer_type='morgan', random_state=42)
        assert isinstance(learner, Learner)
    
    def test_invalid_learner_name(self):
        """Test error handling for invalid learner names."""
        invalid_names = [
            'invalid_learner',
            'random_forest',  # Wrong name format
            'GP',  # Case sensitive
            'svm',  # Not supported
            '',  # Empty string
            'rf_wrong'
        ]
        
        for invalid_name in invalid_names:
            with pytest.raises(ValueError) as exc_info:
                create_learner_from_string(invalid_name, featurizer_type='morgan')
            
            # Verify error message is helpful
            error_msg = str(exc_info.value)
            assert f"Unknown learner '{invalid_name}'" in error_msg
            assert "Available:" in error_msg
    
    def test_error_message_contains_available_learners(self):
        """Test that error messages list available learners."""
        with pytest.raises(ValueError) as exc_info:
            create_learner_from_string('nonexistent', featurizer_type='morgan')
        
        error_msg = str(exc_info.value)
        
        # Should contain some known learner names
        expected_learners = ['rf', 'gp', 'xgb', 'ensemble']
        for learner_name in expected_learners:
            assert learner_name in error_msg
    
    def test_learner_mapping_consistency(self):
        """Test that learner mapping is consistent and complete."""
        # This test ensures the mapping dictionary contains expected entries
        with pytest.raises(ValueError) as exc_info:
            create_learner_from_string('invalid', featurizer_type='morgan')
        
        error_msg = str(exc_info.value)
        available_text = error_msg.split("Available: ")[1]
        available_learners = available_text.split(", ")
        
        # Expected minimum set of learners
        expected_minimum = {'rf', 'gp', 'xgb', 'ensemble'}
        available_set = set(learner for learner in available_learners)
        
        assert expected_minimum.issubset(available_set)
    
    def test_duplicate_ensemble_mapping(self):
        """Test that ensemble and mixed_ensemble create the same type."""
        ensemble_learner = create_learner_from_string('ensemble', featurizer_type='morgan')
        mixed_ensemble_learner = create_learner_from_string('mixed_ensemble', featurizer_type='morgan')
        
        # Should create the same type of learner
        assert type(ensemble_learner) == type(mixed_ensemble_learner)
    
    def test_learner_instantiation_with_missing_dependencies(self):
        """Test graceful handling of missing dependencies."""
        # Mock missing PyTorch for torch learners
        with patch.dict('sys.modules', {'torch': None}):
            # This might raise ImportError or work depending on implementation
            # We just ensure it doesn't crash unexpectedly
            try:
                learner = create_learner_from_string('mlp', featurizer_type='morgan')
                assert isinstance(learner, Learner)
            except ImportError:
                # This is acceptable behavior
                pass
    
    def test_learner_parameter_validation(self):
        """Test that invalid parameters to learners are handled."""
        # Test with obviously invalid parameters
        try:
            # Some learners might validate parameters, others might not
            learner = create_learner_from_string('rf', featurizer_type='morgan', invalid_param='invalid_value')
            assert isinstance(learner, Learner)
        except (TypeError, ValueError):
            # This is acceptable - the underlying learner rejected invalid params
            pass
    
    def test_case_sensitivity(self):
        """Test that learner names are case sensitive."""
        # These should fail
        invalid_cases = ['RF', 'GP', 'XGB', 'Ensemble', 'ENSEMBLE']
        
        for invalid_case in invalid_cases:
            with pytest.raises(ValueError):
                create_learner_from_string(invalid_case, featurizer_type='morgan')
    
    def test_learner_interface_compliance(self):
        """Test that created learners comply with Learner interface."""
        learner_names = ['rf', 'gp']  # Test with basic learners
        
        for learner_name in learner_names:
            learner = create_learner_from_string(learner_name, featurizer_type='morgan')
            
            # Test interface methods exist and are callable
            assert hasattr(learner, 'train')
            assert hasattr(learner, 'predict') 
            assert hasattr(learner, 'supports_uncertainty')
            
            assert callable(getattr(learner, 'train'))
            assert callable(getattr(learner, 'predict'))
            assert callable(getattr(learner, 'supports_uncertainty'))
            
            # supports_uncertainty should return boolean
            uncertainty_support = learner.supports_uncertainty()
            assert isinstance(uncertainty_support, bool)
    
    def test_learner_creation_determinism(self):
        """Test that creating the same learner twice gives consistent results."""
        learner1 = create_learner_from_string('rf', featurizer_type='morgan', random_state=42)
        learner2 = create_learner_from_string('rf', featurizer_type='morgan', random_state=42)
        
        # Should be the same type
        assert type(learner1) == type(learner2)
        
        # Should have same uncertainty support
        assert learner1.supports_uncertainty() == learner2.supports_uncertainty()
    
    def test_learner_with_complex_parameters(self):
        """Test learner creation with complex parameter combinations."""
        # Test random forest with multiple parameters
        learner = create_learner_from_string(
            'rf',
            featurizer_type='morgan',
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=1
        )
        assert isinstance(learner, Learner)
        
        # Test gaussian process with kernel parameters  
        learner = create_learner_from_string(
            'gp',
            featurizer_type='morgan',
            random_state=42,
            normalize_y=True
        )
        assert isinstance(learner, Learner)
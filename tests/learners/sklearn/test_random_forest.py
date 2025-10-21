"""Tests for RandomForestLearner implementation."""

import pytest
import numpy as np
import pandas as pd

from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.features.extraction import extract_features


class TestRandomForestLearner:
    """Test RandomForestLearner functionality with real molecular data."""

    @pytest.fixture
    def learner(self):
        """Create RandomForestLearner instance for testing."""
        return RandomForestLearner(n_estimators=10, random_state=42)
    
    def test_initialization(self, learner):
        """Test learner initialization."""
        assert learner.n_estimators == 10
        assert learner.random_state == 42
        assert not learner.is_trained
        assert learner.supports_uncertainty() is False
    
    def test_train_predict_integration(self, learner, small_real_compounds, tmp_path):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        targets = compounds['Activity'].values

        learner.train(features, targets)
        assert learner.is_trained

        predictions, uncertainty = learner.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is None
        assert np.all(np.isfinite(predictions))
    
    def test_train_with_empty_arrays(self, learner):
        """Test error handling with empty arrays."""
        empty_features = np.array([]).reshape(0, 10)
        empty_targets = np.array([])

        with pytest.raises(ValueError, match="Cannot train on empty dataset"):
            learner.train(empty_features, empty_targets)

    def test_train_with_mismatched_shapes(self, learner):
        """Test error handling with mismatched feature/target shapes."""
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)

        with pytest.raises(ValueError, match="Features and targets must have same length"):
            learner.train(features, targets)

    def test_predict_without_training(self, learner):
        """Test error when predicting without training."""
        features = np.random.randn(5, 10)
        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            learner.predict(features)
    
    def test_get_name(self, learner):
        """Test name generation."""
        name = learner.get_name()
        assert "RandomForest" in name
        assert "n_estimators=10" in name
    
    def test_feature_importance(self, learner, small_real_compounds, tmp_path):
        """Test feature importance retrieval."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        assert learner.get_feature_importance() is None

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        importance = learner.get_feature_importance()
        assert importance is not None
        assert len(importance) > 0
        assert np.all(importance >= 0)

    def test_oob_score(self, learner, small_real_compounds, tmp_path):
        """Test out-of-bag score retrieval."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        assert learner.get_oob_score() is None

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        oob_score = learner.get_oob_score()
        assert oob_score is not None
        assert isinstance(oob_score, float)
    
    def test_different_hyperparameters(self, tmp_path, small_real_compounds):
        """Test learner with different hyperparameters."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner = RandomForestLearner(
            n_estimators=5,
            max_depth=3,
            min_samples_split=5,
            random_state=42
        )

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        predictions, _ = learner.predict(features)

        assert learner.max_depth == 3
        assert predictions.shape[0] == len(compounds)

    def test_edge_case_single_compound(self, learner, tmp_path):
        """Test with single compound."""
        single_compound = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        features = extract_features(single_compound['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, single_compound['Activity'].values)
        predictions, _ = learner.predict(features)

        assert len(predictions) == 1
        assert np.isfinite(predictions[0])
"""Tests for DecisionTreeLearner implementation."""

import pytest
import numpy as np
import pandas as pd

from learnm8.learners.sklearn.decision_tree import DecisionTreeLearner
from learnm8.features.extraction import extract_features


class TestDecisionTreeLearner:
    """Test DecisionTreeLearner functionality with real molecular data."""

    @pytest.fixture
    def learner(self):
        """Create DecisionTreeLearner instance for testing."""
        return DecisionTreeLearner(max_depth=10, random_state=42)

    def test_initialization(self):
        """Test DecisionTreeLearner initializes with correct parameters."""
        learner = DecisionTreeLearner(
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=123
        )

        assert learner.max_depth == 15
        assert learner.min_samples_split == 5
        assert learner.min_samples_leaf == 3
        assert learner.random_state == 123
        assert not learner.is_trained

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
        assert "DecisionTree" in name

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
        assert importance.shape[0] == features.shape[1]
        assert np.all(importance >= 0)
        assert np.isclose(importance.sum(), 1.0)

    def test_supports_uncertainty(self, learner):
        """Test DecisionTreeLearner does not support uncertainty."""
        assert learner.supports_uncertainty() is False

    def test_train_with_1d_features(self, learner):
        """Test training with 1D features raises error."""
        features_1d = np.random.rand(10)
        targets = np.random.rand(10)

        with pytest.raises((ValueError, RuntimeError)):
            learner.train(features_1d, targets)

    def test_deterministic_predictions(self, small_real_compounds, tmp_path):
        """Test predictions are deterministic with same random_state."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        targets = compounds['Activity'].values

        learner1 = DecisionTreeLearner(random_state=42)
        learner1.train(features, targets)
        pred1, _ = learner1.predict(features)

        learner2 = DecisionTreeLearner(random_state=42)
        learner2.train(features, targets)
        pred2, _ = learner2.predict(features)

        np.testing.assert_array_equal(pred1, pred2)

    def test_max_depth_parameter(self, small_real_compounds, tmp_path):
        """Test max_depth parameter affects model complexity."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        targets = compounds['Activity'].values

        learner_shallow = DecisionTreeLearner(max_depth=2, random_state=42)
        learner_shallow.train(features, targets)

        learner_deep = DecisionTreeLearner(max_depth=20, random_state=42)
        learner_deep.train(features, targets)

        assert learner_shallow.model.get_depth() <= 2
        assert learner_deep.model.get_depth() > learner_shallow.model.get_depth()

"""Tests for AdvancedRandomForestLearner implementation."""

import pytest
import numpy as np
import pandas as pd

from learnm8.learners.sklearn.advanced_random_forest import AdvancedRandomForestLearner
from learnm8.features.extraction import extract_features


class TestAdvancedRandomForestLearner:
    """Test AdvancedRandomForestLearner functionality with real molecular data."""

    @pytest.fixture
    def learner(self):
        """Create AdvancedRandomForestLearner instance for testing."""
        return AdvancedRandomForestLearner(n_estimators=10, random_state=42)

    def test_initialization(self, learner):
        """Test learner initialization with advanced hyperparameters."""
        assert learner.n_estimators == 10
        assert learner.random_state == 42
        assert learner.max_depth == 15
        assert learner.max_samples == 0.8
        assert learner.ccp_alpha == 0.001
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

    def test_oob_score_enabled(self, learner, small_real_compounds, tmp_path):
        """Test out-of-bag scoring functionality."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        assert learner.get_oob_score() is None

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)

        oob_score = learner.get_oob_score()
        assert oob_score is not None
        assert isinstance(oob_score, float)
        assert -1.0 <= oob_score <= 1.0

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
        assert len(importance) == features.shape[1]
        assert np.all(importance >= 0)
        assert np.isclose(np.sum(importance), 1.0)

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
        """Test name generation format."""
        name = learner.get_name()
        assert "AdvancedRandomForest" in name
        assert "n_estimators=10" in name
        assert "depth=15" in name
        assert "pruning=0.001" in name

    def test_supports_uncertainty(self, learner):
        """Test that AdvancedRandomForestLearner does not support uncertainty."""
        assert learner.supports_uncertainty() is False

        compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002'],
            'SMILES': ['CCO', 'CCC'],
            'Activity': [0.3, 0.7]
        })

        features = np.random.randn(2, 10)
        learner.train(features, compounds['Activity'].values)
        predictions, uncertainty = learner.predict(features)

        assert uncertainty is None

    def test_get_tree_stats(self, learner, small_real_compounds, tmp_path):
        """Test tree statistics retrieval."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        assert learner.get_tree_stats() is None

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)

        stats = learner.get_tree_stats()
        assert stats is not None
        assert 'n_trees' in stats
        assert 'avg_depth' in stats
        assert 'max_depth' in stats
        assert 'avg_nodes' in stats
        assert 'total_nodes' in stats
        assert 'oob_score' in stats

        assert stats['n_trees'] == learner.n_estimators
        assert stats['max_depth'] <= learner.max_depth
        assert stats['avg_depth'] > 0
        assert stats['total_nodes'] > 0

    def test_advanced_hyperparameters(self, tmp_path, small_real_compounds):
        """Test learner with advanced hyperparameters configuration."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner = AdvancedRandomForestLearner(
            n_estimators=50,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            max_samples=0.7,
            min_impurity_decrease=0.001,
            ccp_alpha=0.01,
            random_state=42
        )

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        predictions, _ = learner.predict(features)

        assert learner.n_estimators == 50
        assert learner.max_depth == 10
        assert learner.max_samples == 0.7
        assert learner.ccp_alpha == 0.01
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

    def test_regularization_effect(self, tmp_path, small_real_compounds):
        """Test that regularization parameters affect model behavior."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner_no_pruning = AdvancedRandomForestLearner(
            n_estimators=10,
            ccp_alpha=0.0,
            min_impurity_decrease=0.0,
            random_state=42
        )

        learner_with_pruning = AdvancedRandomForestLearner(
            n_estimators=10,
            ccp_alpha=0.01,
            min_impurity_decrease=0.001,
            random_state=42
        )

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)

        learner_no_pruning.train(features, compounds['Activity'].values)
        learner_with_pruning.train(features, compounds['Activity'].values)

        stats_no_pruning = learner_no_pruning.get_tree_stats()
        stats_with_pruning = learner_with_pruning.get_tree_stats()

        assert stats_no_pruning['total_nodes'] >= stats_with_pruning['total_nodes']

    def test_bootstrap_sampling(self, tmp_path, small_real_compounds):
        """Test that bootstrap sampling affects model training."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner = AdvancedRandomForestLearner(
            n_estimators=10,
            bootstrap=True,
            oob_score=True,
            max_samples=0.8,
            random_state=42
        )

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)

        assert learner.get_oob_score() is not None
        stats = learner.get_tree_stats()
        assert stats is not None
        assert stats['oob_score'] is not None

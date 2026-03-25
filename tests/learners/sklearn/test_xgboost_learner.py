"""Tests for XGBoostLearner implementation."""

import pytest
import numpy as np
import polars as pl
from unittest.mock import Mock

from learnm8.exceptions import LearnerError
from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner
from learnm8.features.extraction import extract_features


@pytest.mark.integration
@pytest.mark.molecular
class TestXGBoostLearner:
    """Test XGBoostLearner functionality with real molecular data."""

    @pytest.fixture
    def learner(self):
        """Create XGBoostLearner instance for testing."""
        return XGBoostLearner(n_estimators=10, random_state=42)

    def test_initialization_sets_default_hyperparameters_and_no_uncertainty_support(self, learner):
        """Test learner initialization."""
        assert learner.n_estimators == 10
        assert learner.learning_rate == 0.1
        assert learner.max_depth == 6
        assert learner.random_state == 42
        assert not learner.is_trained
        assert learner.supports_uncertainty() is False

    def test_predict_returns_finite_values_without_uncertainty_after_training(self, learner, small_real_compounds, small_real_morgan_features):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity'))

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        assert learner.is_trained

        predictions, uncertainty = learner.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is None
        assert np.all(np.isfinite(predictions))

    def test_predict_without_training(self, learner, small_real_morgan_features):
        """Test error when predicting without training."""
        with pytest.raises(LearnerError, match="must be trained before prediction"):
            learner.predict(small_real_morgan_features)

    def test_get_name_includes_estimators_learning_rate_and_depth(self, learner):
        """Test name generation."""
        name = learner.get_name()
        assert "XGBoost" in name
        assert "n_estimators=10" in name
        assert "lr=0.1" in name
        assert "depth=6" in name

    def test_feature_importance_returns_non_negative_values_after_training(self, learner, small_real_compounds, small_real_morgan_features):
        """Test feature importance retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity'))

        assert learner.get_feature_importance() is None

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        importance = learner.get_feature_importance()
        assert importance is not None
        assert len(importance) > 0
        assert np.all(importance >= 0)

    def test_booster_stats(self, learner, small_real_compounds, small_real_morgan_features):
        """Test booster statistics retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity'))

        assert learner.get_booster_stats() is None

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        stats = learner.get_booster_stats()
        if stats is not None:
            assert isinstance(stats, dict)
            assert 'num_boosted_rounds' in stats or 'num_features' in stats

    def test_custom_hyperparameters_train_and_predict(self, small_real_compounds, small_real_morgan_features):
        """Test learner with different hyperparameters."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity'))

        learner = XGBoostLearner(
            n_estimators=5,
            learning_rate=0.2,
            max_depth=3,
            subsample=0.7,
            random_state=42
        )

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert learner.n_estimators == 5
        assert learner.learning_rate == 0.2
        assert learner.max_depth == 3
        assert predictions.shape[0] == len(compounds)

    def test_small_diverse_dataset_trains_and_predicts_finite_values(self, learner, tmp_path):
        """Test with small diverse dataset."""
        small_compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(5)],
            'SMILES': ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'C1CCNCC1'],
            'Activity': [0.5, 0.3, 0.8, 0.2, 0.6]
        })

        features = extract_features(small_compounds['SMILES'].to_list(), 'morgan', tmp_path)
        learner.train(features, small_compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert len(predictions) == 5
        assert np.all(np.isfinite(predictions))

    def test_large_dataset_handling(self, learner, tmp_path):
        """Test XGBoost efficiency with larger datasets."""
        n_compounds = 500
        large_compounds = pl.DataFrame({
            'ID': [f'COMP_{i:04d}' for i in range(n_compounds)],
            'SMILES': [f'C{"C" * (i % 10)}O' for i in range(n_compounds)],
            'Activity': np.random.beta(2, 5, n_compounds)
        })

        features = extract_features(large_compounds['SMILES'].to_list(), 'morgan', tmp_path)
        learner.train(features, large_compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert len(predictions) == n_compounds
        assert np.all(np.isfinite(predictions))

    def test_numerical_stability(self, learner, tmp_path):
        """Test numerical stability with extreme values."""
        compounds = pl.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCN'],
            'Activity': [0.0, 1000.0, 0.001]
        })

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert np.all(np.isfinite(predictions))

    def test_regularization_parameters(self, small_real_compounds, small_real_morgan_features):
        """Test learner with regularization parameters."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity'))

        learner = XGBoostLearner(
            n_estimators=10,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=42
        )

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))

    def test_uncertainty_consistency(self, learner, small_real_compounds, small_real_morgan_features):
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity'))

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())

        predictions, uncertainty = learner.predict(features)

        assert learner.supports_uncertainty() is False
        assert uncertainty is None

    def test_train_with_empty_arrays(self, learner):
        empty_features = np.array([]).reshape(0, 10)
        empty_targets = np.array([])

        with pytest.raises(LearnerError, match="empty dataset"):
            learner.train(empty_features, empty_targets)

    def test_train_with_mismatched_shapes(self, learner):
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)

        with pytest.raises(LearnerError, match="mismatched lengths"):
            learner.train(features, targets)

    def test_train_with_1d_features(self, learner):
        features_1d = np.random.rand(10)
        targets = np.random.rand(10)

        with pytest.raises((ValueError, RuntimeError)):
            learner.train(features_1d, targets)

"""Tests for RandomForestLearner implementation."""

import numpy as np
import polars as pl
import pytest

from learnm8.exceptions import LearnerError
from learnm8.features.extraction import extract_features
from learnm8.learners.sklearn.random_forest import RandomForestLearner


@pytest.mark.integration
@pytest.mark.molecular
class TestRandomForestLearner:
    """Test RandomForestLearner functionality with real molecular data."""

    @pytest.fixture
    def learner(self):
        """Create RandomForestLearner instance for testing."""
        return RandomForestLearner(n_estimators=10, random_state=42)

    def test_initialization_sets_forest_parameters_and_uncertainty_support(self, learner):
        """Test learner initialization."""
        assert learner.n_estimators == 10
        assert learner.random_state == 42
        assert not learner.is_trained
        assert learner.supports_uncertainty() is True

    def test_predict_returns_finite_values_and_uncertainty_after_training(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity')
            )

        features = small_real_morgan_features
        targets = compounds['Activity'].to_numpy()

        learner.train(features, targets)
        assert learner.is_trained

        predictions, uncertainty = learner.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert np.all(np.isfinite(predictions))
        assert np.all(np.isfinite(uncertainty))

    def test_train_with_empty_arrays(self, learner):
        """Test error handling with empty arrays."""
        empty_features = np.array([]).reshape(0, 10)
        empty_targets = np.array([])

        with pytest.raises(LearnerError, match=r'Cannot train .* on an empty dataset'):
            learner.train(empty_features, empty_targets)

    def test_train_with_mismatched_shapes(self, learner):
        """Test error handling with mismatched feature/target shapes."""
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)

        with pytest.raises(LearnerError, match='mismatched lengths'):
            learner.train(features, targets)

    def test_predict_without_training(self, learner):
        """Test error when predicting without training."""
        features = np.random.randn(5, 10)
        with pytest.raises(LearnerError, match='must be trained before prediction'):
            learner.predict(features)

    def test_get_name_includes_estimator_count(self, learner):
        """Test name generation."""
        name = learner.get_name()
        assert 'RandomForest' in name
        assert 'n_estimators=10' in name

    def test_feature_importance_returns_non_negative_values_after_training(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        """Test feature importance retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity')
            )

        assert learner.get_feature_importance() is None

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        importance = learner.get_feature_importance()
        assert importance is not None
        assert len(importance) > 0
        assert np.all(importance >= 0)

    def test_get_oob_score_returns_float_after_training(self, learner, small_real_compounds, small_real_morgan_features):
        """Test out-of-bag score retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity')
            )

        assert learner.get_oob_score() is None

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        oob_score = learner.get_oob_score()
        assert oob_score is not None
        assert isinstance(oob_score, float)

    def test_custom_hyperparameters_train_and_predict(
        self, small_real_compounds, small_real_morgan_features
    ):
        """Test learner with different hyperparameters."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity')
            )

        learner = RandomForestLearner(
            n_estimators=5, max_depth=3, min_samples_split=5, random_state=42
        )

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert learner.max_depth == 3
        assert predictions.shape[0] == len(compounds)

    def test_small_diverse_dataset_trains_and_predicts_finite_values(self, learner, tmp_path):
        """Test with small diverse dataset."""
        small_compounds = pl.DataFrame(
            {
                'ID': [f'COMP_{i:03d}' for i in range(5)],
                'SMILES': ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'C1CCNCC1'],
                'Activity': [0.5, 0.3, 0.8, 0.2, 0.6],
            }
        )

        features = extract_features(
            small_compounds['SMILES'].to_list(), 'morgan', tmp_path
        )
        learner.train(features, small_compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert len(predictions) == 5
        assert np.all(np.isfinite(predictions))

    def test_repeated_predictions_return_same_predictions_and_uncertainty(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        """Test that uncertainty is returned and consistent across calls."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity')
            )

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())

        predictions1, uncertainty1 = learner.predict(features)
        predictions2, uncertainty2 = learner.predict(features)

        assert learner.supports_uncertainty() is True
        assert uncertainty1 is not None
        assert uncertainty2 is not None
        np.testing.assert_array_almost_equal(predictions1, predictions2)
        np.testing.assert_array_almost_equal(uncertainty1, uncertainty2)

    def test_uncertainty_shape_matches_predictions(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        """Test that uncertainty array shape matches predictions shape."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity')
            )

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())

        predictions, uncertainty = learner.predict(features)
        assert uncertainty is not None
        assert uncertainty.shape == predictions.shape

    def test_uncertainty_non_negative(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        """Test that all uncertainty values are >= 0."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity')
            )

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())

        _, uncertainty = learner.predict(features)
        assert uncertainty is not None
        assert np.all(uncertainty >= 0)

    def test_uncertainty_uses_estimators(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        """Test that uncertainty uses estimators_ and count matches n_estimators."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity')
            )

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())

        assert hasattr(learner.model, 'estimators_')
        assert len(learner.model.estimators_) == learner.n_estimators

        predictions, uncertainty = learner.predict(features)
        assert uncertainty is not None
        assert uncertainty.shape == predictions.shape

    def test_predict_raises_when_estimators_are_removed_after_training(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        """Test LearnerError raised if estimators_ removed after training."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity')
            )

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())

        del learner.model.estimators_

        with pytest.raises(LearnerError):
            learner.predict(features)

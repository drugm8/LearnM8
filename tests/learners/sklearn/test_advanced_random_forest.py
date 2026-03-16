"""Tests for AdvancedRandomForestLearner implementation."""

import numpy as np
import polars as pl
import pytest

from learnm8.exceptions import LearnerError
from learnm8.features.extraction import extract_features
from learnm8.learners.sklearn.advanced_random_forest import AdvancedRandomForestLearner


@pytest.mark.integration
@pytest.mark.molecular
class TestAdvancedRandomForestLearner:
    """Test AdvancedRandomForestLearner functionality with real molecular data."""

    @pytest.fixture
    def learner(self):
        """Create AdvancedRandomForestLearner instance for testing."""
        return AdvancedRandomForestLearner(n_estimators=10, random_state=42)

    def test_initialization_sets_advanced_forest_defaults_and_uncertainty_support(self, learner):
        """Test learner initialization with advanced hyperparameters."""
        assert learner.n_estimators == 10
        assert learner.random_state == 42
        assert learner.max_depth == 15
        assert learner.max_samples == 0.8
        assert learner.ccp_alpha == 0.001
        assert not learner.is_trained
        assert learner.supports_uncertainty() is True

    def test_predict_returns_finite_values_and_uncertainty_after_training(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features
        targets = compounds['Activity'].to_numpy()

        learner.train(features, targets)
        assert learner.is_trained

        predictions, uncertainty = learner.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert np.all(np.isfinite(uncertainty))
        assert np.all(np.isfinite(predictions))

    def test_oob_score_enabled(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        """Test out-of-bag scoring functionality."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        assert learner.get_oob_score() is None

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())

        oob_score = learner.get_oob_score()
        assert oob_score is not None
        assert isinstance(oob_score, float)
        assert -1.0 <= oob_score <= 1.0

    def test_feature_importance_returns_normalized_values_after_training(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        """Test feature importance retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        assert learner.get_feature_importance() is None

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())

        importance = learner.get_feature_importance()
        assert importance is not None
        assert len(importance) == np.sum(learner._valid_feature_mask)
        assert np.all(importance >= 0)
        assert np.isclose(np.sum(importance), 1.0)

    def test_train_with_empty_arrays(self, learner):
        """Test error handling with empty arrays."""
        empty_features = np.array([]).reshape(0, 10)
        empty_targets = np.array([])

        with pytest.raises(
            (ValueError, LearnerError), match=r'Cannot train .* on an empty dataset'
        ):
            learner.train(empty_features, empty_targets)

    def test_train_with_mismatched_shapes(self, learner):
        """Test error handling with mismatched feature/target shapes."""
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)

        with pytest.raises((ValueError, LearnerError), match='mismatched lengths'):
            learner.train(features, targets)

    def test_predict_without_training(self, learner):
        """Test error when predicting without training."""
        features = np.random.randn(5, 10)
        with pytest.raises(RuntimeError, match='must be trained before prediction'):
            learner.predict(features)

    def test_get_name_includes_estimator_depth_and_pruning_configuration(self, learner):
        """Test name generation format."""
        name = learner.get_name()
        assert 'AdvancedRandomForest' in name
        assert 'n_estimators=10' in name
        assert 'depth=15' in name
        assert 'pruning=0.001' in name

    def test_supports_uncertainty_returns_true_and_predicts_uncertainty(self, learner):
        """Test that AdvancedRandomForestLearner supports uncertainty."""
        assert learner.supports_uncertainty() is True

        features = np.random.randn(10, 10)
        targets = np.random.randn(10)
        learner.train(features, targets)
        predictions, uncertainty = learner.predict(features)

        assert uncertainty is not None
        assert uncertainty.shape == predictions.shape
        assert np.all(uncertainty >= 0)

    def test_get_tree_stats(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        """Test tree statistics retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        assert learner.get_tree_stats() is None

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())

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

    def test_advanced_hyperparameters(
        self, small_real_compounds, small_real_morgan_features
    ):
        """Test learner with advanced hyperparameters configuration."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        learner = AdvancedRandomForestLearner(
            n_estimators=50,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            max_samples=0.7,
            min_impurity_decrease=0.001,
            ccp_alpha=0.01,
            random_state=42,
        )

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert learner.n_estimators == 50
        assert learner.max_depth == 10
        assert learner.max_samples == 0.7
        assert learner.ccp_alpha == 0.01
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

    def test_regularization_effect(
        self, small_real_compounds, small_real_morgan_features
    ):
        """Test that regularization parameters affect model behavior."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        learner_no_pruning = AdvancedRandomForestLearner(
            n_estimators=10, ccp_alpha=0.0, min_impurity_decrease=0.0, random_state=42
        )

        learner_with_pruning = AdvancedRandomForestLearner(
            n_estimators=10,
            ccp_alpha=0.01,
            min_impurity_decrease=0.001,
            random_state=42,
        )

        features = small_real_morgan_features

        learner_no_pruning.train(features, compounds['Activity'].to_numpy())
        learner_with_pruning.train(features, compounds['Activity'].to_numpy())

        stats_no_pruning = learner_no_pruning.get_tree_stats()
        stats_with_pruning = learner_with_pruning.get_tree_stats()

        assert stats_no_pruning['total_nodes'] >= stats_with_pruning['total_nodes']

    def test_bootstrap_sampling(self, small_real_compounds, small_real_morgan_features):
        """Test that bootstrap sampling affects model training."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        learner = AdvancedRandomForestLearner(
            n_estimators=10,
            bootstrap=True,
            oob_score=True,
            max_samples=0.8,
            random_state=42,
        )

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())

        assert learner.get_oob_score() is not None
        stats = learner.get_tree_stats()
        assert stats is not None
        assert stats['oob_score'] is not None

    def test_uncertainty_shape_matches_predictions(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )
        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = learner.predict(features)
        assert uncertainty is not None
        assert uncertainty.shape == predictions.shape

    def test_uncertainty_non_negative(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )
        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        _, uncertainty = learner.predict(features)
        assert uncertainty is not None
        assert np.all(uncertainty >= 0)
        assert np.all(np.isfinite(uncertainty))

    def test_uncertainty_consistency(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )
        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        _, unc1 = learner.predict(features)
        _, unc2 = learner.predict(features)
        np.testing.assert_array_equal(unc1, unc2)

    def test_uncertainty_uses_estimators(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )
        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())

        assert hasattr(learner.model, 'estimators_')
        assert len(learner.model.estimators_) == learner.n_estimators

    def test_uncertainty_matches_rf_pattern(self):
        """Same approach as RandomForestLearner — std across trees."""
        rng = np.random.RandomState(42)
        X = rng.randn(30, 5)
        y = rng.randn(30)

        learner = AdvancedRandomForestLearner(n_estimators=10, random_state=42)
        learner.train(X, y)
        _, unc = learner.predict(X)

        assert unc is not None
        assert np.all(unc >= 0)

    def test_uncertainty_std_formula_manual_validation(self):
        """Manual np.std vs learner output."""
        rng = np.random.RandomState(42)
        X = rng.randn(30, 5)
        y = rng.randn(30)

        learner = AdvancedRandomForestLearner(n_estimators=10, random_state=42)
        learner.train(X, y)

        preprocessed = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        variance = np.var(preprocessed, axis=0)
        mask = variance > 1e-10
        preprocessed = preprocessed[:, mask]

        tree_preds = np.array(
            [tree.predict(preprocessed) for tree in learner.model.estimators_]
        )
        expected = np.std(tree_preds, axis=0, ddof=0)

        _, actual = learner.predict(X)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    @pytest.mark.parametrize("n_estimators", [3, 10, 100])
    def test_uncertainty_across_estimator_counts(self, n_estimators):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 5)
        y = rng.randn(30)

        learner = AdvancedRandomForestLearner(
            n_estimators=n_estimators, random_state=42
        )
        learner.train(X, y)
        preds, unc = learner.predict(X)

        assert unc is not None
        assert unc.shape == preds.shape
        assert np.all(unc >= 0)
        assert np.all(np.isfinite(unc))
        assert len(learner.model.estimators_) == n_estimators

    def test_uncertainty_after_zero_variance_removal(self):
        rng = np.random.RandomState(42)
        X_good = rng.randn(30, 5)
        X_const = np.full((30, 3), 5.0)
        X = np.hstack([X_good, X_const])
        y = rng.randn(30)

        learner = AdvancedRandomForestLearner(n_estimators=10, random_state=42)
        learner.train(X, y)
        _, unc = learner.predict(X)

        assert unc is not None
        assert unc.shape == (30,)
        assert np.all(np.isfinite(unc))

    def test_uncertainty_with_tiny_dataset(self):
        rng = np.random.RandomState(42)
        X = rng.randn(3, 2)
        y = rng.randn(3)

        learner = AdvancedRandomForestLearner(
            n_estimators=5, oob_score=False, random_state=42
        )
        learner.train(X, y)
        preds, unc = learner.predict(X)

        assert preds.shape == (3,)
        assert unc is not None
        assert unc.shape == (3,)
        assert np.all(np.isfinite(unc))

    def test_learner_error_if_estimators_removed(
        self, learner, small_real_compounds, small_real_morgan_features
    ):
        """LearnerError raised if estimators_ removed after training."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )
        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())

        del learner.model.estimators_

        with pytest.raises(LearnerError):
            learner.predict(features)

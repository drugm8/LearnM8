import numpy as np
import pytest

from learnm8.learners.base import _preprocess_features
from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner

pytestmark = pytest.mark.unit


def _make_cpu_learners():
    return [
        pytest.param(lambda: RandomForestLearner(n_estimators=5, random_state=42), id='rf'),
        pytest.param(lambda: XGBoostLearner(n_estimators=5, random_state=42), id='xgb'),
    ]


class TestPreprocessFeatures:
    def test_nan_inf_handled_as_zero(self):
        features = np.array([[1.0, np.nan, np.inf], [2.0, -np.inf, 3.0]])
        result, _, _ = _preprocess_features(features, remove_zero_variance=False, is_training=True)
        assert np.all(np.isfinite(result))
        assert result[0, 1] == 0.0
        assert result[0, 2] == 0.0
        assert result[1, 0] == 2.0

    def test_zero_variance_columns_removed_during_train(self):
        features = np.array([[1.0, 5.0, 3.0], [2.0, 5.0, 4.0], [3.0, 5.0, 5.0]])
        result, mask, _ = _preprocess_features(features, is_training=True)
        assert result.shape[1] == 2
        assert mask is not None
        assert not mask[1]

    def test_zero_variance_mask_applied_on_predict(self):
        train_feats = np.array([[1.0, 5.0, 3.0], [2.0, 5.0, 4.0]])
        _, mask, _ = _preprocess_features(train_feats, is_training=True)
        pred_feats = np.array([[10.0, 99.0, 30.0]])
        result, _, _ = _preprocess_features(pred_feats, valid_feature_mask=mask, is_training=False)
        assert result.shape[1] == 2

    def test_zero_variance_mask_persists_across_multiple_predict_calls(self):
        train_feats = np.array([[1.0, 5.0], [2.0, 5.0]])
        _, mask, _ = _preprocess_features(train_feats, is_training=True)
        for _ in range(3):
            pred_feats = np.random.rand(5, 2)
            result, _, _ = _preprocess_features(pred_feats, valid_feature_mask=mask, is_training=False)
            assert result.shape[1] == 1

    def test_nan_replaced_before_variance_check(self):
        features = np.array([[np.nan, 1.0], [np.nan, 2.0], [np.nan, 3.0]])
        result, mask, _ = _preprocess_features(features, is_training=True)
        assert np.all(np.isfinite(result))
        assert not mask[0]

    def test_no_zero_variance_columns_returns_all_features(self):
        features = np.random.rand(10, 5) + np.arange(5)
        result, mask, _ = _preprocess_features(features, is_training=True)
        assert result.shape[1] == 5
        assert np.all(mask)

    def test_remove_zero_variance_false_skips_masking(self):
        features = np.array([[1.0, 5.0, 3.0], [2.0, 5.0, 4.0]])
        result, mask, _ = _preprocess_features(features, remove_zero_variance=False, is_training=True)
        assert result.shape[1] == 3
        assert mask is None

    def test_predict_without_mask_returns_all_columns(self):
        features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result, _, _ = _preprocess_features(features, valid_feature_mask=None, is_training=False)
        assert result.shape == features.shape

    def test_all_zero_variance_columns_produces_empty_features(self):
        features = np.ones((5, 3))
        result, mask, _ = _preprocess_features(features, is_training=True)
        assert result.shape[1] == 0
        assert not np.any(mask)

    def test_threshold_boundary(self):
        n = 100
        col_below = np.full(n, 1.0)
        col_above = np.linspace(0.0, 1.0, n)
        features = np.column_stack([col_below, col_above])
        result, mask, _ = _preprocess_features(features, is_training=True)
        assert result.shape[1] == 1
        assert not mask[0]
        assert mask[1]

    def test_continuous_nan_imputed_with_median(self):
        """T045: Continuous features use SimpleImputer with median strategy."""
        features = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]])
        result, _, imputer = _preprocess_features(
            features, remove_zero_variance=False, is_training=True, feature_type='continuous'
        )
        assert np.all(np.isfinite(result))
        assert imputer is not None
        assert result[0, 1] == pytest.approx(6.5)
        assert result[1, 2] == pytest.approx(6.0)

    def test_continuous_inf_converted_to_nan_then_imputed(self):
        features = np.array([[1.0, np.inf], [2.0, 3.0], [4.0, -np.inf]])
        result, _, imputer = _preprocess_features(
            features, remove_zero_variance=False, is_training=True, feature_type='continuous'
        )
        assert np.all(np.isfinite(result))
        assert imputer is not None

    def test_continuous_imputer_applied_on_predict(self):
        train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        _, _, imputer = _preprocess_features(
            train, remove_zero_variance=False, is_training=True, feature_type='continuous'
        )
        pred = np.array([[np.nan, 10.0]])
        result, _, _ = _preprocess_features(
            pred, remove_zero_variance=False, is_training=False, feature_type='continuous', imputer=imputer
        )
        assert result[0, 0] == pytest.approx(3.0)

    def test_binary_nan_triggers_warning(self, caplog):
        """T046: Binary features with NaN trigger a warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            features = np.array([[1.0, np.nan], [0.0, 1.0]])
            result, _, _ = _preprocess_features(
                features, remove_zero_variance=False, is_training=True, feature_type='binary'
            )
        assert 'NaN detected in binary fingerprint features' in caplog.text
        assert result[0, 1] == 0.0

    def test_binary_no_warning_when_clean(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            features = np.array([[1.0, 0.0], [0.0, 1.0]])
            result, _, _ = _preprocess_features(
                features, remove_zero_variance=False, is_training=True, feature_type='binary'
            )
        assert 'NaN detected' not in caplog.text


@pytest.mark.parametrize('learner_factory', _make_cpu_learners())
class TestLearnerPreprocessing:
    @pytest.mark.slow
    def test_nan_inf_handled_in_train(self, learner_factory):
        learner = learner_factory()
        features = np.random.rand(50, 10)
        features[0, 0] = np.nan
        features[1, 1] = np.inf
        targets = np.random.rand(50)
        learner.train(features, targets)
        preds, _ = learner.predict(features)
        assert np.all(np.isfinite(preds))

    @pytest.mark.slow
    def test_zero_variance_columns_handled(self, learner_factory):
        learner = learner_factory()
        features = np.random.rand(50, 10)
        features[:, 3] = 5.0
        targets = np.random.rand(50)
        learner.train(features, targets)
        preds, _ = learner.predict(features)
        assert preds.shape == (50,)

    @pytest.mark.slow
    def test_feature_shape_consistency_train_predict(self, learner_factory):
        learner = learner_factory()
        features = np.random.rand(50, 10)
        targets = np.random.rand(50)
        learner.train(features, targets)
        pred_feats = np.random.rand(20, 10)
        preds, _ = learner.predict(pred_feats)
        assert preds.shape == (20,)

    @pytest.mark.slow
    def test_nan_in_predict_handled(self, learner_factory):
        learner = learner_factory()
        features = np.random.rand(50, 10)
        targets = np.random.rand(50)
        learner.train(features, targets)
        pred_feats = np.random.rand(20, 10)
        pred_feats[5, 2] = np.nan
        pred_feats[10, 7] = -np.inf
        preds, _ = learner.predict(pred_feats)
        assert np.all(np.isfinite(preds))
        assert preds.shape == (20,)

    @pytest.mark.slow
    def test_zero_variance_mask_stored_after_train(self, learner_factory):
        learner = learner_factory()
        features = np.random.rand(50, 10)
        features[:, 0] = 1.0
        features[:, 9] = 0.0
        targets = np.random.rand(50)
        learner.train(features, targets)
        assert learner._valid_feature_mask is not None
        assert not learner._valid_feature_mask[0]
        assert not learner._valid_feature_mask[9]

    @pytest.mark.slow
    def test_multiple_predict_calls_consistent(self, learner_factory):
        learner = learner_factory()
        features = np.random.rand(50, 8)
        features[:, 2] = 3.14
        targets = np.random.rand(50)
        learner.train(features, targets)
        pred_feats = np.random.rand(15, 8)
        preds1, _ = learner.predict(pred_feats)
        preds2, _ = learner.predict(pred_feats)
        np.testing.assert_allclose(preds1, preds2, rtol=1e-12)

    @pytest.mark.slow
    def test_retrain_updates_zero_variance_mask(self, learner_factory):
        learner = learner_factory()
        features1 = np.random.rand(50, 5)
        features1[:, 1] = 7.0
        targets = np.random.rand(50)
        learner.train(features1, targets)
        assert not learner._valid_feature_mask[1]

        features2 = np.random.rand(50, 5)
        learner.train(features2, targets)
        assert learner._valid_feature_mask is not None
        assert np.all(learner._valid_feature_mask)

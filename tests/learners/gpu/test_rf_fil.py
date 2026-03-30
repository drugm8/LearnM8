"""Unit tests for RfFilLearner — all GPU dependencies are mocked."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from learnm8.exceptions import LearnerError
from learnm8.learners.gpu.rf_fil import RfFilLearner

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(n: int = 50, f: int = 10, dtype=np.float32) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n, f)).astype(dtype)


def _make_targets(n: int = 50) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.random(n).astype(np.float32)


def _make_per_tree(n_samples: int, n_trees: int = 100) -> np.ndarray:
    """Simulate FIL predict_per_tree output shape (n_samples, n_trees)."""
    rng = np.random.default_rng(42)
    return rng.random((n_samples, n_trees)).astype(np.float32)


def _mock_treelite(tl_model_mock: MagicMock | None = None) -> MagicMock:
    """Return a treelite mock suitable for sys.modules injection."""
    tl = MagicMock(name='treelite')
    tl.sklearn = MagicMock()
    if tl_model_mock is None:
        tl_model_mock = MagicMock()
    tl.sklearn.import_model.return_value = tl_model_mock
    return tl


def _mock_cuml(per_tree_output: np.ndarray | None = None) -> MagicMock:
    """Return a cuml mock with ForestInference returning per_tree_output."""
    cuml = MagicMock(name='cuml')
    fil_instance = MagicMock()
    if per_tree_output is not None:
        fil_instance.predict_per_tree.return_value = per_tree_output
    cuml.ForestInference.load.return_value = fil_instance
    return cuml


def _trained_learner(
    n_samples: int = 50,
    n_features: int = 10,
    n_trees: int = 10,
    per_tree_out: np.ndarray | None = None,
) -> tuple[RfFilLearner, np.ndarray, MagicMock, MagicMock]:
    """Return (learner, features, treelite_mock, cuml_mock) after training."""
    features = _make_features(n_samples, n_features)
    targets = _make_targets(n_samples)
    if per_tree_out is None:
        per_tree_out = _make_per_tree(n_samples, n_trees)

    tl_mock = _mock_treelite()
    cuml_mock = _mock_cuml(per_tree_out)

    learner = RfFilLearner(n_estimators=n_trees, random_state=0)
    with patch.dict(sys.modules, {'treelite': tl_mock, 'treelite.sklearn': tl_mock.sklearn, 'cuml': cuml_mock}):
        learner.train(features, targets)

    return learner, features, tl_mock, cuml_mock


# ---------------------------------------------------------------------------
# T011 tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_construction_with_valid_parameters(self):
        learner = RfFilLearner(
            n_estimators=50,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=7,
            n_jobs=2,
        )
        assert learner.get_name() == 'rf_fil'
        assert learner.supports_uncertainty() is True
        assert not learner.is_trained
        assert learner._fil_model is None
        assert learner.n_estimators == 50


class TestTraining:
    def test_train_casts_features_to_float32(self):
        features = _make_features(dtype=np.float64)
        targets = _make_targets()
        tl_mock = _mock_treelite()
        cuml_mock = _mock_cuml(_make_per_tree(50))

        captured = {}

        original_fit = None

        learner = RfFilLearner(n_estimators=5, random_state=0)

        import sklearn.ensemble as _ske

        original_fit = _ske.RandomForestRegressor.fit

        def capturing_fit(self_rf, X, y):
            captured['dtype'] = X.dtype
            return original_fit(self_rf, X, y)

        with (
            patch.dict(
                sys.modules,
                {'treelite': tl_mock, 'treelite.sklearn': tl_mock.sklearn, 'cuml': cuml_mock},
            ),
            patch.object(_ske.RandomForestRegressor, 'fit', capturing_fit),
        ):
            learner.train(features, targets)

        assert captured['dtype'] == np.float32

    def test_train_converts_to_fil_after_sklearn_fit(self):
        learner, _features, tl_mock, _cuml_mock = _trained_learner()
        assert learner.is_trained
        tl_mock.sklearn.import_model.assert_called_once()
        assert learner._fil_model is not None

    def test_empty_input_raises_error(self):
        learner = RfFilLearner(n_estimators=5, random_state=0)
        features = np.empty((0, 10), dtype=np.float32)
        targets = np.empty((0,), dtype=np.float32)
        with pytest.raises(LearnerError):
            learner.train(features, targets)


class TestPrediction:
    def test_predict_before_train_raises_learner_error(self):
        learner = RfFilLearner(n_estimators=5, random_state=0)
        features = _make_features()
        with pytest.raises(LearnerError, match='must be trained'):
            learner.predict(features)

    def test_predict_returns_predictions_and_uncertainty(self):
        n = 20
        n_trees = 8
        per_tree = _make_per_tree(n, n_trees)
        learner, _features, _, cuml_mock = _trained_learner(
            n_samples=50, n_features=10, n_trees=n_trees, per_tree_out=_make_per_tree(50, n_trees)
        )
        pred_features = _make_features(n, 10)
        cuml_mock.ForestInference.load.return_value.predict_per_tree.return_value = per_tree
        preds, unc = learner.predict(pred_features)
        assert preds is not None
        assert unc is not None

    def test_predict_output_shapes_match_input_length(self):
        n = 30
        n_trees = 6
        per_tree_train = _make_per_tree(50, n_trees)
        per_tree_pred = _make_per_tree(n, n_trees)
        learner, _, _, cuml_mock = _trained_learner(n_trees=n_trees, per_tree_out=per_tree_train)
        cuml_mock.ForestInference.load.return_value.predict_per_tree.return_value = per_tree_pred
        features = _make_features(n, 10)
        preds, unc = learner.predict(features)
        assert preds.shape == (n,)
        assert unc.shape == (n,)

    def test_uncertainty_is_finite_and_non_negative(self):
        n = 40
        n_trees = 10
        per_tree_train = _make_per_tree(50, n_trees)
        per_tree_pred = _make_per_tree(n, n_trees)
        learner, _, _, cuml_mock = _trained_learner(n_trees=n_trees, per_tree_out=per_tree_train)
        cuml_mock.ForestInference.load.return_value.predict_per_tree.return_value = per_tree_pred
        features = _make_features(n, 10)
        _, unc = learner.predict(features)
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0)

    def test_uncertainty_is_deterministic_for_fixed_input(self):
        n = 25
        n_trees = 8
        per_tree_train = _make_per_tree(50, n_trees)
        per_tree_pred = _make_per_tree(n, n_trees)
        learner, _, _, cuml_mock = _trained_learner(n_trees=n_trees, per_tree_out=per_tree_train)
        fil_instance = cuml_mock.ForestInference.load.return_value
        fil_instance.predict_per_tree.return_value = per_tree_pred
        features = _make_features(n, 10)
        _, unc1 = learner.predict(features)
        _, unc2 = learner.predict(features)
        np.testing.assert_array_equal(unc1, unc2)

    def test_uncertainty_is_std_ddof0_of_per_tree_predictions(self):
        n = 15
        n_trees = 6
        per_tree_train = _make_per_tree(50, n_trees)
        per_tree_pred = _make_per_tree(n, n_trees)
        learner, _, _, cuml_mock = _trained_learner(n_trees=n_trees, per_tree_out=per_tree_train)
        cuml_mock.ForestInference.load.return_value.predict_per_tree.return_value = per_tree_pred
        features = _make_features(n, 10)
        preds, unc = learner.predict(features)
        expected_mean = per_tree_pred.mean(axis=1)
        expected_std = np.std(per_tree_pred, axis=1, ddof=0)
        np.testing.assert_allclose(preds, expected_mean, rtol=1e-5)
        np.testing.assert_allclose(unc, expected_std, rtol=1e-5)


class TestErrorHandling:
    def test_fil_conversion_failure_raises_learner_error(self):
        features = _make_features()
        targets = _make_targets()
        tl_mock = _mock_treelite()
        tl_mock.sklearn.import_model.side_effect = RuntimeError('treelite serialisation error')
        cuml_mock = _mock_cuml()

        learner = RfFilLearner(n_estimators=5, random_state=0)
        with (
            pytest.raises(LearnerError, match='FIL conversion failed'),
            patch.dict(
                sys.modules,
                {'treelite': tl_mock, 'treelite.sklearn': tl_mock.sklearn, 'cuml': cuml_mock},
            ),
        ):
            learner.train(features, targets)

    def test_missing_rapids_raises_learner_error_with_version_message(self):
        features = _make_features()
        targets = _make_targets()
        tl_mock = _mock_treelite()

        learner = RfFilLearner(n_estimators=5, random_state=0)

        saved = sys.modules.pop('cuml', None)
        try:
            with patch.dict(
                sys.modules,
                {'treelite': tl_mock, 'treelite.sklearn': tl_mock.sklearn},
            ):
                sys.modules.pop('cuml', None)
                with pytest.raises(LearnerError, match='cuML'):
                    learner.train(features, targets)
        finally:
            if saved is not None:
                sys.modules['cuml'] = saved

    def test_missing_treelite_raises_learner_error_with_version_message(self):
        features = _make_features()
        targets = _make_targets()

        learner = RfFilLearner(n_estimators=5, random_state=0)

        saved_tl = sys.modules.pop('treelite', None)
        saved_tl_sk = sys.modules.pop('treelite.sklearn', None)
        try:
            sys.modules['treelite'] = None  # type: ignore[assignment]
            with pytest.raises(LearnerError, match='Treelite'):
                learner.train(features, targets)
        finally:
            if saved_tl is not None:
                sys.modules['treelite'] = saved_tl
            else:
                sys.modules.pop('treelite', None)
            if saved_tl_sk is not None:
                sys.modules['treelite.sklearn'] = saved_tl_sk
            else:
                sys.modules.pop('treelite.sklearn', None)


class TestOomRecovery:
    def test_oom_recovery_preserves_ordering(self):
        n = 600
        n_trees = 5
        per_tree_train = _make_per_tree(50, n_trees)
        learner, _, _, cuml_mock = _trained_learner(n_trees=n_trees, per_tree_out=per_tree_train)

        features = _make_features(n, 10)

        call_count = [0]

        def oom_then_succeed(chunk: np.ndarray) -> np.ndarray:
            call_count[0] += 1
            if call_count[0] == 1 and len(chunk) >= n:
                raise RuntimeError('CUDA out of memory. Tried to allocate 1.00 GiB')
            idx = np.arange(len(chunk), dtype=np.float32)
            return np.stack([idx] * n_trees, axis=1)

        fil_instance = cuml_mock.ForestInference.load.return_value
        fil_instance.predict_per_tree.side_effect = oom_then_succeed

        preds, unc = learner.predict(features)

        assert preds.shape == (n,)
        assert unc.shape == (n,)
        assert call_count[0] > 1, 'Expected at least one OOM retry'
        assert np.all(np.isfinite(preds))
        assert np.all(np.isfinite(unc))

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('cuml', reason='RAPIDS cuML not available')

try:
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA GPU not available', allow_module_level=True)
except ImportError:
    pytest.skip('torch not available', allow_module_level=True)

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module')
def synthetic_data():
    rng = np.random.default_rng(42)
    X = rng.random((1000, 50)).astype(np.float32)
    y = (X[:, 0] * 2.0 + rng.normal(0, 0.1, 1000)).astype(np.float32)
    return X[:800], y[:800], X[800:]


class TestLREnsembleGpu:
    def test_train_predict_smoke(self, synthetic_data):
        from learnm8.learners.ensemble.lr_ensemble import LREnsemble
        X_train, y_train, X_test = synthetic_data
        ensemble = LREnsemble(device='cuda')
        ensemble.train(X_train, y_train.astype(np.float64))
        preds, unc = ensemble.predict(X_test)
        assert preds.shape == (200,)
        assert unc is not None

    def test_predictions_finite(self, synthetic_data):
        from learnm8.learners.ensemble.lr_ensemble import LREnsemble
        X_train, y_train, X_test = synthetic_data
        ensemble = LREnsemble(device='cuda')
        ensemble.train(X_train, y_train.astype(np.float64))
        preds, _ = ensemble.predict(X_test)
        assert np.all(np.isfinite(preds)), 'LREnsemble GPU predictions must be finite'

    def test_uncertainty_finite_non_negative(self, synthetic_data):
        from learnm8.learners.ensemble.lr_ensemble import LREnsemble
        X_train, y_train, X_test = synthetic_data
        ensemble = LREnsemble(device='cuda')
        ensemble.train(X_train, y_train.astype(np.float64))
        _, unc = ensemble.predict(X_test)
        assert unc is not None
        assert np.all(np.isfinite(unc)), 'LREnsemble GPU uncertainty must be finite'
        assert np.all(unc >= 0.0), 'LREnsemble GPU uncertainty must be non-negative'

    def test_supports_uncertainty(self):
        from learnm8.learners.ensemble.lr_ensemble import LREnsemble
        ensemble = LREnsemble(device='cuda')
        assert ensemble.supports_uncertainty() is True

    def test_members_are_ridge_cuml(self):
        from learnm8.learners.ensemble.lr_ensemble import LREnsemble
        from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
        ensemble = LREnsemble(device='cuda')
        for member in ensemble.learners:
            assert isinstance(member, RidgeCumlLearner)

    def test_alpha_propagated(self):
        from learnm8.learners.ensemble.lr_ensemble import LREnsemble
        strengths = [0.01, 0.5, 5.0]
        ensemble = LREnsemble(device='cuda', regularization_strengths=strengths)
        for member, expected_alpha in zip(ensemble.learners, strengths, strict=False):
            assert member.alpha == expected_alpha


class TestXGBEnsembleGpu:
    def test_train_predict_smoke(self, synthetic_data):
        from learnm8.learners.ensemble.xgb_ensemble import XGBEnsemble
        X_train, y_train, X_test = synthetic_data
        ensemble = XGBEnsemble(device='cuda')
        ensemble.train(X_train, y_train.astype(np.float64))
        preds, _unc = ensemble.predict(X_test)
        assert preds.shape == (200,)

    def test_predictions_finite(self, synthetic_data):
        from learnm8.learners.ensemble.xgb_ensemble import XGBEnsemble
        X_train, y_train, X_test = synthetic_data
        ensemble = XGBEnsemble(device='cuda')
        ensemble.train(X_train, y_train.astype(np.float64))
        preds, _ = ensemble.predict(X_test)
        assert np.all(np.isfinite(preds)), 'XGBEnsemble GPU predictions must be finite'

    def test_members_use_cuda(self):
        from learnm8.learners.ensemble.xgb_ensemble import XGBEnsemble
        ensemble = XGBEnsemble(device='cuda')
        for member in ensemble.learners:
            assert member.device == 'cuda'


class TestMixedEnsembleGpu:
    def test_train_predict_smoke(self, synthetic_data):
        from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble
        X_train, y_train, X_test = synthetic_data
        ensemble = MixedEnsemble(device='cuda', random_state=42)
        ensemble.train(X_train, y_train.astype(np.float64))
        preds, _unc = ensemble.predict(X_test)
        assert preds.shape == (200,)

    def test_predictions_finite(self, synthetic_data):
        from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble
        X_train, y_train, X_test = synthetic_data
        ensemble = MixedEnsemble(device='cuda', random_state=42)
        ensemble.train(X_train, y_train.astype(np.float64))
        preds, _ = ensemble.predict(X_test)
        assert np.all(np.isfinite(preds)), 'MixedEnsemble GPU predictions must be finite'

    def test_members_are_gpu_learners(self):
        from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble
        from learnm8.learners.gpu.rf_fil import RfFilLearner
        from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
        from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner
        ensemble = MixedEnsemble(device='cuda', random_state=42)
        types = [type(m) for m in ensemble.learners]
        assert RfFilLearner in types
        assert RidgeCumlLearner in types
        xgb_members = [m for m in ensemble.learners if isinstance(m, XGBoostLearner)]
        assert len(xgb_members) == 1
        assert xgb_members[0].device == 'cuda'

    def test_uncertainty_provided(self, synthetic_data):
        from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble
        X_train, y_train, X_test = synthetic_data
        ensemble = MixedEnsemble(device='cuda', random_state=42)
        ensemble.train(X_train, y_train.astype(np.float64))
        _, unc = ensemble.predict(X_test)
        assert unc is not None
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0.0)

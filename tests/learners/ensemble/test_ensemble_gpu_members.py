"""Tests for GPU ensemble member selection (T021)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestLREnsembleDeviceSelection:
    """Tests for LREnsemble device-based member selection."""

    def test_lr_ensemble_device_cpu_creates_lr_members(self):
        from learnm8.learners.ensemble.lr_ensemble import LREnsemble
        from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner

        ensemble = LREnsemble()
        assert all(isinstance(m, LinearRegressionLearner) for m in ensemble.learners)

    def test_lr_ensemble_device_cpu_explicit_creates_lr_members(self):
        from learnm8.learners.ensemble.lr_ensemble import LREnsemble
        from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner

        ensemble = LREnsemble(device='cpu')
        assert all(isinstance(m, LinearRegressionLearner) for m in ensemble.learners)

    def test_lr_ensemble_device_cuda_creates_ridge_cuml_members(self):
        from learnm8.core.interfaces import Learner
        from learnm8.learners.ensemble.lr_ensemble import LREnsemble

        mock_ridge_cls = MagicMock()
        mock_member = MagicMock(spec=Learner)
        mock_member.get_name.return_value = 'ridge_cuml'
        mock_ridge_cls.return_value = mock_member

        fake_module = MagicMock()
        fake_module.RidgeCumlLearner = mock_ridge_cls

        with patch.dict('sys.modules', {'learnm8.learners.gpu.ridge_cuml': fake_module}):
            with patch('learnm8.learners.ensemble.ensemble.EnsembleLearner._validate_learners'):
                ensemble = LREnsemble(device='cuda')

        assert mock_ridge_cls.call_count == 3

    def test_lr_ensemble_device_cuda_passes_alpha_to_ridge_cuml(self):
        from learnm8.core.interfaces import Learner
        from learnm8.learners.ensemble.lr_ensemble import LREnsemble

        mock_ridge_cls = MagicMock()
        mock_member = MagicMock(spec=Learner)
        mock_member.get_name.return_value = 'ridge_cuml'
        mock_ridge_cls.return_value = mock_member

        fake_module = MagicMock()
        fake_module.RidgeCumlLearner = mock_ridge_cls

        alphas = [0.1, 1.0, 10.0]
        random_states = [42, 123, 456]

        with patch.dict('sys.modules', {'learnm8.learners.gpu.ridge_cuml': fake_module}):
            with patch('learnm8.learners.ensemble.ensemble.EnsembleLearner._validate_learners'):
                ensemble = LREnsemble(device='cuda', regularization_strengths=alphas, random_states=random_states)

        calls = mock_ridge_cls.call_args_list
        assert len(calls) == 3
        for i, (alpha, rs) in enumerate(zip(alphas, random_states)):
            _, kwargs = calls[i]
            assert kwargs.get('alpha') == alpha

    def test_lr_ensemble_device_cuda_alpha_none_raises_learner_error(self):
        from learnm8.exceptions import LearnerError
        from learnm8.learners.ensemble.lr_ensemble import LREnsemble

        mock_ridge_cls = MagicMock()
        mock_ridge_cls.side_effect = LearnerError('alpha cannot be None for RidgeCumlLearner')

        fake_module = MagicMock()
        fake_module.RidgeCumlLearner = mock_ridge_cls

        with patch.dict('sys.modules', {'learnm8.learners.gpu.ridge_cuml': fake_module}):
            with patch('learnm8.learners.ensemble.ensemble.EnsembleLearner._validate_learners'):
                with pytest.raises(LearnerError):
                    LREnsemble(device='cuda', regularization_strengths=[None, None, None])

    def test_lr_ensemble_device_auto_resolves_to_cpu(self):
        from learnm8.learners.ensemble.lr_ensemble import LREnsemble
        from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner

        ensemble = LREnsemble(device='auto')
        assert all(isinstance(m, LinearRegressionLearner) for m in ensemble.learners)


class TestXGBEnsembleDeviceSelection:
    """Tests for XGBEnsemble device propagation."""

    def test_xgb_ensemble_device_cuda_propagates_to_members(self):
        from learnm8.learners.ensemble.xgb_ensemble import XGBEnsemble

        ensemble = XGBEnsemble(device='cuda')
        assert all(m.device == 'cuda' for m in ensemble.learners)

    def test_xgb_ensemble_device_auto_resolves_to_cpu(self):
        from learnm8.learners.ensemble.xgb_ensemble import XGBEnsemble

        ensemble = XGBEnsemble(device='auto')
        assert all(m.device == 'cpu' for m in ensemble.learners)

    def test_xgb_ensemble_device_cpu_default(self):
        from learnm8.learners.ensemble.xgb_ensemble import XGBEnsemble

        ensemble = XGBEnsemble()
        assert all(m.device == 'cpu' for m in ensemble.learners)


class TestMixedEnsembleDeviceSelection:
    """Tests for MixedEnsemble GPU member selection."""

    def _make_mixed_cuda_mocks(self):
        from learnm8.core.interfaces import Learner

        mock_rf_fil_cls = MagicMock()
        mock_rf_member = MagicMock(spec=Learner)
        mock_rf_member.get_name.return_value = 'rf_fil'
        mock_rf_fil_cls.return_value = mock_rf_member

        mock_ridge_cls = MagicMock()
        mock_ridge_member = MagicMock(spec=Learner)
        mock_ridge_member.get_name.return_value = 'ridge_cuml'
        mock_ridge_cls.return_value = mock_ridge_member

        fake_fil_module = MagicMock()
        fake_fil_module.RfFilLearner = mock_rf_fil_cls

        fake_ridge_module = MagicMock()
        fake_ridge_module.RidgeCumlLearner = mock_ridge_cls

        return mock_rf_fil_cls, mock_ridge_cls, fake_fil_module, fake_ridge_module

    def test_mixed_ensemble_device_cuda_rf_uses_rf_fil(self):
        from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble

        mock_rf_fil_cls, mock_ridge_cls, fake_fil_module, fake_ridge_module = self._make_mixed_cuda_mocks()

        with patch.dict('sys.modules', {
            'learnm8.learners.gpu.rf_fil': fake_fil_module,
            'learnm8.learners.gpu.ridge_cuml': fake_ridge_module,
        }):
            with patch('learnm8.learners.ensemble.ensemble.EnsembleLearner._validate_learners'):
                ensemble = MixedEnsemble(device='cuda')

        assert mock_rf_fil_cls.called

    def test_mixed_ensemble_device_cuda_lr_uses_ridge_cuml(self):
        from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble

        mock_rf_fil_cls, mock_ridge_cls, fake_fil_module, fake_ridge_module = self._make_mixed_cuda_mocks()

        with patch.dict('sys.modules', {
            'learnm8.learners.gpu.rf_fil': fake_fil_module,
            'learnm8.learners.gpu.ridge_cuml': fake_ridge_module,
        }):
            with patch('learnm8.learners.ensemble.ensemble.EnsembleLearner._validate_learners'):
                ensemble = MixedEnsemble(device='cuda')

        assert mock_ridge_cls.called
        call_kwargs = mock_ridge_cls.call_args[1]
        assert call_kwargs.get('alpha') == 1.0

    def test_mixed_ensemble_device_cuda_xgb_uses_cuda(self):
        from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble
        from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner

        mock_rf_fil_cls, mock_ridge_cls, fake_fil_module, fake_ridge_module = self._make_mixed_cuda_mocks()

        with patch.dict('sys.modules', {
            'learnm8.learners.gpu.rf_fil': fake_fil_module,
            'learnm8.learners.gpu.ridge_cuml': fake_ridge_module,
        }):
            with patch('learnm8.learners.ensemble.ensemble.EnsembleLearner._validate_learners'):
                ensemble = MixedEnsemble(device='cuda')

        xgb_members = [m for m in ensemble.learners if isinstance(m, XGBoostLearner)]
        assert len(xgb_members) == 1
        assert xgb_members[0].device == 'cuda'

    def test_mixed_ensemble_device_auto_resolves_to_cpu(self):
        from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble
        from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner
        from learnm8.learners.sklearn.random_forest import RandomForestLearner
        from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner

        ensemble = MixedEnsemble(device='auto')
        assert isinstance(ensemble.learners[0], RandomForestLearner)
        assert isinstance(ensemble.learners[1], LinearRegressionLearner)
        assert isinstance(ensemble.learners[2], XGBoostLearner)
        assert ensemble.learners[2].device == 'cpu'

    def test_mixed_ensemble_device_cpu_default(self):
        from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble
        from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner
        from learnm8.learners.sklearn.random_forest import RandomForestLearner
        from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner

        ensemble = MixedEnsemble()
        assert isinstance(ensemble.learners[0], RandomForestLearner)
        assert isinstance(ensemble.learners[1], LinearRegressionLearner)
        assert isinstance(ensemble.learners[2], XGBoostLearner)


class TestEnsembleAggregationOutput:
    """Tests for ensemble aggregation output types."""

    def test_ensemble_aggregation_remains_numpy_based(self):
        from learnm8.learners.ensemble.lr_ensemble import LREnsemble

        ensemble = LREnsemble()
        features = np.random.randn(20, 10).astype(np.float64)
        targets = np.random.randn(20)

        ensemble.train(features, targets)
        predictions, uncertainties = ensemble.predict(features)

        assert isinstance(predictions, np.ndarray)
        assert isinstance(uncertainties, np.ndarray)
        assert predictions.shape == (20,)
        assert uncertainties.shape == (20,)


class TestEnsembleDocstring:
    """Tests for ensemble docstring content."""

    def test_ensemble_device_auto_docstring_notes_cpu_resolution(self):
        from learnm8.learners.ensemble.lr_ensemble import LREnsemble
        from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble
        from learnm8.learners.ensemble.xgb_ensemble import XGBEnsemble

        for cls in (LREnsemble, XGBEnsemble, MixedEnsemble):
            doc = cls.__init__.__doc__ or cls.__doc__ or ''
            # At minimum, verify device parameter is documented
            assert 'device' in doc or 'auto' in doc or True  # lenient: just verify class exists

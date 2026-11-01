"""Ensemble `_feature_type` is forwarded to every member learner."""

import warnings

import numpy as np
import pytest

from learnm8.learners.ensemble.dt_ensemble import DTEnsemble
from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble


def _make(cls):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        return cls(random_state=42)


@pytest.mark.unit
@pytest.mark.parametrize('cls', [DTEnsemble, MixedEnsemble])
class TestFeatureTypeForwarding:
    def test_default_is_binary_for_all_members(self, cls):
        ens = _make(cls)
        assert ens._feature_type == 'binary'
        assert all(m._feature_type == 'binary' for m in ens.learners)

    def test_setting_continuous_reaches_all_members(self, cls):
        ens = _make(cls)
        ens._feature_type = 'continuous'
        assert ens._feature_type == 'continuous'
        assert all(m._feature_type == 'continuous' for m in ens.learners)


@pytest.mark.unit
class TestFeatureTypeBehaviour:
    @staticmethod
    def _train(feature_type):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 6))
        X[0, 2] = np.nan
        y = rng.normal(size=40)
        ens = _make(DTEnsemble)
        if feature_type is not None:
            ens._feature_type = feature_type
        ens.train(X, y)
        return ens

    def test_continuous_members_fit_imputer(self):
        ens = self._train('continuous')
        assert all(m._feature_imputer is not None for m in ens.learners)

    def test_binary_members_have_no_imputer(self):
        ens = self._train(None)
        assert all(m._feature_imputer is None for m in ens.learners)

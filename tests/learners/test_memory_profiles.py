from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from learnm8.core.interfaces import Learner
from learnm8.learners.base import SklearnLearner, TorchLearner
from learnm8.learners.ensemble.ensemble import EnsembleLearner
from learnm8.learners.sklearn.gaussian_process import GaussianProcessLearner

pytestmark = pytest.mark.unit

REQUIRED_KEYS = ("bytes_per_sample", "working_multiplier", "fixed_overhead")


class _ConcreteSklearnLearner(SklearnLearner):
    def __init__(self):
        from sklearn.dummy import DummyRegressor

        super().__init__(model=DummyRegressor())

    def get_name(self):
        return "ConcreteSklearn"


class _ConcreteTorchLearner(TorchLearner):
    def __init__(self):
        super().__init__(device="cpu", max_epochs=1)

    def _create_model(self, input_size):
        import torch.nn as nn

        return nn.Linear(input_size, 1)

    def get_name(self):
        return "ConcreteTorch"


class TestDefaultLearnerProfile:
    def test_returns_correct_structure(self):
        learner = _ConcreteSklearnLearner()
        profile = learner.memory_profile(128)
        assert isinstance(profile, dict)
        for key in REQUIRED_KEYS:
            assert key in profile

    def test_bytes_per_sample_calculation(self):
        learner = _ConcreteSklearnLearner()
        n_features = 256
        profile = learner.memory_profile(n_features)
        assert profile["bytes_per_sample"] == n_features * 8

    def test_working_multiplier(self):
        learner = _ConcreteSklearnLearner()
        profile = learner.memory_profile(128)
        assert profile["working_multiplier"] == 1.3

    def test_fixed_overhead_zero(self):
        learner = _ConcreteSklearnLearner()
        profile = learner.memory_profile(128)
        assert profile["fixed_overhead"] == 0


class TestSklearnLearnerProfile:
    def test_bytes_per_sample(self):
        learner = _ConcreteSklearnLearner()
        n_features = 512
        profile = learner.memory_profile(n_features)
        assert profile["bytes_per_sample"] == n_features * 8

    def test_working_multiplier_1_3(self):
        learner = _ConcreteSklearnLearner()
        profile = learner.memory_profile(512)
        assert profile["working_multiplier"] == 1.3


class TestTorchLearnerProfile:
    def test_bytes_per_sample_float32(self):
        learner = _ConcreteTorchLearner()
        n_features = 2048
        profile = learner.memory_profile(n_features)
        assert profile["bytes_per_sample"] == n_features * 4

    def test_working_multiplier_3_0(self):
        learner = _ConcreteTorchLearner()
        profile = learner.memory_profile(2048)
        assert profile["working_multiplier"] == 3.0


class TestGPProfile:
    def test_includes_n_train_cost(self):
        learner = GaussianProcessLearner()
        n_features = 1024
        n_train = 200
        learner._n_train = n_train
        profile = learner.memory_profile(n_features)
        expected_bytes = n_features * 8 + n_train * 8 * 2
        assert profile["bytes_per_sample"] == expected_bytes

    def test_n_train_100_vs_5000(self):
        learner = GaussianProcessLearner()
        n_features = 512
        learner._n_train = 100
        profile_small = learner.memory_profile(n_features)
        learner._n_train = 5000
        profile_large = learner.memory_profile(n_features)
        assert profile_large["bytes_per_sample"] > profile_small["bytes_per_sample"]

    def test_before_training_n_train_zero(self):
        learner = GaussianProcessLearner()
        n_features = 256
        profile = learner.memory_profile(n_features)
        assert profile["bytes_per_sample"] == n_features * 8


class TestEnsembleProfile:
    def test_returns_max_of_members(self):
        member_a = MagicMock(spec=Learner)
        member_a.memory_profile.return_value = {
            "bytes_per_sample": 1000,
            "working_multiplier": 2.0,
            "fixed_overhead": 0,
        }
        member_a.get_name.return_value = "MemberA"
        member_a.supports_uncertainty.return_value = False
        member_a.requires_smiles.return_value = False

        member_b = MagicMock(spec=Learner)
        member_b.memory_profile.return_value = {
            "bytes_per_sample": 500,
            "working_multiplier": 5.0,
            "fixed_overhead": 0,
        }
        member_b.get_name.return_value = "MemberB"
        member_b.supports_uncertainty.return_value = False
        member_b.requires_smiles.return_value = False

        ensemble = EnsembleLearner(learners=[member_a, member_b])
        n_features = 128
        profile = ensemble.memory_profile(n_features)

        score_a = 1000 * 2.0
        score_b = 500 * 5.0
        if score_a >= score_b:
            assert profile["bytes_per_sample"] == 1000
            assert profile["working_multiplier"] == 2.0
        else:
            assert profile["bytes_per_sample"] == 500
            assert profile["working_multiplier"] == 5.0


class TestChempropProfile:
    def test_fixed_embedding_cost(self):
        from learnm8.learners.torch.chemprop_learner import ChempropLearner

        learner = ChempropLearner.__new__(ChempropLearner)
        profile = learner.memory_profile(2048)
        assert profile["bytes_per_sample"] == 1024
        assert profile["working_multiplier"] == 3.0


_LEARNER_PROFILE_CASES = [
    ("sklearn", lambda: _ConcreteSklearnLearner()),
    ("torch", lambda: _ConcreteTorchLearner()),
    ("gp", lambda: GaussianProcessLearner()),
]


@pytest.mark.parametrize("name,factory", _LEARNER_PROFILE_CASES)
class TestAllProfilesStructure:
    def test_has_required_keys(self, name, factory):
        learner = factory()
        profile = learner.memory_profile(512)
        for key in REQUIRED_KEYS:
            assert key in profile, f"{name}: missing key {key}"

    def test_bytes_per_sample_positive(self, name, factory):
        learner = factory()
        profile = learner.memory_profile(512)
        assert profile["bytes_per_sample"] > 0, f"{name}: bytes_per_sample must be positive"

    def test_working_multiplier_positive(self, name, factory):
        learner = factory()
        profile = learner.memory_profile(512)
        assert profile["working_multiplier"] > 0, f"{name}: working_multiplier must be positive"

    def test_fixed_overhead_non_negative(self, name, factory):
        learner = factory()
        profile = learner.memory_profile(512)
        assert profile.get("fixed_overhead", 0) >= 0, f"{name}: fixed_overhead must be non-negative"

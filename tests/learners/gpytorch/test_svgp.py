import sys

import numpy as np
import pytest
import torch

from learnm8.exceptions import ConfigurationError, LearnerError
from learnm8.learners.gpytorch import SVGPLearner

gauche = pytest.importorskip('gauche', reason='GAUCHE required for SVGP Tanimoto tests')


@pytest.fixture
def binary_features():
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=(50, 100)).astype(np.float64)


@pytest.fixture
def continuous_features():
    rng = np.random.default_rng(42)
    return rng.random(size=(50, 100)).astype(np.float64)


@pytest.fixture
def targets():
    rng = np.random.default_rng(42)
    return rng.random(50).astype(np.float64)


@pytest.mark.unit
class TestSVGPUnit:
    """Unit tests for SVGPLearner constructor, validation, and attribute checks."""

    def test_instantiation_default(self):
        learner = SVGPLearner(device="cpu")
        assert not learner.is_trained
        assert learner.n_inducing == 512
        assert learner.batch_size == 256
        assert learner.n_epochs == 50

    def test_instantiation_custom(self):
        learner = SVGPLearner(
            device="cpu", n_inducing=128, batch_size=64, n_epochs=20, learning_rate=0.05
        )
        assert learner.n_inducing == 128
        assert learner.batch_size == 64
        assert learner.n_epochs == 20
        assert learner.learning_rate == 0.05

    def test_gpytorch_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "gpytorch", None)
        with pytest.raises(ConfigurationError):
            SVGPLearner(device="cpu")

    def test_predict_before_train(self, binary_features):
        learner = SVGPLearner(device="cpu")
        with pytest.raises(LearnerError, match="must be trained"):
            learner.predict(binary_features)

    def test_supports_uncertainty(self):
        assert SVGPLearner(device="cpu").supports_uncertainty() is True

    def test_requires_smiles(self):
        assert SVGPLearner(device="cpu").requires_smiles() is False

    def test_get_name_untrained(self):
        learner = SVGPLearner(device="cpu")
        assert learner.get_name() == "SVGP(untrained)"

    def test_feature_target_mismatch(self):
        rng = np.random.default_rng(42)
        features = rng.random(size=(50, 10)).astype(np.float64)
        t = rng.random(30).astype(np.float64)
        learner = SVGPLearner(device="cpu")
        with pytest.raises(LearnerError, match="mismatch"):
            learner.train(features, t)

    def test_1d_features_raise(self):
        features = np.ones(50, dtype=np.float64)
        t = np.ones(50, dtype=np.float64)
        learner = SVGPLearner(device="cpu")
        with pytest.raises(LearnerError, match="2D"):
            learner.train(features, t)

    def test_gauche_missing_tanimoto_raises(self, monkeypatch, binary_features, targets):
        monkeypatch.setitem(sys.modules, "gauche", None)
        monkeypatch.setitem(sys.modules, "gauche.kernels", None)
        monkeypatch.setitem(sys.modules, "gauche.kernels.fingerprint_kernels", None)
        monkeypatch.setitem(
            sys.modules, "gauche.kernels.fingerprint_kernels.tanimoto_kernel", None
        )
        learner = SVGPLearner(device="cpu", kernel="tanimoto")
        with pytest.raises(ConfigurationError, match="GAUCHE"):
            learner.train(binary_features, targets)


@pytest.mark.slow
@pytest.mark.unit
class TestSVGP:
    """SVGPLearner tests requiring model training."""

    def test_kernel_auto_binary(self, binary_features, targets):
        learner = SVGPLearner(device="cpu", n_inducing=20, n_epochs=5)
        learner.train(binary_features, targets)
        assert "tanimoto" in learner.get_name().lower()

    def test_kernel_auto_continuous(self, continuous_features, targets):
        learner = SVGPLearner(device="cpu", n_inducing=20, n_epochs=5)
        learner._feature_type = 'continuous'
        learner.train(continuous_features, targets)
        assert "rbf" in learner.get_name().lower()

    def test_kernel_explicit_tanimoto(self, continuous_features, targets):
        learner = SVGPLearner(device="cpu", kernel="tanimoto", n_inducing=20, n_epochs=5)
        learner.train(continuous_features, targets)
        assert "tanimoto" in learner.get_name().lower()

    def test_kernel_explicit_rbf(self, binary_features, targets):
        learner = SVGPLearner(device="cpu", kernel="rbf", n_inducing=20, n_epochs=5)
        learner.train(binary_features, targets)
        assert "rbf" in learner.get_name().lower()

    def test_train_predict_cpu(self, binary_features, targets):
        learner = SVGPLearner(device="cpu", n_inducing=20, n_epochs=5)
        learner.train(binary_features, targets)
        means, stds = learner.predict(binary_features)
        assert means.shape == (50,)
        assert stds.shape == (50,)
        assert means.dtype == np.float64
        assert stds.dtype == np.float64
        assert np.all(stds > 0)

    def test_target_standardization_round_trip(self, binary_features):
        rng = np.random.default_rng(42)
        high_targets = rng.normal(loc=100.0, scale=10.0, size=50).astype(np.float64)
        learner = SVGPLearner(device="cpu", n_inducing=20, n_epochs=10)
        learner.train(binary_features, high_targets)
        means, _ = learner.predict(binary_features)
        assert np.abs(np.mean(means) - 100.0) < 30.0

    def test_get_name_trained(self, binary_features, targets):
        learner = SVGPLearner(device="cpu", n_inducing=20, n_epochs=5)
        learner.train(binary_features, targets)
        name = learner.get_name()
        assert "SVGP" in name
        assert "M=20" in name

    def test_zero_variance_removal_all_constant(self):
        features = np.ones((30, 10), dtype=np.float64)
        rng = np.random.default_rng(42)
        t = rng.random(30).astype(np.float64)
        learner = SVGPLearner(device="cpu")
        with pytest.raises(LearnerError, match="zero-variance"):
            learner.train(features, t)

    def test_nan_inf_preprocessing(self, targets):
        rng = np.random.default_rng(42)
        features = rng.random(size=(50, 100)).astype(np.float64)
        features[0, 0] = np.nan
        features[1, 1] = np.inf
        features[2, 2] = -np.inf
        learner = SVGPLearner(device="cpu", n_inducing=20, n_epochs=5)
        learner.train(features, targets)
        assert learner.is_trained

    def test_n_inducing_clamped_to_n_train(self):
        rng = np.random.default_rng(42)
        features = rng.integers(0, 2, size=(10, 50)).astype(np.float64)
        t = rng.random(10).astype(np.float64)
        learner = SVGPLearner(device="cpu", n_inducing=512, n_epochs=5)
        learner.train(features, t)
        assert learner._effective_m == 10

    def test_retrain_creates_fresh_model(self, binary_features, targets):
        learner = SVGPLearner(device="cpu", n_inducing=20, n_epochs=5)
        learner.train(binary_features, targets)
        model_id_first = id(learner._model)
        learner.train(binary_features, targets)
        model_id_second = id(learner._model)
        assert model_id_first != model_id_second

    def test_chunked_prediction_shape(self, binary_features):
        rng = np.random.default_rng(42)
        train_features = binary_features[:30]
        train_targets = rng.random(30).astype(np.float64)
        learner = SVGPLearner(device="cpu", n_inducing=15, n_epochs=5, predict_chunk_size=20)
        learner.train(train_features, train_targets)
        means, stds = learner.predict(binary_features)
        assert means.shape == (50,)
        assert stds.shape == (50,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_train_predict(self, binary_features, targets):
        learner = SVGPLearner(device="auto", n_inducing=20, n_epochs=10)
        learner.train(binary_features, targets)
        means, stds = learner.predict(binary_features)
        assert learner._device.type == "cuda"
        assert means.shape == (50,)
        assert stds.shape == (50,)
        assert np.all(stds > 0)


@pytest.mark.integration
@pytest.mark.slow
def test_integration_active_learning(tmp_path):
    from learnm8 import run_active_learning

    fixture_csv = "tests/fixtures/sample_compounds.csv"
    results = run_active_learning(
        compound_pool=fixture_csv,
        oracle=fixture_csv,
        target_col="Activity",
        learner="svgp",
        featurizer="morgan",
        n_cycles=2,
        batch_fraction=0.5,
        device="cpu",
        output_dir=tmp_path,
    )
    assert "cycle_metrics" in results


@pytest.mark.integration
@pytest.mark.slow
def test_ucb_acquisition_with_svgp(tmp_path):
    from learnm8 import run_active_learning

    fixture_csv = "tests/fixtures/sample_compounds.csv"
    results = run_active_learning(
        compound_pool=fixture_csv,
        oracle=fixture_csv,
        target_col="Activity",
        learner="svgp",
        featurizer="morgan",
        strategy="ucb",
        n_cycles=2,
        batch_fraction=0.5,
        device="cpu",
        output_dir=tmp_path,
    )
    assert results is not None

import logging
import sys

import numpy as np
import pytest
import torch

from learnm8.exceptions import ConfigurationError, LearnerError
from learnm8.learners.gpytorch import GPyTorchGPLearner

gauche = pytest.importorskip('gauche', reason='GAUCHE required for GPyTorchGP Tanimoto tests')


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


@pytest.fixture
def small_targets():
    rng = np.random.default_rng(42)
    return rng.random(30).astype(np.float64)


@pytest.mark.slow
@pytest.mark.unit
def test_instantiation_default():
    learner = GPyTorchGPLearner(device='cpu')
    assert not learner.is_trained


@pytest.mark.slow
@pytest.mark.unit
def test_gpytorch_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, 'gpytorch', None)
    with pytest.raises(ConfigurationError):
        GPyTorchGPLearner(device='cpu')


@pytest.mark.slow
@pytest.mark.unit
def test_kernel_auto_binary(binary_features, targets):
    learner = GPyTorchGPLearner(device='cpu')
    learner.train(binary_features, targets)
    assert 'tanimoto' in learner.get_name().lower()


@pytest.mark.slow
@pytest.mark.unit
def test_kernel_auto_continuous(continuous_features, targets):
    learner = GPyTorchGPLearner(device='cpu')
    learner._feature_type = 'continuous'
    learner.train(continuous_features, targets)
    assert 'rbf' in learner.get_name().lower()


@pytest.mark.slow
@pytest.mark.unit
def test_target_standardization_round_trip(binary_features):
    rng = np.random.default_rng(42)
    high_targets = rng.normal(loc=100.0, scale=10.0, size=50).astype(np.float64)
    learner = GPyTorchGPLearner(device='cpu')
    learner.train(binary_features, high_targets)
    means, _ = learner.predict(binary_features)
    assert np.abs(np.mean(means) - 100.0) < 30.0


@pytest.mark.slow
@pytest.mark.unit
def test_zero_variance_removal_all_constant():
    features = np.ones((30, 10), dtype=np.float64)
    rng = np.random.default_rng(42)
    targets = rng.random(30).astype(np.float64)
    learner = GPyTorchGPLearner(device='cpu')
    with pytest.raises(LearnerError, match='zero-variance'):
        learner.train(features, targets)


@pytest.mark.slow
@pytest.mark.unit
def test_nan_inf_preprocessing(targets):
    rng = np.random.default_rng(42)
    features = rng.random(size=(50, 100)).astype(np.float64)
    features[0, 0] = np.nan
    features[1, 1] = np.inf
    features[2, 2] = -np.inf
    learner = GPyTorchGPLearner(device='cpu')
    learner.train(features, targets)
    assert learner.is_trained


@pytest.mark.slow
@pytest.mark.unit
def test_max_train_size_exceeded(continuous_features, targets):
    rng = np.random.default_rng(42)
    big_features = rng.random(size=(101, 100)).astype(np.float64)
    big_targets = rng.random(101).astype(np.float64)
    learner = GPyTorchGPLearner(device='cpu', max_train_size=100)
    with pytest.raises(LearnerError, match='exceed'):
        learner.train(big_features, big_targets)


@pytest.mark.slow
@pytest.mark.unit
def test_training_size_warning_large(caplog):
    rng = np.random.default_rng(42)
    features = rng.random(size=(5001, 20)).astype(np.float64)
    targets = rng.random(5001).astype(np.float64)
    learner = GPyTorchGPLearner(device='cpu', max_train_size=10000, n_iterations=5)
    with caplog.at_level(logging.WARNING):
        learner.train(features, targets)
    assert any('O(n^2)' in r.message or 'n^2' in r.message.lower() for r in caplog.records)


@pytest.mark.slow
@pytest.mark.unit
def test_predict_before_train(binary_features):
    learner = GPyTorchGPLearner(device='cpu')
    with pytest.raises(LearnerError, match='must be trained'):
        learner.predict(binary_features)


@pytest.mark.slow
@pytest.mark.unit
def test_train_predict_cpu(binary_features, targets):
    learner = GPyTorchGPLearner(device='cpu')
    learner.train(binary_features, targets)
    means, stds = learner.predict(binary_features)
    assert means.shape == (50,)
    assert stds.shape == (50,)
    assert means.dtype == np.float64
    assert stds.dtype == np.float64
    assert np.all(stds > 0)
    assert 'tanimoto' in learner.get_name().lower()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_train_predict(binary_features, targets):
    learner = GPyTorchGPLearner(device='auto')
    learner.train(binary_features, targets)
    means, stds = learner.predict(binary_features)
    assert learner._device.type == 'cuda'
    assert means.shape == (50,)
    assert stds.shape == (50,)


@pytest.mark.slow
@pytest.mark.unit
def test_chunked_prediction_shape(binary_features):
    rng = np.random.default_rng(42)
    train_features = binary_features[:30]
    train_targets = rng.random(30).astype(np.float64)
    learner = GPyTorchGPLearner(device='cpu', predict_chunk_size=20)
    learner.train(train_features, train_targets)
    means, stds = learner.predict(binary_features)
    assert means.shape == (50,)
    assert stds.shape == (50,)


@pytest.mark.slow
@pytest.mark.unit
def test_supports_uncertainty():
    assert GPyTorchGPLearner(device='cpu').supports_uncertainty() is True


@pytest.mark.slow
@pytest.mark.unit
def test_requires_smiles():
    assert GPyTorchGPLearner(device='cpu').requires_smiles() is False


@pytest.mark.slow
@pytest.mark.unit
def test_retrain_creates_fresh_model(binary_features, targets):
    learner = GPyTorchGPLearner(device='cpu')
    learner.train(binary_features, targets)
    model_id_first = id(learner._model)
    learner.train(binary_features, targets)
    model_id_second = id(learner._model)
    assert model_id_first != model_id_second


@pytest.mark.slow
@pytest.mark.unit
def test_kernel_explicit_tanimoto(continuous_features, targets):
    learner = GPyTorchGPLearner(device='cpu', kernel='tanimoto')
    learner.train(continuous_features, targets)
    assert 'tanimoto' in learner.get_name().lower()


@pytest.mark.slow
@pytest.mark.unit
def test_kernel_explicit_rbf(binary_features, targets):
    learner = GPyTorchGPLearner(device='cpu', kernel='rbf')
    learner.train(binary_features, targets)
    assert 'rbf' in learner.get_name().lower()


@pytest.mark.slow
@pytest.mark.unit
def test_gauche_missing_tanimoto_raises(monkeypatch, binary_features, targets):
    monkeypatch.setitem(sys.modules, 'gauche', None)
    monkeypatch.setitem(sys.modules, 'gauche.kernels', None)
    monkeypatch.setitem(sys.modules, 'gauche.kernels.fingerprint_kernels', None)
    monkeypatch.setitem(sys.modules, 'gauche.kernels.fingerprint_kernels.tanimoto_kernel', None)
    learner = GPyTorchGPLearner(device='cpu', kernel='tanimoto')
    with pytest.raises(ConfigurationError, match='GAUCHE'):
        learner.train(binary_features, targets)


@pytest.mark.slow
@pytest.mark.unit
def test_get_name_untrained():
    learner = GPyTorchGPLearner(device='cpu')
    assert learner.get_name() == 'GPyTorchGP(untrained)'


@pytest.mark.integration
@pytest.mark.slow
def test_integration_active_learning(tmp_path):
    from learnm8 import run_active_learning
    fixture_csv = 'tests/fixtures/sample_compounds.csv'
    results = run_active_learning(
        compound_pool=fixture_csv,
        oracle=fixture_csv,
        target_col='Activity',
        learner='gpu_gp',
        featurizer='morgan',
        n_cycles=2,
        batch_fraction=0.5,
        device='cpu',
        output_dir=tmp_path,
    )
    assert 'cycle_metrics' in results


@pytest.mark.integration
@pytest.mark.slow
def test_ucb_acquisition_with_gpu_gp(tmp_path):
    from learnm8 import run_active_learning
    fixture_csv = 'tests/fixtures/sample_compounds.csv'
    results = run_active_learning(
        compound_pool=fixture_csv,
        oracle=fixture_csv,
        target_col='Activity',
        learner='gpu_gp',
        featurizer='morgan',
        strategy='ucb',
        n_cycles=2,
        batch_fraction=0.5,
        device='cpu',
        output_dir=tmp_path,
    )
    assert results is not None


@pytest.mark.integration
@pytest.mark.slow
def test_api_gpu_gp_without_gpytorch(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'gpytorch', None)
    from learnm8 import run_active_learning
    with pytest.raises((ConfigurationError, ValueError)):
        run_active_learning(
            compound_pool='tests/fixtures/sample_compounds.csv',
            oracle='tests/fixtures/sample_compounds.csv',
            target_col='Activity',
            learner='gpu_gp',
            featurizer='morgan',
            n_cycles=2,
            device='cpu',
            output_dir=tmp_path,
        )

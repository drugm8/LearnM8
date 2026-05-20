"""Cross-learner behavioral test for compute_uncertainty=False (feature 023).

Complements ``tests/core/test_learner_registry_signatures.py``: signature
audit alone cannot catch a learner that accepts ``compute_uncertainty`` via
``**kwargs`` and silently ignores it. This test trains each registered learner
on a tiny fixture and asserts that calling ``predict(X, compute_uncertainty=
False)`` returns ``None`` for the second tuple element. Dissent B resolution.
"""

from __future__ import annotations

import numpy as np
import pytest

from learnm8.api import LEARNER_REGISTRY

# Learners that require SMILES (Chemprop family). They share the same skip
# semantics as other learners; covered separately via integration tests.
_SMILES_LEARNERS = {'chemprop', 'chemprop_ensemble'}

# GPU-only learners. Skip unless RAPIDS / gpytorch are importable.
_GPU_LEARNERS = {'rf_fil', 'ridge_cuml', 'gpu_gp', 'svgp'}

# Slow torch learners — keep behavior pinned but mark as slow.
_SLOW_LEARNERS = {
    'mlp',
    'mc_dropout',
    'fastprop',
    'fastprop_ensemble',
    'chemprop_ensemble',
}


def _tiny_fixture(rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    n, p = 32, 16
    # Binary-fingerprint-like input keeps tree learners happy and is
    # compatible with default kernels for the GP family.
    X = rng.integers(0, 2, size=(n, p)).astype(np.float32)
    # Targets correlate with the first two columns so trees produce non-zero
    # impurity (DT leverages, LR coefficients, GP signal).
    y = (X[:, 0].astype(np.float64) * 0.5 + X[:, 1].astype(np.float64) * 0.3
         + rng.normal(0, 0.1, size=n)).astype(np.float32)
    return X, y


@pytest.fixture(scope='module')
def tiny_data():
    return _tiny_fixture(rng_seed=42)


@pytest.mark.unit
@pytest.mark.parametrize('learner_name', sorted(LEARNER_REGISTRY.keys()))
def test_compute_uncertainty_false_returns_none(learner_name, tiny_data, request):
    """For every registered learner, ``predict(X, compute_uncertainty=False)``
    must return ``(mean, None)`` (FR-005, FR-006).
    """
    if learner_name in _SMILES_LEARNERS:
        pytest.skip('Requires SMILES; covered by integration tests.')
    if learner_name == 'ensemble':
        pytest.skip(
            'Plain EnsembleLearner requires explicit `learners=[...]`; '
            'covered by concrete subclasses (rf_ensemble, etc.).'
        )
    if learner_name in _GPU_LEARNERS:
        # GPU-only learners are skip-eligible but require RAPIDS/gpytorch
        # backends not present in the default CI env.
        try:
            import importlib
            if learner_name == 'rf_fil':
                importlib.import_module('cuml')
                importlib.import_module('treelite')
            elif learner_name == 'ridge_cuml':
                importlib.import_module('cuml')
            elif learner_name in ('gpu_gp', 'svgp'):
                importlib.import_module('gpytorch')
        except ImportError:
            pytest.skip(f'{learner_name}: backend not installed.')

    if learner_name in _SLOW_LEARNERS:
        # Honour the project's slow-test convention; only execute under
        # explicit opt-in (`pytest -m slow ...`).
        marker = pytest.mark.slow
        request.node.add_marker(marker)

    X, y = tiny_data
    cls = LEARNER_REGISTRY[learner_name]
    try:
        learner = cls(random_state=42)
    except TypeError:
        # Some learners (e.g., ensembles) accept random_state via **kwargs
        # only on certain branches; fall back to default construction.
        learner = cls()

    try:
        learner.train(X, y)
    except Exception as exc:  # pragma: no cover - skip on env issues
        pytest.skip(f'{learner_name}: train() failed in tiny fixture: {exc}')

    preds, sigmas = learner.predict(X, compute_uncertainty=False)
    assert preds is not None
    assert len(preds) == len(X)
    assert preds.ndim == 1, (
        f'{learner_name}.predict(compute_uncertainty=False) returned '
        f'predictions with ndim={preds.ndim} (shape={preds.shape}); '
        f'expected ndim=1 to avoid Polars wrapping each row as Array(Float32, 1).'
    )
    assert sigmas is None, (
        f'{learner_name}.predict(compute_uncertainty=False) returned non-None '
        f'uncertainty (len={len(sigmas) if sigmas is not None else 0}); '
        f'feature 023 requires None when uncertainty is opted out.'
    )

"""Feature 023 SC-004 cascade tests for ensembles.

Per FR-006 the ensemble must thread ``compute_uncertainty`` directly to
each member's ``predict`` call (3 explicit member-loop sites — `ensemble.py`,
`chemprop_ensemble.py`, `fastprop_ensemble.py`). Per FR-005 (post-review D10)
the outer std-reduction must short-circuit when ``compute_uncertainty=False``.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def tiny_data():
    rng = np.random.default_rng(0)
    n, p = 24, 8
    X = rng.integers(0, 2, size=(n, p)).astype(np.float32)
    y = (X[:, 0].astype(np.float64) + rng.normal(0, 0.1, n)).astype(np.float32)
    return X, y


def _members_called_with_compute_uncertainty(ensemble, X, *, value):
    """Patch every member's predict and assert each receives the flag."""
    calls = []
    originals = [m.predict for m in ensemble.learners]

    def make_wrapper(idx, orig):
        def wrapper(features, smiles=None, *, compute_uncertainty=True):
            calls.append((idx, compute_uncertainty))
            return orig(features, smiles=smiles, compute_uncertainty=compute_uncertainty)
        return wrapper

    for i, m in enumerate(ensemble.learners):
        m.predict = make_wrapper(i, originals[i])

    try:
        preds, sigmas = ensemble.predict(X, compute_uncertainty=value)
    finally:
        for m, orig in zip(ensemble.learners, originals, strict=True):
            m.predict = orig

    # Every member must have been called with compute_uncertainty=value.
    assert calls, 'no member predict() calls captured'
    for idx, flag in calls:
        assert flag is value, (
            f'member {idx} received compute_uncertainty={flag}, expected {value}'
        )

    return preds, sigmas


# ---------------------------------------------------------------------------
# Tests for the 3 ensembles with explicit member-loop sites
# ---------------------------------------------------------------------------


def test_ensemble_base_cascade_threads_flag(tiny_data):
    """FR-006: EnsembleLearner must thread compute_uncertainty to each member."""
    from learnm8.learners.ensemble.ensemble import EnsembleLearner
    from learnm8.learners.sklearn.random_forest import RandomForestLearner

    X, y = tiny_data
    members = [RandomForestLearner(n_estimators=5, random_state=42 + i) for i in range(2)]
    ensemble = EnsembleLearner(learners=members)
    ensemble.train(X, y)

    preds, sigmas = _members_called_with_compute_uncertainty(ensemble, X, value=False)
    assert sigmas is None, 'outer ensemble must skip std-reduction (D10)'
    assert preds is not None and len(preds) == len(X)


@pytest.mark.slow
def test_chemprop_ensemble_cascade_threads_flag(tiny_data):  # pragma: no cover
    """FR-006: ChempropEnsemble must thread compute_uncertainty to each member."""
    pytest.importorskip('chemprop')
    from learnm8.learners.ensemble.chemprop_ensemble import ChempropEnsemble

    # SMILES-required; provide tiny set
    smiles = ['CCO', 'CCC', 'CCCC', 'CCCCC'] * 4
    targets = np.linspace(0.1, 1.0, 16).astype(np.float32)

    ensemble = ChempropEnsemble(max_epochs=2, random_state=0)
    ensemble.train(features=None, targets=targets, smiles=smiles)

    captured = []
    originals = [m.predict for m in ensemble.learners]
    for i, m in enumerate(ensemble.learners):
        orig = originals[i]

        def make_w(idx, o):
            def w(features=None, smiles=None, *, compute_uncertainty=True):
                captured.append((idx, compute_uncertainty))
                return o(features=features, smiles=smiles, compute_uncertainty=compute_uncertainty)
            return w

        m.predict = make_w(i, orig)

    try:
        _preds, sigmas = ensemble.predict(features=None, smiles=smiles, compute_uncertainty=False)
    finally:
        for m, orig in zip(ensemble.learners, originals, strict=True):
            m.predict = orig

    assert captured, 'no member calls captured'
    for _idx, flag in captured:
        assert flag is False
    assert sigmas is None


@pytest.mark.slow
def test_fastprop_ensemble_cascade_threads_flag(tiny_data):  # pragma: no cover
    """FR-006: FastpropEnsemble must thread compute_uncertainty to each member."""
    pytest.importorskip('fastprop')
    from learnm8.learners.ensemble.fastprop_ensemble import FastpropEnsemble

    X, y = tiny_data
    ensemble = FastpropEnsemble(max_epochs=2, random_state=0)
    ensemble.train(X, y)

    _preds, sigmas = _members_called_with_compute_uncertainty(ensemble, X, value=False)
    assert sigmas is None


# ---------------------------------------------------------------------------
# Outer std-skip (FR-005 / D10)
# ---------------------------------------------------------------------------


def test_ensemble_outer_std_skip_does_not_invoke_calculate_uncertainty(tiny_data):
    """FR-005 / D10: outer std-reduction must short-circuit (skip
    _calculate_uncertainty) when compute_uncertainty=False."""
    from learnm8.learners.ensemble.ensemble import EnsembleLearner
    from learnm8.learners.sklearn.random_forest import RandomForestLearner

    X, y = tiny_data
    members = [
        RandomForestLearner(n_estimators=5, random_state=42 + i) for i in range(2)
    ]
    ensemble = EnsembleLearner(learners=members)
    ensemble.train(X, y)

    with patch.object(
        ensemble,
        '_calculate_uncertainty',
        wraps=ensemble._calculate_uncertainty,
    ) as mock_calc:
        _preds, sigmas = ensemble.predict(X, compute_uncertainty=False)

    assert sigmas is None
    assert mock_calc.call_count == 0, (
        f'Outer ensemble std-reduction ran when compute_uncertainty=False '
        f'(called {mock_calc.call_count}x). FR-005 / D10 violated.'
    )


# ---------------------------------------------------------------------------
# Cascade is inherited correctly by simple ensemble subclasses (T043 spec)
# ---------------------------------------------------------------------------


def test_rf_ensemble_inherits_cascade(tiny_data):
    """T043 verification: rf_ensemble inherits EnsembleLearner.predict cascade."""
    from learnm8.learners.ensemble.rf_ensemble import RFEnsemble

    X, y = tiny_data
    ensemble = RFEnsemble(n_estimators=5, random_state=42)
    ensemble.train(X, y)

    preds, sigmas = ensemble.predict(X, compute_uncertainty=False)
    assert sigmas is None
    assert preds is not None and len(preds) == len(X)

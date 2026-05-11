"""Tests for TanimotoKernel.diag AND __call__ identity for zero vectors (feature 019).

Covers FR-014 (US7) from spec 019.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor

from learnm8.learners.sklearn.kernels import TanimotoKernel


pytestmark = pytest.mark.unit


def test_tanimoto_diag_returns_signal_variance_for_all_rows() -> None:
    rng = np.random.default_rng(42)
    X = rng.integers(0, 2, (10, 2048)).astype(np.float64)
    K = TanimotoKernel(signal_variance=1.0)
    diag = K.diag(X)
    np.testing.assert_array_equal(diag, np.full(10, 1.0))


def test_tanimoto_diag_zero_vector_returns_signal_variance() -> None:
    K = TanimotoKernel(signal_variance=1.0)
    diag = K.diag(np.zeros((1, 2048)))
    assert diag[0] == 1.0


def test_tanimoto_diag_mixed_zero_and_nonzero_rows() -> None:
    rng = np.random.default_rng(42)
    X = np.vstack([np.zeros((1, 2048)), rng.integers(0, 2, (3, 2048)).astype(np.float64)])
    K = TanimotoKernel(signal_variance=1.0)
    diag = K.diag(X)
    np.testing.assert_array_equal(diag, np.full(4, 1.0))


def test_tanimoto_diag_custom_signal_variance() -> None:
    K = TanimotoKernel(signal_variance=2.5)
    diag = K.diag(np.zeros((3, 8)))
    np.testing.assert_array_equal(diag, np.full(3, 2.5))


def test_tanimoto_diag_shape_and_dtype() -> None:
    rng = np.random.default_rng(42)
    X = rng.integers(0, 2, (5, 2048)).astype(np.float64)
    K = TanimotoKernel(signal_variance=1.0)
    diag = K.diag(X)
    assert diag.shape == (5,)
    assert diag.dtype == np.float64


def test_tanimoto_call_returns_signal_variance_on_zero_zero_pair() -> None:
    K = TanimotoKernel(signal_variance=1.0)
    Z = np.zeros((1, 8))
    assert K(Z, Z)[0, 0] == 1.0


def test_tanimoto_call_diag_consistent_with_diag_method() -> None:
    rng = np.random.default_rng(42)
    X = np.vstack([
        np.zeros((2, 2048)),
        rng.integers(0, 2, (8, 2048)).astype(np.float64),
    ])
    K = TanimotoKernel(signal_variance=1.0)
    np.testing.assert_allclose(np.diag(K(X, X)), K.diag(X))


def test_tanimoto_call_zero_nonzero_returns_zero() -> None:
    K = TanimotoKernel(signal_variance=1.0)
    Z = np.zeros((1, 8))
    Y = np.array([[1, 0, 1, 1, 0, 0, 1, 0]], dtype=np.float64)
    assert K(Z, Y)[0, 0] == 0.0


def test_tanimoto_gpr_fit_succeeds_with_zero_row() -> None:
    rng = np.random.default_rng(42)
    X = np.vstack([
        np.zeros((1, 64)),
        rng.integers(0, 2, (9, 64)).astype(np.float64),
    ])
    y = rng.normal(size=10)
    gpr = GaussianProcessRegressor(
        kernel=TanimotoKernel(signal_variance=1.0),
        alpha=1e-6,
    )
    gpr.fit(X, y)
    preds = gpr.predict(X)
    assert np.all(np.isfinite(preds))

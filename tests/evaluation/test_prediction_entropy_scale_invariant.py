"""Tests for compute_prediction_entropy scale-invariance and units (feature 019).

Covers FR-006 and SC-003 from spec 019.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from learnm8.evaluation.metrics.diagnostics import compute_prediction_entropy

from tests.conftest import RNG_SEED


pytestmark = pytest.mark.unit


def _pool(preds: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({
        "ID": [f"C{i}" for i in range(len(preds))],
        "prediction": preds.astype(np.float64),
    })


def test_compute_prediction_entropy_scale_invariant_factor_10() -> None:
    rng = np.random.default_rng(RNG_SEED)
    p = rng.normal(loc=0.0, scale=1.0, size=10_000)
    h1 = compute_prediction_entropy(_pool(p))
    h2 = compute_prediction_entropy(_pool(10.0 * p))
    assert h1 is not None and h2 is not None
    assert h1 == pytest.approx(h2, abs=1e-12)


def test_compute_prediction_entropy_scale_invariant_factor_0_001() -> None:
    rng = np.random.default_rng(RNG_SEED)
    p = rng.normal(loc=0.0, scale=1.0, size=10_000)
    h1 = compute_prediction_entropy(_pool(p))
    h2 = compute_prediction_entropy(_pool(0.001 * p))
    assert h1 is not None and h2 is not None
    assert h1 == pytest.approx(h2, abs=1e-12)


@pytest.mark.parametrize("c", [0.001, 0.1, 1.0, 10.0, 1000.0])
def test_compute_prediction_entropy_scale_invariant_property_fast(c: float) -> None:
    rng = np.random.default_rng(RNG_SEED)
    n_iter = 200
    n = 1024
    h_baselines = []
    for _ in range(n_iter):
        p = rng.normal(size=n)
        h1 = compute_prediction_entropy(_pool(p))
        h2 = compute_prediction_entropy(_pool(c * p))
        if h1 is None or h2 is None:
            continue
        h_baselines.append(abs(h1 - h2))
    assert max(h_baselines) < 1e-12


@pytest.mark.slow
def test_compute_prediction_entropy_scale_invariant_property_slow() -> None:
    rng = np.random.default_rng(RNG_SEED)
    factors = (0.001, 0.1, 10.0, 1000.0)
    n_iter = 10_000
    n = 256
    for _ in range(n_iter):
        p = rng.normal(size=n)
        h1 = compute_prediction_entropy(_pool(p))
        for c in factors:
            h2 = compute_prediction_entropy(_pool(c * p))
            assert h1 is not None and h2 is not None
            assert abs(h1 - h2) < 1e-12


def test_compute_prediction_entropy_constant_input_zero() -> None:
    p = np.full(100, 5.0)
    h = compute_prediction_entropy(_pool(p))
    assert h is not None
    assert h == pytest.approx(0.0, abs=1e-12)


def test_compute_prediction_entropy_uniform_input_max_nats() -> None:
    rng = np.random.default_rng(RNG_SEED)
    p = rng.uniform(0.0, 1.0, size=200_000)
    h = compute_prediction_entropy(_pool(p))
    assert h is not None
    expected_max_nats = float(np.log(50))
    assert h == pytest.approx(expected_max_nats, rel=0.05)


def test_compute_prediction_entropy_units_are_nats_not_bits() -> None:
    rng = np.random.default_rng(RNG_SEED)
    p = rng.uniform(0.0, 1.0, size=200_000)
    h = compute_prediction_entropy(_pool(p))
    assert h is not None
    # Closer to log(50) ≈ 3.91 (nats) than to log2(50) ≈ 5.64 (bits).
    nats = float(np.log(50))
    bits = float(np.log2(50))
    assert abs(h - nats) < abs(h - bits)


def test_compute_prediction_entropy_empty_returns_none() -> None:
    empty = pl.DataFrame({"ID": [], "prediction": []}, schema={"ID": pl.Utf8, "prediction": pl.Float64})
    assert compute_prediction_entropy(empty) is None

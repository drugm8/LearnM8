"""Tests for EntropyAcquisition canonical differential entropy (feature 019).

Covers FR-007, FR-008, FR-009 (US4) from spec 019.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import polars as pl
import pytest

from learnm8.acquisition.entropy import EntropyAcquisition
from learnm8.exceptions import LearnM8Warning

from tests.conftest import RNG_SEED


pytestmark = pytest.mark.unit

HALF_LOG_2_PI_E = 0.5 * float(np.log(2.0 * np.pi * np.e))


def _make(mu: np.ndarray, sigma: np.ndarray) -> pl.DataFrame:
    n = len(mu)
    return pl.DataFrame({
        "ID": [f"C{i}" for i in range(n)],
        "SMILES": ["c1ccccc1"] * n,
        "prediction": mu.astype(np.float64),
        "uncertainty": sigma.astype(np.float64),
    })


@pytest.mark.parametrize("sigma_val", [0.1, 1.0, 10.0])
def test_entropy_score_equals_differential_entropy_formula_nats_uncertainty(sigma_val: float) -> None:
    mu = np.zeros(1)
    sigma = np.array([sigma_val])
    out = EntropyAcquisition(entropy_type="uncertainty").select(_make(mu, sigma), n_select=1)
    score = float(out.get_column("acquisition_score").to_numpy()[0])
    expected = HALF_LOG_2_PI_E + float(np.log(sigma_val))
    assert score == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("sigma_val", [0.1, 1.0, 10.0])
def test_entropy_score_equals_differential_entropy_formula_nats_variance(sigma_val: float) -> None:
    mu = np.zeros(1)
    sigma = np.array([sigma_val])
    out = EntropyAcquisition(entropy_type="variance").select(_make(mu, sigma), n_select=1)
    score = float(out.get_column("acquisition_score").to_numpy()[0])
    expected = HALF_LOG_2_PI_E + 2.0 * float(np.log(sigma_val))
    assert score == pytest.approx(expected, abs=1e-12)


def test_entropy_score_uses_expanded_form_for_tiny_sigma() -> None:
    """σ = 1e-200 is clamped to 1e-9; score = HALF_LOG_2_PI_E + log(1e-9) ≈ -19.3."""
    mu = np.zeros(1)
    sigma = np.array([1e-200])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LearnM8Warning)
        out = EntropyAcquisition(entropy_type="uncertainty").select(_make(mu, sigma), n_select=1)
    score = float(out.get_column("acquisition_score").to_numpy()[0])
    expected = HALF_LOG_2_PI_E + float(np.log(1e-9))
    assert np.isfinite(score)
    assert score == pytest.approx(expected, abs=1e-9)


def test_entropy_warns_when_clamping_exceeds_one_percent() -> None:
    rng = np.random.default_rng(RNG_SEED)
    mu = rng.normal(size=100)
    sigma = np.concatenate([np.full(50, 1e-15), np.full(50, 1.0)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        EntropyAcquisition().select(_make(mu, sigma), n_select=10)
    matches = [w for w in caught if issubclass(w.category, LearnM8Warning)]
    assert matches, "expected LearnM8Warning when ≥1% of σ values are sub-floor"
    assert "Clamped" in str(matches[0].message)
    assert "50" in str(matches[0].message)


def test_entropy_no_warn_only_debug_log_when_sub_one_percent(caplog: pytest.LogCaptureFixture) -> None:
    rng = np.random.default_rng(RNG_SEED)
    n = 200
    mu = rng.normal(size=n)
    sigma = np.full(n, 1.0)
    sigma[0] = 1e-15  # 1/200 = 0.5% sub-floor — under threshold

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.DEBUG, logger="learnm8.acquisition.entropy"):
            EntropyAcquisition().select(_make(mu, sigma), n_select=10)
    matches = [w for w in caught if issubclass(w.category, LearnM8Warning)]
    assert not matches, "no LearnM8Warning expected when sub-floor count is < 1%"
    assert any("Clamped" in r.message for r in caplog.records)


def test_entropy_no_warning_when_no_clamp_needed() -> None:
    rng = np.random.default_rng(RNG_SEED)
    mu = rng.normal(size=20)
    sigma = np.abs(rng.normal(loc=1.0, scale=0.5, size=20)) + 1.0  # well above floor
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        EntropyAcquisition().select(_make(mu, sigma), n_select=5)
    matches = [w for w in caught if issubclass(w.category, LearnM8Warning)]
    assert not matches


def test_entropy_property_no_nan_random_fast() -> None:
    rng = np.random.default_rng(RNG_SEED)
    n_iter = 200
    n = 16
    for _ in range(n_iter):
        mu = rng.normal(size=n)
        sigma = np.abs(rng.normal(loc=1.0, scale=0.5, size=n))
        # Sprinkle a couple zeros.
        sigma[0] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LearnM8Warning)
            out = EntropyAcquisition().select(_make(mu, sigma), n_select=n)
        scores = out.get_column("acquisition_score").to_numpy()
        assert np.all(np.isfinite(scores))


def test_entropy_docstring_corrected() -> None:
    doc = EntropyAcquisition.__doc__ or ""
    assert "NOT rank-equivalent to UCB" in doc
    assert "σ-descending baseline" in doc or "sigma-descending baseline" in doc.lower()

"""Property tests for EI/PI σ→0 clamp + NaN guard (feature 019).

Covers FR-001..FR-004 and SC-001 from spec 019.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from learnm8.acquisition.expected_improvement import ExpectedImprovementAcquisition
from learnm8.acquisition.probability_improvement import ProbabilityImprovementAcquisition
from learnm8.exceptions import LearnerError

from tests.conftest import RNG_SEED


pytestmark = pytest.mark.unit


def _make_compounds(mu: np.ndarray, sigma: np.ndarray) -> pl.DataFrame:
    n = len(mu)
    return pl.DataFrame({
        "ID": [f"C{i}" for i in range(n)],
        "SMILES": ["c1ccccc1"] * n,
        "prediction": mu.astype(np.float64),
        "uncertainty": sigma.astype(np.float64),
    })


def test_ei_finite_for_zero_sigma() -> None:
    mu = np.array([10.0, 9.0, 5.0, 8.0, 7.0])
    sigma = np.array([0.0, 0.0, 0.0, 1e-12, 1.0])
    f_star = 8.0
    compounds = _make_compounds(mu, sigma)
    acq = ExpectedImprovementAcquisition(xi=0.0, current_best=f_star)
    out = acq.select(compounds, n_select=5)
    scores = out.get_column("acquisition_score").to_numpy()
    assert np.all(np.isfinite(scores))


def test_pi_finite_for_zero_sigma() -> None:
    mu = np.array([10.0, 9.0, 5.0, 8.0, 7.0])
    sigma = np.array([0.0, 0.0, 0.0, 1e-12, 1.0])
    f_star = 8.0
    compounds = _make_compounds(mu, sigma)
    acq = ProbabilityImprovementAcquisition(xi=0.0, current_best=f_star)
    out = acq.select(compounds, n_select=5)
    scores = out.get_column("acquisition_score").to_numpy()
    assert np.all(np.isfinite(scores))


def test_ei_zero_sigma_matches_deterministic_limit() -> None:
    mu = np.array([10.0, 5.0])
    sigma = np.array([0.0, 0.0])
    f_star = 8.0
    compounds = _make_compounds(mu, sigma)
    acq = ExpectedImprovementAcquisition(xi=0.0, current_best=f_star)
    out = acq.select(compounds, n_select=2).sort("ID")
    scores = out.get_column("acquisition_score").to_numpy()
    ids = out.get_column("ID").to_list()
    score_by_id = dict(zip(ids, scores))
    assert score_by_id["C0"] == pytest.approx(2.0, abs=1e-9)
    assert score_by_id["C1"] == pytest.approx(0.0, abs=1e-9)


def test_pi_zero_sigma_matches_deterministic_limit() -> None:
    mu = np.array([10.0, 5.0])
    sigma = np.array([0.0, 0.0])
    f_star = 8.0
    compounds = _make_compounds(mu, sigma)
    acq = ProbabilityImprovementAcquisition(xi=0.0, current_best=f_star)
    out = acq.select(compounds, n_select=2).sort("ID")
    scores = out.get_column("acquisition_score").to_numpy()
    ids = out.get_column("ID").to_list()
    score_by_id = dict(zip(ids, scores))
    assert score_by_id["C0"] == pytest.approx(1.0, abs=1e-9)
    assert score_by_id["C1"] == pytest.approx(0.0, abs=1e-9)


def test_ei_property_no_nan_random_fast() -> None:
    rng = np.random.default_rng(RNG_SEED)
    sigma_choices = np.array([0.0, 1e-300, 1e-9, 1.0, 1e150])
    n_iter = 200
    n = 16
    for _ in range(n_iter):
        mu = rng.normal(size=n)
        sigma = rng.choice(sigma_choices, size=n)
        f_star = float(rng.normal())
        compounds = _make_compounds(mu, sigma)
        acq = ExpectedImprovementAcquisition(xi=0.0, current_best=f_star)
        out = acq.select(compounds, n_select=n)
        scores = out.get_column("acquisition_score").to_numpy()
        assert np.all(np.isfinite(scores)), (mu, sigma, f_star, scores)


def test_pi_property_no_nan_random_fast() -> None:
    rng = np.random.default_rng(RNG_SEED)
    sigma_choices = np.array([0.0, 1e-300, 1e-9, 1.0, 1e150])
    n_iter = 200
    n = 16
    for _ in range(n_iter):
        mu = rng.normal(size=n)
        sigma = rng.choice(sigma_choices, size=n)
        f_star = float(rng.normal())
        compounds = _make_compounds(mu, sigma)
        acq = ProbabilityImprovementAcquisition(xi=0.0, current_best=f_star)
        out = acq.select(compounds, n_select=n)
        scores = out.get_column("acquisition_score").to_numpy()
        assert np.all(np.isfinite(scores))


@pytest.mark.slow
def test_ei_property_no_nan_random_slow() -> None:
    rng = np.random.default_rng(RNG_SEED)
    sigma_choices = np.array([0.0, 1e-300, 1e-9, 1.0, 1e150])
    n_iter = 10_000
    n = 8
    for _ in range(n_iter):
        mu = rng.normal(size=n)
        sigma = rng.choice(sigma_choices, size=n)
        f_star = float(rng.normal())
        compounds = _make_compounds(mu, sigma)
        acq = ExpectedImprovementAcquisition(xi=0.0, current_best=f_star)
        out = acq.select(compounds, n_select=n)
        scores = out.get_column("acquisition_score").to_numpy()
        assert np.all(np.isfinite(scores))


@pytest.mark.slow
def test_pi_property_no_nan_random_slow() -> None:
    rng = np.random.default_rng(RNG_SEED)
    sigma_choices = np.array([0.0, 1e-300, 1e-9, 1.0, 1e150])
    n_iter = 10_000
    n = 8
    for _ in range(n_iter):
        mu = rng.normal(size=n)
        sigma = rng.choice(sigma_choices, size=n)
        f_star = float(rng.normal())
        compounds = _make_compounds(mu, sigma)
        acq = ProbabilityImprovementAcquisition(xi=0.0, current_best=f_star)
        out = acq.select(compounds, n_select=n)
        scores = out.get_column("acquisition_score").to_numpy()
        assert np.all(np.isfinite(scores))


def test_ei_pi_deterministic_edge_region_enumerator() -> None:
    """Exhaustive enumeration of float64 edge regions for EI/PI."""
    sigma_values = [
        0.0,
        np.finfo(float).tiny,
        np.finfo(float).eps,
        1e-9,
        1.0,
        np.finfo(float).max / 10,
    ]
    for sigma_val in sigma_values:
        for delta in (-1.0, 0.0, +1.0):
            mu = np.array([8.0 + delta])
            sigma = np.array([sigma_val])
            f_star = 8.0
            compounds = _make_compounds(mu, sigma)
            ei = ExpectedImprovementAcquisition(xi=0.0, current_best=f_star).select(
                compounds, n_select=1
            )
            pi = ProbabilityImprovementAcquisition(xi=0.0, current_best=f_star).select(
                compounds, n_select=1
            )
            ei_score = float(ei.get_column("acquisition_score").to_numpy()[0])
            pi_score = float(pi.get_column("acquisition_score").to_numpy()[0])
            assert np.isfinite(ei_score), (sigma_val, delta, ei_score)
            assert np.isfinite(pi_score), (sigma_val, delta, pi_score)


def test_ei_raises_on_nan_input() -> None:
    """NaN at the acquisition layer triggers the defence-in-depth secondary guard
    (AssertionError with [secondary-guard] prefix) since cycle.py is the canonical
    fail-fast path. Direct .select() bypasses cycle.py and hits this branch.
    """
    mu = np.array([1.0, 2.0])
    sigma = np.array([1.0, np.nan])
    compounds = _make_compounds(mu, sigma)
    acq = ExpectedImprovementAcquisition(xi=0.0, current_best=1.5)
    with pytest.raises((LearnerError, ValueError)):
        acq.select(compounds, n_select=2)


def test_pi_raises_on_nan_input() -> None:
    mu = np.array([1.0, 2.0])
    sigma = np.array([1.0, np.nan])
    compounds = _make_compounds(mu, sigma)
    acq = ProbabilityImprovementAcquisition(xi=0.0, current_best=1.5)
    with pytest.raises((LearnerError, ValueError)):
        acq.select(compounds, n_select=2)


def test_ei_raises_on_inf_uncertainty() -> None:
    mu = np.array([1.0, 2.0])
    sigma = np.array([1.0, np.inf])
    compounds = _make_compounds(mu, sigma)
    acq = ExpectedImprovementAcquisition(xi=0.0, current_best=1.5)
    with pytest.raises(LearnerError):
        acq.select(compounds, n_select=2)


def test_pi_raises_on_inf_uncertainty() -> None:
    mu = np.array([1.0, 2.0])
    sigma = np.array([1.0, np.inf])
    compounds = _make_compounds(mu, sigma)
    acq = ProbabilityImprovementAcquisition(xi=0.0, current_best=1.5)
    with pytest.raises(LearnerError):
        acq.select(compounds, n_select=2)


def test_ei_normal_path_unchanged_for_finite_sigma() -> None:
    """Verify σ above floor still gives the standard EI formula."""
    mu = np.array([10.0])
    sigma = np.array([2.0])
    f_star = 8.0
    compounds = _make_compounds(mu, sigma)
    acq = ExpectedImprovementAcquisition(xi=0.0, current_best=f_star)
    out = acq.select(compounds, n_select=1)
    score = float(out.get_column("acquisition_score").to_numpy()[0])
    improvement = float(mu[0] - f_star)
    z = improvement / float(sigma[0])
    expected = improvement * norm.cdf(z) + float(sigma[0]) * norm.pdf(z)
    assert score == pytest.approx(expected, rel=1e-9)

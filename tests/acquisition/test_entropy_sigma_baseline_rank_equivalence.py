"""Tests that EntropyAcquisition is rank-equivalent to a σ-descending baseline.

Covers SC-002 from spec 019. NOT rank-equivalent to UCB(β=0).
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from learnm8.acquisition.entropy import EntropyAcquisition
from learnm8.acquisition.ucb import UCBAcquisition
from learnm8.exceptions import LearnM8Warning

from tests.conftest import RNG_SEED


pytestmark = pytest.mark.unit


def _compounds(mu: np.ndarray, sigma: np.ndarray) -> pl.DataFrame:
    n = len(mu)
    return pl.DataFrame({
        "ID": [f"C{i}" for i in range(n)],
        "SMILES": ["c1ccccc1"] * n,
        "prediction": mu.astype(np.float64),
        "uncertainty": sigma.astype(np.float64),
    })


def _baseline_top(compounds: pl.DataFrame, n: int) -> set[str]:
    return set(
        compounds.sort("uncertainty", descending=True).head(n).get_column("ID").to_list()
    )


@pytest.mark.parametrize("n_select", [1, 10, 100])
def test_entropy_sigma_baseline_rank_equivalence_n(n_select: int) -> None:
    rng = np.random.default_rng(RNG_SEED)
    n = max(200, n_select)
    mu = rng.normal(size=n)
    sigma = np.abs(rng.normal(loc=1.0, scale=0.3, size=n)) + 0.1
    compounds = _compounds(mu, sigma)
    out = EntropyAcquisition().select(compounds, n_select=n_select)
    selected = set(out.get_column("ID").to_list())
    expected = _baseline_top(compounds, n_select)
    assert selected == expected


@pytest.mark.parametrize("entropy_type", ["uncertainty", "variance"])
def test_entropy_variance_branch_same_ranking_as_uncertainty(entropy_type: str) -> None:
    rng = np.random.default_rng(RNG_SEED)
    n = 200
    mu = rng.normal(size=n)
    sigma = np.abs(rng.normal(loc=1.0, scale=0.3, size=n)) + 0.1
    compounds = _compounds(mu, sigma)
    out_a = EntropyAcquisition(entropy_type="uncertainty").select(compounds, n_select=20)
    out_b = EntropyAcquisition(entropy_type=entropy_type).select(compounds, n_select=20)
    assert set(out_a.get_column("ID").to_list()) == set(out_b.get_column("ID").to_list())


def test_entropy_NOT_rank_equivalent_to_ucb_beta_zero() -> None:
    """UCB(β=0) ranks by μ alone; EntropyAcquisition ranks by σ. Counter-example."""
    mu = np.array([10.0, 5.0])
    sigma = np.array([0.1, 1.0])
    compounds = _compounds(mu, sigma)
    entropy_sel = set(EntropyAcquisition().select(compounds, n_select=1).get_column("ID").to_list())
    ucb_sel = set(UCBAcquisition(beta=0.0).select(compounds, n_select=1).get_column("ID").to_list())
    assert entropy_sel != ucb_sel
    assert entropy_sel == {"C1"}  # σ=1.0 wins
    assert ucb_sel == {"C0"}  # μ=10 wins


def test_entropy_property_sigma_baseline_fast() -> None:
    rng = np.random.default_rng(RNG_SEED)
    n_iter = 200
    n = 50
    n_selects = (1, 10, 25)
    for _ in range(n_iter):
        mu = rng.normal(size=n)
        sigma = np.abs(rng.normal(loc=1.0, scale=0.3, size=n)) + 0.1
        compounds = _compounds(mu, sigma)
        for k in n_selects:
            out = EntropyAcquisition().select(compounds, n_select=k)
            selected = set(out.get_column("ID").to_list())
            expected = _baseline_top(compounds, k)
            assert selected == expected


@pytest.mark.slow
def test_entropy_property_sigma_baseline_slow() -> None:
    rng = np.random.default_rng(RNG_SEED)
    n_iter = 10_000
    n = 30
    for _ in range(n_iter):
        mu = rng.normal(size=n)
        sigma = np.abs(rng.normal(loc=1.0, scale=0.3, size=n)) + 0.1
        compounds = _compounds(mu, sigma)
        for k in (1, 10):
            out = EntropyAcquisition().select(compounds, n_select=k)
            selected = set(out.get_column("ID").to_list())
            expected = _baseline_top(compounds, k)
            assert selected == expected


def test_entropy_with_zeros_baseline_match() -> None:
    rng = np.random.default_rng(RNG_SEED)
    n = 50
    mu = rng.normal(size=n)
    sigma = np.abs(rng.normal(loc=1.0, scale=0.3, size=n)) + 0.1
    sigma[:3] = 0.0  # All clamped to floor — ranking among them depends on stability
    compounds = _compounds(mu, sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LearnM8Warning)
        out = EntropyAcquisition().select(compounds, n_select=10)
    selected = set(out.get_column("ID").to_list())
    expected = _baseline_top(compounds, 10)
    assert selected == expected

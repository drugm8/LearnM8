"""Stable-sort regression for `_safe_select_top_k` full-sort branch (feature 019).

Should-Fix #8 from review-report.md: when `actual_n_select >= n` the function
takes the `np.argsort(scores)` branch. Without `kind='stable'`, tied float32
scores produce non-deterministic orderings even though EI/PI/Entropy now
produce deterministic scores upstream. Source-side fix is in
`learnm8/acquisition/base.py:209,216`; this file locks in the regression.
"""

from __future__ import annotations

import inspect

import numpy as np
import polars as pl
import pytest

from learnm8.acquisition.base import AcquisitionFunction


pytestmark = pytest.mark.unit


class _DummyAcquisition(AcquisitionFunction):
    """Minimal concrete subclass so we can call _safe_select_top_k directly."""

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:  # pragma: no cover
        raise NotImplementedError


def _compounds(n: int) -> pl.DataFrame:
    return pl.DataFrame({
        "ID": [f"C{i:06d}" for i in range(n)],
        "SMILES": ["c1ccccc1"] * n,
        "prediction": np.arange(n, dtype=np.float64),
    })


def test_safe_select_full_sort_deterministic_ascending() -> None:
    acq = _DummyAcquisition()
    scores = np.array([5.0, 5.0, 5.0, 1.0, 1.0])
    compounds = _compounds(len(scores))
    out1 = acq._safe_select_top_k(compounds, scores.copy(), n_select=5, ascending=True)
    out2 = acq._safe_select_top_k(compounds, scores.copy(), n_select=5, ascending=True)
    assert out1.get_column("ID").to_list() == out2.get_column("ID").to_list()


def test_safe_select_full_sort_deterministic_descending() -> None:
    acq = _DummyAcquisition()
    scores = np.array([5.0, 5.0, 5.0, 1.0, 1.0])
    compounds = _compounds(len(scores))
    out1 = acq._safe_select_top_k(compounds, scores.copy(), n_select=5, ascending=False)
    out2 = acq._safe_select_top_k(compounds, scores.copy(), n_select=5, ascending=False)
    assert out1.get_column("ID").to_list() == out2.get_column("ID").to_list()


def test_safe_select_no_argsort_without_kind_in_base_py() -> None:
    """Static assertion: every np.argsort call in _safe_select_top_k passes kind='stable'."""
    src = inspect.getsource(AcquisitionFunction._safe_select_top_k)
    for line in src.splitlines():
        stripped = line.strip()
        if "np.argsort(" in stripped and not stripped.startswith("#"):
            assert "kind=" in stripped, f"argsort without kind=: {line}"


def test_safe_select_full_sort_stable_tie_break_ascending() -> None:
    """For ascending full-sort with stable tie-break, lower input indices win among equals."""
    acq = _DummyAcquisition()
    scores = np.array([5.0, 5.0, 5.0, 1.0, 1.0])
    compounds = _compounds(len(scores))
    out = acq._safe_select_top_k(compounds, scores.copy(), n_select=5, ascending=True)
    ids = out.get_column("ID").to_list()
    # Tied 1.0 values come first (ascending) in input order; tied 5.0 values follow in input order.
    assert ids == ["C000003", "C000004", "C000000", "C000001", "C000002"]

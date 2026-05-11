"""Tests for stable sort in ScoreBasedPruner (feature 019).

Covers FR-015 (US8). Determinism + tie-break on input order.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from learnm8.pruning.score_based import ScoreBasedPruner


pytestmark = pytest.mark.unit


def _compounds(predictions: np.ndarray) -> pl.DataFrame:
    n = len(predictions)
    return pl.DataFrame({
        "ID": [f"C{i:06d}" for i in range(n)],
        "SMILES": ["c1ccccc1"] * n,
    })


def test_pruning_deterministic_within_process() -> None:
    rng = np.random.default_rng(19_2026)
    predictions = rng.choice(np.array([1.0, 5.0, 10.0]), size=10_000).astype(np.float32)
    compounds = _compounds(predictions)
    pruner = ScoreBasedPruner(pruning_fraction=0.5, score_direction="higher")
    out1 = pruner.prune(compounds, predictions).get_column("ID").to_list()
    out2 = pruner.prune(compounds, predictions).get_column("ID").to_list()
    assert out1 == out2


def test_pruning_tied_predictions_stable_input_order_higher() -> None:
    """For score_direction='higher', kind='stable' picks lowest input indices among ties for the kept set."""
    predictions = np.array([5.0, 5.0, 5.0, 1.0, 1.0])
    compounds = _compounds(predictions)
    pruner = ScoreBasedPruner(pruning_fraction=0.4, score_direction="higher")
    out = pruner.prune(compounds, predictions)
    kept_ids = sorted(out.get_column("ID").to_list())
    assert kept_ids == ["C000000", "C000001", "C000002"]


def test_pruning_tied_predictions_lower_direction() -> None:
    predictions = np.array([5.0, 1.0, 1.0, 1.0, 9.0])
    compounds = _compounds(predictions)
    pruner = ScoreBasedPruner(pruning_fraction=0.4, score_direction="lower")
    out = pruner.prune(compounds, predictions)
    kept_ids = sorted(out.get_column("ID").to_list())
    assert kept_ids == ["C000001", "C000002", "C000003"]


def test_pruning_uses_argpartition_for_hot_path() -> None:
    """Static check: prune() uses argpartition (O(n)) rather than argsort (O(n log n))."""
    src = inspect.getsource(ScoreBasedPruner)
    assert "np.argpartition" in src, (
        "ScoreBasedPruner.prune should use np.argpartition for the O(n) hot-path; "
        "stable secondary sort within the kept slice preserves input-order stability."
    )
    for line in src.splitlines():
        stripped = line.strip()
        if "np.argsort(" in stripped and not stripped.startswith("#"):
            assert "kind=" in stripped, f"argsort without kind=: {line}"


@pytest.mark.slow
def test_pruning_property_no_change_in_kept_set_across_runs() -> None:
    rng = np.random.default_rng(19_2026)
    predictions = rng.choice(np.array([1.0, 5.0, 10.0]), size=1_000_000).astype(np.float32)
    compounds = _compounds(predictions)
    pruner = ScoreBasedPruner(pruning_fraction=0.5, score_direction="higher")
    baseline = pruner.prune(compounds, predictions).get_column("ID").to_list()
    for _ in range(9):
        repeat = pruner.prune(compounds, predictions).get_column("ID").to_list()
        assert repeat == baseline


def test_pruning_matches_golden_fixture_when_available() -> None:
    """Cross-machine determinism via committed golden fixture."""
    fixture = Path(__file__).parent / "fixtures" / "golden_stable_sort.npz"
    if not fixture.exists():
        pytest.skip("golden_stable_sort.npz not available; skipping")
    data = np.load(fixture)
    predictions = data["predictions"]
    expected_kept = data["kept_ids"]
    compounds = _compounds(predictions)
    pruner = ScoreBasedPruner(pruning_fraction=0.5, score_direction="higher")
    actual = pruner.prune(compounds, predictions).get_column("ID").to_numpy()
    np.testing.assert_array_equal(actual, expected_kept)

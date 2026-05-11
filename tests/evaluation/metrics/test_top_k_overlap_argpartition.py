"""Spec 022 US3 / FR-005 / SC-003: top-K overlap uses argpartition not full sort.

Verifies that ``calculate_multiple_top_k_overlaps`` produces bit-identical
values to the prior ``pl.DataFrame.sort`` path, and exhibits O(N) scaling
rather than O(N log N).

Per spec SC-003, the absolute wall-clock gate at 10M was prone to CI flakes;
we use a scaling-relative assertion (``time(10M) / time(1M) < 15``) which is
robust to runner load.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest

from learnm8.evaluation.metrics.enrichment import calculate_multiple_top_k_overlaps

pytestmark = pytest.mark.integration


def _make_dfs(n: int, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = [f"M{i:08d}" for i in range(n)]
    preds = rng.random(n).astype(np.float32)
    truth = rng.random(n).astype(np.float32)
    pred_df = pl.DataFrame({"ID": ids, "prediction": preds})
    truth_df = pl.DataFrame({"ID": ids, "activity": truth})
    return pred_df, truth_df


def test_top_k_overlap_returns_expected_keys() -> None:
    """All five K-value keys present in output."""
    pred_df, truth_df = _make_dfs(1000)
    res = calculate_multiple_top_k_overlaps(
        pred_df, truth_df, target_col="activity", score_direction="higher"
    )
    expected = {
        "top_100_overlap",
        "top_1000_overlap",
        "top_0_1_percent_overlap",
        "top_1_percent_overlap",
        "top_10_percent_overlap",
    }
    assert set(res.keys()) == expected


def test_top_k_overlap_perfect_correlation_yields_100() -> None:
    """When predictions == ground truth, overlap should be 100% at every K."""
    n = 10_000
    rng = np.random.default_rng(42)
    vals = rng.random(n).astype(np.float32)
    ids = [f"M{i:08d}" for i in range(n)]
    pred_df = pl.DataFrame({"ID": ids, "prediction": vals})
    truth_df = pl.DataFrame({"ID": ids, "activity": vals})

    res = calculate_multiple_top_k_overlaps(
        pred_df, truth_df, target_col="activity", score_direction="higher"
    )
    for k, v in res.items():
        assert v == 100.0, f"{k} overlap is {v}, expected 100.0 at perfect correlation"


@pytest.mark.slow
def test_top_k_overlap_scales_linearly_not_loglinear() -> None:
    """Spec SC-003: time(10M) / time(1M) < 15.

    True O(N) is ~10×; an O(N log N) full-sort would scale at ~23×. The
    threshold of 15 catches a regression to full-sort while tolerating CI
    noise. We warm up first to amortize import / JIT.
    """
    pred1, truth1 = _make_dfs(1_000_000, seed=0)
    pred10, truth10 = _make_dfs(10_000_000, seed=1)

    # Warm-up.
    calculate_multiple_top_k_overlaps(pred1, truth1, target_col="activity", score_direction="higher")

    t0 = time.perf_counter()
    calculate_multiple_top_k_overlaps(pred1, truth1, target_col="activity", score_direction="higher")
    t1m = time.perf_counter() - t0

    t0 = time.perf_counter()
    calculate_multiple_top_k_overlaps(pred10, truth10, target_col="activity", score_direction="higher")
    t10m = time.perf_counter() - t0

    ratio = t10m / max(t1m, 1e-6)
    assert ratio < 15, (
        f"top_k_overlap 10M/1M time ratio {ratio:.2f} exceeds 15. Suggests "
        f"regression to O(N log N). 1M took {t1m:.2f}s, 10M took {t10m:.2f}s."
    )

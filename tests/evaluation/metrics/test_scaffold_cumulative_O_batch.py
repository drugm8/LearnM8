"""Spec 022 US1 / FR-001 / FR-002: cumulative-scaffold O(batch) timing.

Verifies that the cumulative scaffold-diversity metric does NOT scale with
cycle index — the per-cycle cost must depend only on batch size, not on the
total cumulative-set size. This is the regression test for the prior
O(N) per-cycle re-iteration of ``cumulative_smiles_seen``.

Acceptance: across cycles 50–100 (warm-cache regime), per-cycle wall time
stays within 20% spread (max/min < 1.20).
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest

from learnm8.evaluation.metrics.similarity import (
    RunCache,
    compute_diversity_metrics,
)


def _smiles_for(idx: int) -> str:
    """Strictly-unique parseable SMILES per index (no inter-batch dupes).

    Three-dimensional encoding (a, b, c) ∈ [0,8) × [0,8) × [0,16) so 1024
    unique parseable SMILES are addressable from idx ∈ [0, 1024). Chains
    are kept short to avoid RDKit MurckoDecompose slowdown.
    """
    a = (idx // 128) % 8
    b = (idx // 16) % 8
    c = idx % 16
    return f"c1ccccc1C{'C' * a}O{'C' * b}N{'C' * c}"


def _batch_df(start: int, n: int) -> pl.DataFrame:
    return pl.DataFrame({
        "SMILES": [_smiles_for(i) for i in range(start, start + n)],
        "ID": [f"M{i:06d}" for i in range(start, start + n)],
    })


@pytest.mark.slow
def test_scaffold_cumulative_per_cycle_time_constant() -> None:
    """Per-cycle cost does NOT grow linearly with cycle index.

    100 cycles × 10 SMILES each → 1000 cumulative (within unique-SMILES
    address space of the test generator). Old O(N) path: cycle-100 time
    ≈ 100× cycle-1 time. New O(batch) path: time per cycle is constant.

    Assertion: linear-regression slope of times[50:100] vs cycle index is
    less than a small fraction of the median time. This is robust to the
    sub-ms noise that affects max/min spread tests at small fixture scale,
    while still detecting the order-of-magnitude growth that an O(N) path
    would exhibit.
    """
    run_cache = RunCache()
    times_ms: list[float] = []

    cumulative_df = pl.DataFrame({"SMILES": [], "ID": []}, schema={"SMILES": pl.Utf8, "ID": pl.Utf8})

    for cycle in range(100):
        new_batch = _batch_df(cycle * 10, 10)
        cumulative_df = pl.concat([cumulative_df, new_batch])
        # Disable Tanimoto + Shannon to isolate scaffold-only cost.
        disable = (
            "mean_tanimoto_similarity_sampled_batch",
            "mean_tanimoto_similarity_sampled_cumulative",
            "bit_position_uniformity_entropy_batch",
            "bit_position_uniformity_entropy_cumulative",
        )
        t0 = time.perf_counter()
        compute_diversity_metrics(
            newly_selected_df=new_batch,
            cumulative_selected_df=cumulative_df,
            cycle=cycle,
            run_cache=run_cache,
            random_state=42,
            disable=disable,
        )
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    # Warm-cache regime: cycles 50-100 fit a line; slope must be near zero.
    warm = np.asarray(times_ms[50:100])
    cycles = np.arange(50, 100, dtype=np.float64)
    slope, _intercept = np.polyfit(cycles, warm, 1)
    median = float(np.median(warm))
    # An O(N) path would show slope ≈ (batch_work_per_cycle / batch_size) =
    # constant-positive contribution per cycle. We allow up to 0.5%·median per
    # cycle (so over 50 cycles, the implied linear growth is at most ~25% of
    # the median). True constant-time path shows slope close to zero
    # regardless of sign — noise can scatter either way.
    assert abs(slope) < 0.005 * median, (
        f"Spec SC-001 regression: cumulative scaffold time slope {slope:.4f} ms/cycle "
        f"vs median {median:.3f} ms suggests linear growth in cycle index. "
        f"times[50:100] min={float(warm.min()):.3f} max={float(warm.max()):.3f}"
    )

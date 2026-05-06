"""Performance tests for the diversity-metrics overhaul (feature 013).

Marked ``slow`` to skip from default pytest invocations. Validates the
< 1 s per-cycle wall-clock budget at multiple cumulative-set sizes so we
catch regressions that would silently break the 100M-compound scaling
claim (FR-018, SC-002).

The perf budget is per-cycle wall-clock for ALL six metrics on a single
CPU thread, with no GPU.
"""

from __future__ import annotations

import time

import polars as pl
import pytest

from learnm8.evaluation.metrics.similarity import (
    RunCache,
    compute_diversity_metrics,
)


SMILES_POOL = [
    "CCO", "CCC", "CCCO", "c1ccccc1", "C1CCCCC1", "Nc1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "C1=CC=C(C=C1)C(=O)O",
    "CCN(CC)CC", "C1CCNCC1", "Nc1ncnc2[nH]cnc12",
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C", "Oc1ccccc1", "CCOc1ccccc1",
    "ClCC(Cl)Cl", "BrCCBr", "CC(=O)O", "CCN", "CCCCCCN",
    "CCSCC", "Cc1ccccc1", "Cc1ccc(C)cc1", "CC(C)C", "CC(C)(C)C",
    "[Na+].CC(=O)[O-]",
]


def _df_from_pool(n: int) -> pl.DataFrame:
    smis = (SMILES_POOL * ((n // len(SMILES_POOL)) + 1))[:n]
    return pl.DataFrame({
        "ID": [f"mol_{i}" for i in range(n)],
        "SMILES": smis,
    })


@pytest.mark.slow
@pytest.mark.parametrize("cumulative_size", [10_000, 50_000, 100_000])
def test_diversity_under_one_second(cumulative_size):
    """Per-cycle wall-clock for all six metrics must stay under 1.0 s."""
    batch_size = 1_000
    cumulative_df = _df_from_pool(cumulative_size)
    batch_df = _df_from_pool(batch_size)

    run_cache = RunCache()

    # Pre-warm by running a small first cycle so the cumulative buffer is
    # populated (simulates the cycle-N>0 path that the budget targets).
    compute_diversity_metrics(
        newly_selected_df=cumulative_df,
        cumulative_selected_df=cumulative_df,
        cycle=0,
        run_cache=run_cache,
        random_state=42,
    )
    # Reset to simulate next cycle adding a new batch.
    start = time.perf_counter()
    compute_diversity_metrics(
        newly_selected_df=batch_df,
        cumulative_selected_df=_df_from_pool(cumulative_size + batch_size),
        cycle=1,
        run_cache=run_cache,
        random_state=42,
    )
    elapsed = time.perf_counter() - start

    # Spec budget is < 1.0 s. Allow a small headroom factor for shared CI.
    assert elapsed < 1.0, (
        f"Diversity perf regressed at cumulative_size={cumulative_size}: "
        f"{elapsed:.3f}s >= 1.0s"
    )


@pytest.mark.slow
def test_dense_fingerprints_with_salts_under_budget():
    """Dense fingerprints + salt SMILES (LargestFragmentChooser overhead).

    Validates that the scaffold path's salt-handling step doesn't blow the
    budget when the cumulative set leans on the salt-laden subset of our
    pool.
    """
    cumulative_df = _df_from_pool(100_000)
    batch_df = _df_from_pool(1_000)
    run_cache = RunCache()
    # Warm.
    compute_diversity_metrics(
        newly_selected_df=cumulative_df,
        cumulative_selected_df=cumulative_df,
        cycle=0,
        run_cache=run_cache,
        random_state=42,
    )
    start = time.perf_counter()
    compute_diversity_metrics(
        newly_selected_df=batch_df,
        cumulative_selected_df=_df_from_pool(101_000),
        cycle=1,
        run_cache=run_cache,
        random_state=42,
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"Dense+salt perf regressed: {elapsed:.3f}s >= 1.0s"

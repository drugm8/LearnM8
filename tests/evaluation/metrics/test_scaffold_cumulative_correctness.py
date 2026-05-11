"""Spec 022 US1 / FR-001 / FR-002: incremental cumulative-scaffold correctness.

Verifies the incremental ``cumulative_unique_scaffolds`` accumulator produces
bit-identical values to a one-shot batch computation over the union of all
cumulative SMILES. This is the correctness gate for the optimization in
``test_scaffold_cumulative_O_batch.py`` (which only checks timing).
"""

from __future__ import annotations

import polars as pl
import pytest

from learnm8.evaluation.metrics.similarity import (
    RunCache,
    _scaffold_diversity_index,
    compute_diversity_metrics,
)

pytestmark = pytest.mark.integration


def _smiles_for(idx: int) -> str:
    """Produce a strictly-unique parseable SMILES per index.

    Active learning never selects the same compound twice; the spec's
    acceptance scenario ("incremental == one-shot") only holds for a
    deduplicated cumulative set. Three-dim encoding keeps chain lengths
    short while producing 1024 unique parseable SMILES.
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


def test_scaffold_cumulative_incremental_matches_one_shot() -> None:
    """Incremental cumulative scaffold equals one-shot batch over the union.

    10 cycles × 50 compounds each (500 cumulative; within the 1024 unique
    address space of the test generator). After each cycle the
    incrementally-accumulated metric must match a fresh one-shot computation
    over all-seen SMILES.
    """
    run_cache = RunCache()
    one_shot_cache: dict[str, str] = {}

    cumulative_df = pl.DataFrame({"SMILES": [], "ID": []}, schema={"SMILES": pl.Utf8, "ID": pl.Utf8})

    for cycle in range(10):
        new_batch = _batch_df(cycle * 50, 50)
        cumulative_df = pl.concat([cumulative_df, new_batch])
        # Disable everything except scaffold metrics.
        disable = (
            "mean_tanimoto_similarity_sampled_batch",
            "mean_tanimoto_similarity_sampled_cumulative",
            "bit_marginal_entropy_batch",
            "bit_marginal_entropy_cumulative",
        )
        result = compute_diversity_metrics(
            newly_selected_df=new_batch,
            cumulative_selected_df=cumulative_df,
            cycle=cycle,
            run_cache=run_cache,
            random_state=42,
            disable=disable,
        )

        # One-shot reference: re-compute the cumulative scaffold metric from
        # scratch over the full union, with a fresh cache (simulates "old"
        # O(N) recomputation path).
        all_smiles: list[str] = cumulative_df.get_column("SMILES").to_list()
        one_shot = _scaffold_diversity_index(all_smiles, one_shot_cache)

        incremental = result["scaffold_diversity_index_cumulative"]
        assert incremental == one_shot, (
            f"Cycle {cycle}: incremental scaffold {incremental!r} does not match "
            f"one-shot {one_shot!r} over {len(all_smiles)} cumulative SMILES."
        )

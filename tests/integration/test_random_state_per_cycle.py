"""Spec 021 / US2 / FR-004/FR-005 — random_state plumbed through the cycle.

Two ``run_active_learning(random_state=42, ...)`` invocations with the
``random`` acquisition strategy must produce bit-identical per-cycle selected
ID sets, AND cycles 1 and 2 within one run must select different IDs (proving
per-cycle seed derivation actually varies across cycles).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

pytestmark = pytest.mark.integration


def _write_combined_csv(tmp_path: Path, n: int = 40) -> Path:
    """Write a single CSV used as both compound_pool and oracle.

    Columns: ID, SMILES, Activity. Activity values are deterministic so two
    runs with the same seed produce bit-identical experiments.
    """
    smiles_alphabet = ["CCO", "CCC", "CCN", "CCl", "CCBr", "C=O", "CN", "C(=O)O"]
    rows = [
        (f"C{i:04d}", smiles_alphabet[i % len(smiles_alphabet)], float(i % 7) + 0.01 * i)
        for i in range(n)
    ]
    df = pl.DataFrame(rows, schema=["ID", "SMILES", "Activity"], orient="row")
    csv_path = tmp_path / "compounds.csv"
    df.write_csv(csv_path)
    return csv_path


def _per_cycle_selected_ids(results: dict) -> dict[int, set[str]]:
    """Extract {cycle_index: selected_ids} from run_active_learning results.

    Cycle 0 is the initialization batch; cycles ≥ 1 use the configured
    acquisition strategy and are the cycles whose seeds must vary.
    """
    cdf: pl.DataFrame = results["compounds_df"]
    cycles = sorted({int(c) for c in cdf["selected_cycle"].drop_nulls().to_list()})
    return {
        c: set(cdf.filter(pl.col("selected_cycle") == c)["ID"].to_list())
        for c in cycles
    }


def _run(csv: Path, output_dir: Path, *, random_state: int = 42, n_cycles: int = 3):
    from learnm8.api import run_active_learning

    return run_active_learning(
        compound_pool=str(csv),
        oracle=str(csv),
        target_col="Activity",
        learner="lr",
        strategy="random",
        initial_strategy="random",
        featurizer="morgan",
        n_cycles=n_cycles,
        batch_fraction=0.1,
        random_state=random_state,
        output_dir=str(output_dir),
        score_direction="higher",
    )


def test_two_runs_with_same_seed_produce_bit_identical_per_cycle_selections(tmp_path):
    csv = _write_combined_csv(tmp_path, n=40)

    res_a = _run(csv, tmp_path / "run_a", random_state=42, n_cycles=3)
    res_b = _run(csv, tmp_path / "run_b", random_state=42, n_cycles=3)

    sel_a = _per_cycle_selected_ids(res_a)
    sel_b = _per_cycle_selected_ids(res_b)

    assert sel_a, "Could not extract per-cycle selections from run A."
    assert set(sel_a.keys()) == set(sel_b.keys())
    for c in sel_a:
        assert sel_a[c] == sel_b[c], f"Cycle {c} selections differ across runs."


def test_per_cycle_selections_differ_across_cycles_within_one_run(tmp_path):
    csv = _write_combined_csv(tmp_path, n=80)

    res = _run(csv, tmp_path / "run_one", random_state=42, n_cycles=3)
    sel = _per_cycle_selected_ids(res)

    acquisition_cycles = sorted(c for c in sel if c >= 1)
    assert len(acquisition_cycles) >= 2, (
        f"Need at least 2 acquisition cycles; got cycles {sorted(sel.keys())}."
    )
    c1, c2 = acquisition_cycles[0], acquisition_cycles[1]
    assert sel[c1] != sel[c2], (
        f"Cycle {c1} and {c2} produced identical selection sets — per-cycle "
        f"seed derivation is not active (FR-005 regression)."
    )

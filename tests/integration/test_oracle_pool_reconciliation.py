"""Integration test: pool must be reconciled with oracle's measurable IDs.

Regression test for the case where the source CSV contains compounds with
non-numeric target values (e.g. ``"no_score"`` from a failed docking run).
``CSVOracle`` silently drops those rows from its ground truth at construction,
so the validated AL pool must be filtered to match — otherwise random
selection will eventually hit a compound the oracle cannot measure and
``select_initial_batch`` will raise ``OracleError``.
"""

from pathlib import Path

import polars as pl
import pytest

from learnm8 import run_active_learning


@pytest.fixture
def mixed_score_csv(tmp_path: Path) -> Path:
    """Twelve compounds; three have ``no_score`` and must be excluded."""
    rows = [
        ("c01", "CCO", "0.10"),
        ("c02", "CCC", "0.20"),
        ("c03", "CCN", "no_score"),
        ("c04", "c1ccccc1", "0.30"),
        ("c05", "COC", "0.40"),
        ("c06", "CC(=O)O", "no_score"),
        ("c07", "CCBr", "0.50"),
        ("c08", "CCS", "0.60"),
        ("c09", "CCF", "0.70"),
        ("c10", "CCI", "0.80"),
        ("c11", "C1CCCCC1", "no_score"),
        ("c12", "CCOCC", "0.90"),
    ]
    df = pl.DataFrame(rows, schema=["ID", "SMILES", "Activity"], orient="row")
    csv_path = tmp_path / "mixed_score.csv"
    df.write_csv(csv_path)
    return csv_path


@pytest.mark.integration
def test_pool_reconciled_with_oracle_drops_unmeasurable(mixed_score_csv, tmp_path):
    """Compounds with non-numeric target in source CSV must be filtered from pool.

    Before the fix this raised ``OracleError`` once a 'no_score' compound was
    picked during cycle 0 random initialization. After the fix the pool only
    contains the 9 compounds the oracle can actually measure.
    """
    results = run_active_learning(
        compound_pool=str(mixed_score_csv),
        oracle=str(mixed_score_csv),
        target_col="Activity",
        learner="rf",
        featurizer="morgan",
        n_cycles=2,
        batch_fraction=0.5,
        random_state=42,
        output_dir=str(tmp_path / "out"),
    )

    validation_result = results["validation_result"]
    valid_ids = set(validation_result.valid_compounds["ID"].to_list())
    invalid_ids = set(validation_result.invalid_compounds["ID"].to_list())

    assert valid_ids == {"c01", "c02", "c04", "c05", "c07", "c08", "c09", "c10", "c12"}
    assert invalid_ids == {"c03", "c06", "c11"}
    for cid in ("c03", "c06", "c11"):
        assert cid in validation_result.validation_errors
        assert "oracle" in validation_result.validation_errors[cid].lower()

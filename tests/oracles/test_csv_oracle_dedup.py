"""Spec 021 / US4 / FR-009 — CSVOracle deduplicates duplicate IDs at load."""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl
import pytest

from learnm8.exceptions import LearnM8Warning
from learnm8.oracles.csv_oracle import CSVOracle

pytestmark = pytest.mark.unit


def _write_oracle_csv(tmp_path: Path, rows: list[tuple[str, str, float]]) -> Path:
    df = pl.DataFrame(
        {
            "ID": [r[0] for r in rows],
            "SMILES": [r[1] for r in rows],
            "Activity": [r[2] for r in rows],
        }
    )
    csv = tmp_path / "oracle.csv"
    df.write_csv(csv)
    return csv


def test_csv_oracle_emits_warning_on_duplicate_ids(tmp_path):
    rows = [
        ("C001", "CCO", 1.0),
        ("C002", "CCC", 2.0),
        ("C001", "CCO", 9.9),
        ("C003", "CCN", 3.0),
        ("C002", "CCC", 8.8),
        ("C002", "CCC", 7.7),
    ]
    csv = _write_oracle_csv(tmp_path, rows)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CSVOracle(str(csv))
        warn_msgs = [str(item.message) for item in w if issubclass(item.category, LearnM8Warning)]

    assert any("3" in msg and "duplicate" in msg.lower() for msg in warn_msgs), (
        f"Expected LearnM8Warning mentioning 3 duplicates; got {warn_msgs}"
    )


def test_csv_oracle_measure_returns_first_occurrence_only(tmp_path):
    rows = [
        ("C001", "CCO", 1.0),
        ("C002", "CCC", 2.0),
        ("C001", "CCO", 9.9),
        ("C002", "CCC", 8.8),
    ]
    csv = _write_oracle_csv(tmp_path, rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LearnM8Warning)
        oracle = CSVOracle(str(csv))

    query = pl.DataFrame({"ID": ["C001", "C002"], "SMILES": ["CCO", "CCC"]})
    result = oracle.measure(query, ["Activity"])

    assert result.height == 2
    assert result.filter(pl.col("ID") == "C001")["Activity"].item() == 1.0
    assert result.filter(pl.col("ID") == "C002")["Activity"].item() == 2.0


def test_csv_oracle_no_warning_when_ids_unique(tmp_path):
    rows = [
        ("C001", "CCO", 1.0),
        ("C002", "CCC", 2.0),
        ("C003", "CCN", 3.0),
    ]
    csv = _write_oracle_csv(tmp_path, rows)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CSVOracle(str(csv))
        learnm8_warnings = [item for item in w if issubclass(item.category, LearnM8Warning)]

    dup_warnings = [item for item in learnm8_warnings if "duplicate" in str(item.message).lower()]
    assert not dup_warnings, f"Unexpected duplicate-ID warning on unique CSV: {dup_warnings}"


def test_csv_oracle_measure_is_bit_identical_across_runs(tmp_path):
    rows = [
        ("C001", "CCO", 1.0),
        ("C002", "CCC", 2.0),
        ("C001", "CCO", 9.9),
        ("C003", "CCN", 3.0),
    ]
    csv = _write_oracle_csv(tmp_path, rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LearnM8Warning)
        oracle1 = CSVOracle(str(csv))
        oracle2 = CSVOracle(str(csv))

    query = pl.DataFrame({"ID": ["C001", "C002", "C003"], "SMILES": ["CCO", "CCC", "CCN"]})
    out1 = oracle1.measure(query, ["Activity"])
    out2 = oracle2.measure(query, ["Activity"])

    assert out1.equals(out2)

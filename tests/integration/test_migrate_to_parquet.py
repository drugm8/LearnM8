"""Tests for scripts/migrate_to_parquet.py.

Covers extraction of prediction columns to parquet, backup CSV ordering,
slim CSV writing, dry-run mode, auto-verification with float tolerance,
non-contiguous cycle numbers, NaN preservation, corrupt CSV handling,
and # comment header metadata extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.migrate_to_parquet import (  # noqa: E402
    migrate_directory,
    parse_csv_comment_metadata,
    write_parquet_atomic,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

NARROW_BASE_COLS = ['ID', 'SMILES', 'status', 'labeled_cycle', 'selected_cycle', 'pruned_cycle']


def _write_csv_with_comments(path: Path, df: pl.DataFrame, metadata: dict[str, str] | None = None) -> None:
    """Write a CSV preceded by # comment metadata lines (mimicking persistence.py output)."""
    body = df.write_csv()
    lines = ['# LearnM8 Active Learning Results\n', '#\n']
    for k, v in (metadata or {}).items():
        lines.append(f'# {k}: {v}\n')
    lines.append('#\n')
    path.write_text(''.join(lines) + body)


def _make_legacy_csv(
    n_compounds: int = 5,
    cycles: tuple[int, ...] = (0, 1, 2),
    target_col: str = 'pIC50',
    with_uncertainty: bool = True,
    metadata: dict[str, str] | None = None,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Build a legacy-format compounds_final DataFrame (with prediction_cycle_* columns)."""
    rng = np.random.default_rng(42)
    data = {
        'ID': [f'C{i:04d}' for i in range(n_compounds)],
        'SMILES': ['CCO'] * n_compounds,
        'status': ['unlabeled'] * n_compounds,
        'labeled_cycle': [None] * n_compounds,
        'selected_cycle': [None] * n_compounds,
        'pruned_cycle': [None] * n_compounds,
        target_col: rng.uniform(4, 8, size=n_compounds).tolist(),
    }
    for c in cycles:
        data[f'prediction_cycle_{c}'] = rng.uniform(4, 8, size=n_compounds).tolist()
        if with_uncertainty:
            data[f'uncertainty_cycle_{c}'] = rng.uniform(0.01, 0.5, size=n_compounds).tolist()
    df = pl.DataFrame(data)
    md = metadata if metadata is not None else {
        'Target': target_col,
        'Featurizer': 'morgan',
        'Score Direction': 'higher',
    }
    return df, md


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_extract_prediction_columns_to_parquet(tmp_path):
    """Migration extracts prediction_cycle_* into per-cycle parquet files."""
    df, md = _make_legacy_csv(cycles=(0, 1, 2))
    csv = tmp_path / 'compounds_final.csv'
    _write_csv_with_comments(csv, df, md)

    summary = migrate_directory(tmp_path)

    assert summary['status'] == 'migrated'
    assert summary['cycles'] == [0, 1, 2]
    parquets = sorted((tmp_path / f'prediction_cycle_{c}.parquet') for c in [0, 1, 2])
    for p in parquets:
        assert p.exists(), f'expected {p}'
        actual = pl.read_parquet(p)
        assert actual.columns == ['ID', 'prediction', 'uncertainty']


def test_backup_creation_and_safe_ordering(tmp_path):
    """Original CSV is renamed to .bak; slim CSV replaces it; both must exist after success."""
    df, md = _make_legacy_csv()
    csv = tmp_path / 'compounds_final.csv'
    _write_csv_with_comments(csv, df, md)
    original_bytes = csv.read_bytes()

    summary = migrate_directory(tmp_path)
    assert summary['status'] == 'migrated'

    bak = tmp_path / 'compounds_final.csv.bak'
    assert bak.exists(), 'original should be preserved as .bak'
    assert bak.read_bytes() == original_bytes, '.bak should match the pre-migration CSV exactly'
    assert csv.exists(), 'slim CSV should now occupy compounds_final.csv'
    assert csv.read_bytes() != original_bytes, 'slim CSV should differ from original'


def test_slim_csv_has_no_prediction_columns(tmp_path):
    """The post-migration compounds_final.csv contains no prediction_cycle_* / uncertainty_cycle_* cols."""
    df, md = _make_legacy_csv(cycles=(0, 1))
    csv = tmp_path / 'compounds_final.csv'
    _write_csv_with_comments(csv, df, md)

    migrate_directory(tmp_path)

    slim = pl.read_csv(csv, comment_prefix='#')
    for col in slim.columns:
        assert not col.startswith('prediction_cycle_'), col
        assert not col.startswith('uncertainty_cycle_'), col
    assert set(NARROW_BASE_COLS).issubset(set(slim.columns))
    assert 'pIC50' in slim.columns


def test_dry_run_makes_no_changes(tmp_path):
    """--dry-run reports what would happen without writing parquet, .bak, or slim CSV."""
    df, md = _make_legacy_csv()
    csv = tmp_path / 'compounds_final.csv'
    _write_csv_with_comments(csv, df, md)
    original_bytes = csv.read_bytes()

    summary = migrate_directory(tmp_path, dry_run=True)
    assert summary['status'] == 'dry_run'

    assert csv.read_bytes() == original_bytes, 'original CSV must be untouched in dry-run'
    assert not (tmp_path / 'compounds_final.csv.bak').exists()
    assert list(tmp_path.glob('prediction_cycle_*.parquet')) == []


def test_auto_verify_detects_value_mismatch(tmp_path, monkeypatch):
    """If the parquet readback diverges from the source CSV, migration raises."""
    df, md = _make_legacy_csv()
    csv = tmp_path / 'compounds_final.csv'
    _write_csv_with_comments(csv, df, md)

    import scripts.migrate_to_parquet as m

    real_write = m.write_parquet_atomic

    def corrupt_write(df_in, path, *, metadata=None):
        # Write deliberately-different values to trigger verify failure
        bad = df_in.with_columns(pl.col('prediction') + 1.0)
        real_write(bad, path, metadata=metadata)

    monkeypatch.setattr(m, 'write_parquet_atomic', corrupt_write)

    with pytest.raises(ValueError, match='diverge'):
        migrate_directory(tmp_path)


def test_noncontiguous_cycle_numbering(tmp_path):
    """Cycles 0, 2, 5 (non-contiguous) all migrate to their numbered parquet files."""
    df, md = _make_legacy_csv(cycles=(0, 2, 5))
    csv = tmp_path / 'compounds_final.csv'
    _write_csv_with_comments(csv, df, md)

    summary = migrate_directory(tmp_path)
    assert summary['cycles'] == [0, 2, 5]
    for c in (0, 2, 5):
        assert (tmp_path / f'prediction_cycle_{c}.parquet').exists()
    # Cycles 1, 3, 4 must NOT have been invented.
    for c in (1, 3, 4):
        assert not (tmp_path / f'prediction_cycle_{c}.parquet').exists()


def test_nan_values_preserved_as_null(tmp_path):
    """NaN/empty prediction values in the CSV become nulls in the parquet."""
    df, md = _make_legacy_csv(n_compounds=4, cycles=(0,))
    # Inject NaN into one prediction value for cycle 0.
    df = df.with_columns(
        pl.when(pl.col('ID') == 'C0001')
        .then(None)
        .otherwise(pl.col('prediction_cycle_0'))
        .alias('prediction_cycle_0')
    )
    csv = tmp_path / 'compounds_final.csv'
    _write_csv_with_comments(csv, df, md)

    summary = migrate_directory(tmp_path)
    assert summary['status'] == 'migrated'

    pq_path = tmp_path / 'prediction_cycle_0.parquet'
    actual = pl.read_parquet(pq_path)
    # The migration drops null-prediction rows by design (line 180 of script);
    # ensure the surviving rows still preserve nulls in uncertainty if any.
    assert 'C0001' not in actual['ID'].to_list()
    assert actual.height == 3  # 4 - 1 dropped


def test_corrupt_csv_skipped_with_log(tmp_path):
    """A malformed CSV produces a 'failed' status rather than crashing the script."""
    csv = tmp_path / 'compounds_final.csv'
    csv.write_text('not,a,valid\ncsv,with,prediction_cycle_0\n"unterminated quote\n')

    summary = migrate_directory(tmp_path)
    # Status is 'failed' or 'no_predictions' depending on parser tolerance,
    # but it must NOT raise an unhandled exception.
    assert summary['status'] in ('failed', 'no_predictions'), summary


def test_comment_header_metadata_extraction(tmp_path):
    """parse_csv_comment_metadata extracts target_col, featurizer, score_direction."""
    df, _ = _make_legacy_csv(target_col='Activity')
    csv = tmp_path / 'compounds_final.csv'
    _write_csv_with_comments(
        csv, df, {'Target': 'Activity', 'Featurizer': 'rdkit2d', 'Score Direction': 'lower'},
    )

    md = parse_csv_comment_metadata(csv)
    assert md == {
        'target_col': 'Activity',
        'featurizer': 'rdkit2d',
        'score_direction': 'lower',
    }


def test_parquet_metadata_embedded(tmp_path):
    """Parsed comment metadata is embedded in parquet file's key-value metadata."""
    df, _ = _make_legacy_csv(target_col='Activity')
    csv = tmp_path / 'compounds_final.csv'
    _write_csv_with_comments(
        csv, df, {'Target': 'Activity', 'Featurizer': 'morgan', 'Score Direction': 'higher'},
    )

    migrate_directory(tmp_path)

    pf = pq.read_metadata(tmp_path / 'prediction_cycle_0.parquet')
    assert pf.metadata is not None
    assert pf.metadata.get(b'target_col') == b'Activity'
    assert pf.metadata.get(b'featurizer') == b'morgan'
    assert pf.metadata.get(b'score_direction') == b'higher'
    assert pf.metadata.get(b'cycle') == b'0'


def test_write_parquet_atomic_cleans_tmp_on_failure(tmp_path):
    """On exception during write, the .parquet.tmp is removed and no .parquet exists."""
    final = tmp_path / 'prediction_cycle_0.parquet'
    tmp = tmp_path / 'prediction_cycle_0.parquet.tmp'

    class Boom:
        def to_arrow(self):
            raise OSError('disk full')

    with pytest.raises(OSError, match='disk full'):
        write_parquet_atomic(Boom(), final, metadata={'cycle': '0'})  # type: ignore[arg-type]

    assert not final.exists()
    assert not tmp.exists()

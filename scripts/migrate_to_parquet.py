#!/usr/bin/env python3
"""Migrate a single experiment output directory to parquet prediction format.

Before this migration, ``compounds_final.csv`` widened by 1-2 columns per
cycle (``prediction_cycle_N`` and ``uncertainty_cycle_N``). After migration:

- Per-cycle predictions live in ``prediction_cycle_N.parquet`` files.
- ``compounds_final.csv`` is rewritten as a narrow 7-or-8 column table.
- The original CSV is preserved as ``compounds_final.csv.bak``.

The migration is auto-verified: parquet values are read back and compared
to the original CSV values within float tolerance, and the slim CSV is
only swapped in after every parquet is verified.

Safe ordering:
    1. Read original CSV, parse comment metadata.
    2. Write each cycle's parquet to ``prediction_cycle_N.parquet.tmp``,
       then rename to the final ``.parquet`` path (atomic on POSIX).
    3. Read each parquet back, compare with the original CSV column
       within float64 tolerance (NaN-aware).
    4. Write slim CSV to ``compounds_final.csv.slim.tmp``.
    5. Rename original ``compounds_final.csv`` to ``compounds_final.csv.bak``.
    6. Rename ``compounds_final.csv.slim.tmp`` to ``compounds_final.csv``.

If any step fails, the original CSV is never lost.

Usage:
    python scripts/migrate_to_parquet.py --output-dir path/to/results

Options:
    --output-dir PATH    Directory containing compounds_final.csv (required)
    --dry-run            Print actions without writing any files
    --skip-verify        Skip auto-verification (faster but not recommended)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

PRED_COL_RE = re.compile(r'^prediction_cycle_(\d+)$')
UNC_COL_RE = re.compile(r'^uncertainty_cycle_(\d+)$')
COMMENT_KV_RE = re.compile(r'^#\s*([A-Za-z][A-Za-z0-9 _-]*?)\s*:\s*(.+?)\s*$')

NARROW_COLUMNS_BASE = [
    'ID', 'SMILES', 'status', 'labeled_cycle', 'selected_cycle', 'pruned_cycle',
]

METADATA_KEYS = {
    'target': 'target_col',
    'target column': 'target_col',
    'featurizer': 'featurizer',
    'score direction': 'score_direction',
}


def parse_csv_comment_metadata(csv_path: Path) -> dict[str, str]:
    """Parse leading ``# Key: Value`` comment lines from a CSV.

    Returns a dict with normalized keys (target_col, featurizer,
    score_direction). Stops at the first non-comment line.
    """
    metadata: dict[str, str] = {}
    try:
        with open(csv_path, encoding='utf-8') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                m = COMMENT_KV_RE.match(line)
                if not m:
                    continue
                raw_key = m.group(1).strip().lower()
                value = m.group(2).strip()
                normalized = METADATA_KEYS.get(raw_key)
                if normalized is not None:
                    metadata[normalized] = value
    except OSError as exc:
        logger.warning('Failed to read comment headers from %s: %s', csv_path, exc)
    return metadata


def detect_cycle_columns(columns: list[str]) -> tuple[dict[int, str], dict[int, str]]:
    """Map cycle number to prediction/uncertainty column names."""
    pred_cols: dict[int, str] = {}
    unc_cols: dict[int, str] = {}
    for col in columns:
        m = PRED_COL_RE.match(col)
        if m:
            pred_cols[int(m.group(1))] = col
            continue
        m = UNC_COL_RE.match(col)
        if m:
            unc_cols[int(m.group(1))] = col
    return pred_cols, unc_cols


def detect_target_column(columns: list[str]) -> str | None:
    """Detect the target column by elimination (first non-system, non-cycle column)."""
    system = set(NARROW_COLUMNS_BASE)
    for col in columns:
        if col in system:
            continue
        if PRED_COL_RE.match(col) or UNC_COL_RE.match(col):
            continue
        return col
    return None


def write_parquet_atomic(
    df: pl.DataFrame,
    path: Path,
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write parquet using temp + rename for atomicity.

    If ``metadata`` is provided, it is embedded in the parquet schema's
    key-value metadata via pyarrow.
    """
    tmp_path = path.with_suffix('.parquet.tmp')
    try:
        if metadata:
            table = df.to_arrow()
            kv = {
                k.encode('utf-8'): str(v).encode('utf-8')
                for k, v in metadata.items()
            }
            existing = table.schema.metadata or {}
            merged = {**existing, **kv}
            table = table.replace_schema_metadata(merged)
            pq.write_table(table, tmp_path, compression='zstd')
        else:
            df.write_parquet(tmp_path, compression='zstd')
        tmp_path.rename(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def verify_parquet(parquet_path: Path, expected_df: pl.DataFrame) -> None:
    """Read back parquet and verify it matches expected within float tolerance."""
    actual = pl.read_parquet(parquet_path, glob=False)
    if actual.height != expected_df.height:
        raise ValueError(
            f'{parquet_path}: row count mismatch '
            f'(parquet={actual.height}, expected={expected_df.height})'
        )
    expected_sorted = expected_df.sort('ID')
    actual_sorted = actual.sort('ID')
    for col in ('prediction', 'uncertainty'):
        if col in expected_sorted.columns and col in actual_sorted.columns:
            a = expected_sorted[col].to_numpy()
            b = actual_sorted[col].to_numpy()
            mask = ~(np.isnan(a) | np.isnan(b))
            if not np.allclose(a[mask], b[mask], rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f'{parquet_path}: column "{col}" values diverge from CSV'
                )


def migrate_directory(
    output_dir: Path,
    *,
    dry_run: bool = False,
    skip_verify: bool = False,
) -> dict:
    """Migrate one experiment directory.

    Returns a summary dict with keys:
        - status: 'migrated', 'already_migrated', 'no_csv', 'failed'
        - parquets_written: list[Path]
        - cycles: list[int]
        - error: str | None
    """
    summary = {
        'status': 'failed',
        'parquets_written': [],
        'cycles': [],
        'error': None,
        'output_dir': str(output_dir),
    }

    csv_path = output_dir / 'compounds_final.csv'
    if not csv_path.exists():
        summary['status'] = 'no_csv'
        summary['error'] = f'compounds_final.csv not found in {output_dir}'
        return summary

    bak_path = output_dir / 'compounds_final.csv.bak'
    if bak_path.exists() and not csv_path.exists():
        summary['status'] = 'already_migrated'
        return summary

    csv_metadata = parse_csv_comment_metadata(csv_path)
    try:
        df = pl.read_csv(csv_path, comment_prefix='#', glob=False)
    except Exception as exc:
        logger.warning('Failed to parse %s: %s', csv_path, exc)
        summary['error'] = f'CSV parse failed: {exc!r}'
        return summary
    pred_cols, unc_cols = detect_cycle_columns(df.columns)

    if not pred_cols:
        # Either already migrated (no prediction columns) or never had them.
        if any(output_dir.glob('prediction_cycle_*.parquet')):
            summary['status'] = 'already_migrated'
            return summary
        summary['status'] = 'no_predictions'
        summary['error'] = 'No prediction_cycle_* columns found and no parquet files exist'
        return summary

    # Prefer comment-header target_col over column-elimination heuristic.
    target_col = csv_metadata.get('target_col') or detect_target_column(df.columns)
    if target_col and target_col not in df.columns:
        # Header value disagrees with actual columns; fall back to detection.
        logger.warning(
            'Target "%s" from CSV header not in columns; falling back to detection',
            target_col,
        )
        target_col = detect_target_column(df.columns)
    cycles = sorted(pred_cols.keys())
    summary['metadata'] = dict(csv_metadata)

    written: list[Path] = []
    for cycle in cycles:
        pred_col = pred_cols[cycle]
        unc_col = unc_cols.get(cycle)

        select_cols = ['ID', pred_col]
        if unc_col is not None:
            select_cols.append(unc_col)

        cycle_df = df.select(select_cols)
        rename_map = {pred_col: 'prediction'}
        if unc_col is not None:
            rename_map[unc_col] = 'uncertainty'
        cycle_df = cycle_df.rename(rename_map)

        # Drop rows where prediction is null (unlabeled compounds without a prediction this cycle).
        cycle_df = cycle_df.filter(pl.col('prediction').is_not_null())

        parquet_path = output_dir / f'prediction_cycle_{cycle}.parquet'
        parquet_metadata: dict[str, str] = {
            'cycle': str(cycle),
            'migrated_from': csv_path.name,
            'migrated_at': datetime.now(UTC).isoformat(),
        }
        if target_col is not None:
            parquet_metadata['target_col'] = target_col
        for key in ('featurizer', 'score_direction'):
            if key in csv_metadata:
                parquet_metadata[key] = csv_metadata[key]

        if dry_run:
            logger.info('DRY-RUN would write %s (%d rows)', parquet_path, cycle_df.height)
            written.append(parquet_path)
            continue

        write_parquet_atomic(cycle_df, parquet_path, metadata=parquet_metadata)
        if not skip_verify:
            verify_parquet(parquet_path, cycle_df)
        written.append(parquet_path)

    # Slim CSV: keep base columns + target only.
    keep_cols = list(NARROW_COLUMNS_BASE)
    if target_col is not None and target_col not in keep_cols:
        keep_cols.append(target_col)
    keep_cols = [c for c in keep_cols if c in df.columns]
    slim_df = df.select(keep_cols)

    if dry_run:
        logger.info(
            'DRY-RUN would back up %s -> %s and write slim CSV (%d cols)',
            csv_path, bak_path, len(keep_cols),
        )
        summary['status'] = 'dry_run'
        summary['parquets_written'] = written
        summary['cycles'] = cycles
        return summary

    slim_tmp = output_dir / 'compounds_final.csv.slim.tmp'
    try:
        slim_df.write_csv(slim_tmp)
        csv_path.rename(bak_path)
        slim_tmp.rename(csv_path)
    except Exception as exc:
        slim_tmp.unlink(missing_ok=True)
        if not csv_path.exists() and bak_path.exists():
            bak_path.rename(csv_path)
        summary['error'] = f'CSV swap failed: {exc!r}'
        return summary

    summary['status'] = 'migrated'
    summary['parquets_written'] = written
    summary['cycles'] = cycles
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument(
        '--output-dir', required=True, type=Path,
        help='Path to experiment directory containing compounds_final.csv',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print actions without writing files',
    )
    parser.add_argument(
        '--skip-verify', action='store_true',
        help='Skip auto-verification of parquet files (faster, less safe)',
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress info logging',
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        logger.error('Not a directory: %s', output_dir)
        return 2

    summary = migrate_directory(
        output_dir,
        dry_run=args.dry_run,
        skip_verify=args.skip_verify,
    )

    status = summary['status']
    cycles = summary.get('cycles', [])
    if status in ('migrated', 'dry_run'):
        logger.info(
            '%s %s: cycles=%s, parquets=%d',
            'Would migrate' if status == 'dry_run' else 'Migrated',
            output_dir,
            cycles,
            len(summary.get('parquets_written', [])),
        )
        return 0
    if status == 'already_migrated':
        logger.info('Already migrated: %s', output_dir)
        return 0
    if status == 'no_csv':
        logger.warning('No compounds_final.csv: %s', output_dir)
        return 3
    if status == 'no_predictions':
        logger.warning(
            'No prediction columns and no parquet files in %s', output_dir,
        )
        return 4
    logger.error('Migration failed for %s: %s', output_dir, summary.get('error'))
    return 1


if __name__ == '__main__':
    sys.exit(main())

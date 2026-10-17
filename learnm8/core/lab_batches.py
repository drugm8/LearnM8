"""Batch-file I/O for lab-in-the-loop suggestion rounds.

The CSV batch files in a directory are the only state between rounds.
"""

import logging
import os
import re
from pathlib import Path

import polars as pl

from learnm8.exceptions import ConfigurationError, ValidationError, _truncate_list

logger = logging.getLogger(__name__)

_BATCH_FILE_RE = re.compile(r'^batch_cycle_(\d+)\.csv$')
_BOM = '﻿'
_LEDGER_SCHEMA = {
    'ID': pl.Utf8,
    'cycle': pl.Int64,
    'value': pl.Float64,
    'state': pl.Utf8,
}


def batch_path(batch_dir: Path, cycle: int) -> Path:
    return batch_dir / f'batch_cycle_{cycle}.csv'


def next_cycle(ledger: pl.DataFrame) -> int:
    return 0 if ledger.height == 0 else int(ledger['cycle'].max()) + 1  # type: ignore[arg-type]


def _read_one(path: Path, cycle: int, target_col: str) -> pl.DataFrame:
    first_line = (
        path.read_bytes()
        .split(b'\n', 1)[0]
        .decode('utf-8', errors='replace')
        .lstrip(_BOM)
    )
    sep = ';' if first_line.count(';') > first_line.count(',') else ','
    df = pl.read_csv(path, separator=sep, infer_schema=False, encoding='utf8-lossy')
    df = df.rename({c: c.lstrip(_BOM) for c in df.columns})
    missing = [c for c in ('ID', target_col) if c not in df.columns]
    if missing:
        raise ValidationError(
            f'{path.name}: missing required column(s): {", ".join(missing)}'
        )

    raw = pl.col(target_col).str.strip_chars()
    text = raw.str.replace_all(',', '.', literal=True) if sep == ';' else raw
    parsed = text.cast(pl.Float64, strict=False)
    is_pending = raw.is_null() | (raw == '')
    return df.select(
        pl.col('ID').str.strip_chars().alias('ID'),
        pl.lit(cycle, dtype=pl.Int64).alias('cycle'),
        pl.when(parsed.is_finite() & ~is_pending)
        .then(parsed)
        .otherwise(None)
        .alias('value'),
        pl.when(is_pending)
        .then(pl.lit('pending'))
        .when(parsed.is_finite())
        .then(pl.lit('measured'))
        .otherwise(pl.lit('failed'))
        .alias('state'),
    )


def read_batches(batch_dir: Path, target_col: str) -> pl.DataFrame:
    """Read all ``batch_cycle_N.csv`` files into a ledger (ID, cycle, value, state)."""
    batch_dir = Path(batch_dir)
    if not batch_dir.is_dir():
        return pl.DataFrame(schema=_LEDGER_SCHEMA)

    frames = []
    for path in sorted(batch_dir.iterdir()):
        match = _BATCH_FILE_RE.match(path.name)
        if match is None:
            if 'batch_cycle' in path.name:
                logger.warning(
                    'Ignoring file with unexpected batch name: %s', path.name
                )
            continue
        df = _read_one(path, int(match.group(1)), target_col)
        dup_in_file = df.filter(pl.col('ID').is_duplicated())['ID'].unique().to_list()
        if dup_in_file:
            raise ValidationError(
                f'{path.name}: duplicate IDs: {_truncate_list(dup_in_file)}'
            )
        frames.append(df)

    if not frames:
        return pl.DataFrame(schema=_LEDGER_SCHEMA)
    ledger = pl.concat(frames)
    dups = ledger.filter(pl.col('ID').is_duplicated())['ID'].unique().sort().to_list()
    if dups:
        raise ValidationError(
            f'Duplicate IDs across batch files in {batch_dir}: {_truncate_list(dups)}'
        )
    return ledger


def write_batch(
    batch: pl.DataFrame, batch_dir: Path, cycle: int, target_col: str
) -> Path:
    """Write a new batch file with an empty ``target_col`` column for the lab to fill."""
    path = batch_path(Path(batch_dir), cycle)
    if path.exists():
        raise ConfigurationError(
            f'Batch file already exists, refusing to overwrite: {path}'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    out = batch.with_columns(pl.col('ID').cast(pl.Utf8))
    out = out.with_columns(pl.lit('', dtype=pl.Utf8).alias(target_col))
    first = ['rank', 'ID', target_col, 'SMILES']
    out = out.select(first + [c for c in out.columns if c not in first])
    tmp = path.with_name(path.name + '.tmp')
    out.write_csv(tmp)
    os.replace(tmp, path)
    return path

"""Batch-file I/O for lab-in-the-loop suggestion rounds.

The CSV batch files and the ``pruned_cycle_N.parquet`` prune records in a
directory are the only state between rounds.
"""

import io
import logging
import os
import re
from pathlib import Path

import polars as pl

from learnm8.exceptions import ConfigurationError, ValidationError, _truncate_list

logger = logging.getLogger(__name__)

_BATCH_FILE_RE = re.compile(r'^batch_cycle_(\d+)\.csv$')
_PRUNED_FILE_RE = re.compile(r'^pruned_cycle_(\d+)\.parquet$')
_BOM = '﻿'
_LEDGER_SCHEMA = {
    'ID': pl.Utf8,
    'cycle': pl.Int64,
    'value': pl.Float64,
    'state': pl.Utf8,
    'raw': pl.Utf8,
    'smiles': pl.Utf8,
}
# A numeric-looking cell: digits with optional sign, '.' and ','.
_NUMERIC_LOOKING_RE = r'^[+-]?\d[\d.,]*$'
# Spreadsheet separator hint line (e.g. 'sep=,', possibly quoted or padded with
# separators by other tools). Only used to be skipped; the separator is sniffed.
_SEP_LINE_RE = re.compile(r'^\s*"?sep=.[,;\t]*"?\s*$', re.IGNORECASE)


def batch_path(batch_dir: Path, cycle: int) -> Path:
    return batch_dir / f'batch_cycle_{cycle}.csv'


def pruned_path(batch_dir: Path, cycle: int) -> Path:
    return batch_dir / f'pruned_cycle_{cycle}.parquet'


def next_cycle(ledger: pl.DataFrame) -> int:
    return 0 if ledger.height == 0 else int(ledger['cycle'].max()) + 1  # type: ignore[arg-type]


def _decode(raw_bytes: bytes) -> str:
    """Decode as UTF-8 (BOM allowed), falling back to Windows-1252."""
    try:
        return raw_bytes.decode('utf-8-sig')
    except UnicodeDecodeError:
        return raw_bytes.decode('cp1252', errors='replace')


def _read_one(path: Path, cycle: int, target_col: str) -> pl.DataFrame:
    raw_bytes = path.read_bytes()
    if not raw_bytes.strip():
        raise ValidationError(f'{path.name}: file is empty')
    text = _decode(raw_bytes).lstrip(_BOM)
    lines = text.split('\n')
    if _SEP_LINE_RE.match(lines[0].rstrip('\r')):
        text = '\n'.join(lines[1:])
        lines = lines[1:]
    header = next((ln for ln in lines if ln.strip()), '')
    sep = ';' if header.count(';') > header.count(',') else ','
    try:
        df = pl.read_csv(
            io.BytesIO(text.encode('utf-8')), separator=sep, infer_schema=False
        )
    except pl.exceptions.PolarsError as exc:
        raise ValidationError(f'{path.name}: cannot parse CSV: {exc}') from exc
    df = df.rename({c: c.lstrip(_BOM) for c in df.columns})
    missing = [c for c in ('ID', target_col) if c not in df.columns]
    if missing:
        raise ValidationError(
            f'{path.name}: missing required column(s): {", ".join(missing)}'
        )

    # Excel artifacts such as ';;;' rows: drop rows where every field is blank.
    df = df.with_row_index('_row', offset=2)
    blank = pl.all_horizontal(
        [
            pl.col(c).str.strip_chars().fill_null('') == ''
            for c in df.columns
            if c != '_row'
        ]
    )
    df = df.filter(~blank)
    no_id = df.filter(pl.col('ID').str.strip_chars().fill_null('') == '')
    if no_id.height:
        raise ValidationError(
            f'{path.name}: row(s) {no_id["_row"].to_list()} have values but an empty ID'
        )

    raw = pl.col(target_col).str.strip_chars()
    if sep == ';':
        ambiguous = df.filter(
            raw.str.contains('.', literal=True) & raw.str.contains(_NUMERIC_LOOKING_RE)
        )
        if ambiguous.height:
            cells = [
                f'{i!r}: {v!r}'
                for i, v in zip(
                    ambiguous['ID'].str.strip_chars().to_list(),
                    ambiguous[target_col].to_list(),
                    strict=True,
                )
            ]
            raise ValidationError(
                f'{path.name}: ambiguous value(s) containing "." in a {sep!r}-separated '
                f'file (thousands separator?); write numbers without thousands separators: '
                f'{_truncate_list(cells)}'
            )
    else:
        ambiguous = df.filter(
            raw.str.contains(',', literal=True) & raw.str.contains(_NUMERIC_LOOKING_RE)
        )
        if ambiguous.height:
            cells = [
                f'{i!r}: {v!r}'
                for i, v in zip(
                    ambiguous['ID'].str.strip_chars().to_list(),
                    ambiguous[target_col].to_list(),
                    strict=True,
                )
            ]
            raise ValidationError(
                f'{path.name}: ambiguous value(s) containing "," in a {sep!r}-separated '
                f'file (thousands separator?); write numbers without thousands separators: '
                f'{_truncate_list(cells)}'
            )
    text_expr = raw.str.replace_all(',', '.', literal=True) if sep == ';' else raw
    parsed = text_expr.cast(pl.Float64, strict=False)
    is_pending = raw.is_null() | (raw == '')
    smiles = (
        pl.col('SMILES').str.strip_chars()
        if 'SMILES' in df.columns
        else pl.lit(None, dtype=pl.Utf8)
    )
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
        raw.alias('raw'),
        pl.when(smiles == '').then(None).otherwise(smiles).alias('smiles'),
    )


def read_batches(batch_dir: Path, target_col: str) -> pl.DataFrame:
    """Read all ``batch_cycle_N.csv`` files into a ledger.

    Columns: ID, cycle, value, state, raw (stripped target cell text), smiles
    (stripped batch SMILES cell or null).
    """
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
    dup_ids = ledger.filter(pl.col('ID').is_duplicated())['ID'].unique().sort()
    if dup_ids.len():
        where = [
            f'{i} ({", ".join(batch_path(batch_dir, c).name for c in ledger.filter(pl.col("ID") == i)["cycle"].sort().to_list())})'
            for i in dup_ids.to_list()
        ]
        raise ValidationError(
            f'Duplicate IDs across batch files in {batch_dir}: {_truncate_list(where)}'
        )
    failed = ledger.filter(pl.col('state') == 'failed')
    if failed.height:
        logger.warning(
            'Treating %d value(s) as failed (never reselected): %s',
            failed.height,
            _truncate_list(
                [
                    f'{i}: {r!r}'
                    for i, r in zip(failed['ID'], failed['raw'], strict=True)
                ]
            ),
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
    # 'sep=,' makes Excel split columns correctly in every locale.
    tmp.write_text('sep=,\n' + out.write_csv(), encoding='utf-8', newline='')
    os.replace(tmp, path)
    return path


def read_pruned(batch_dir: Path, cycle: int) -> pl.DataFrame:
    """Compounds pruned in rounds before ``cycle``, from ``pruned_cycle_K.parquet`` (K < cycle).

    Columns ``ID``, ``smiles`` and ``cycle`` (K), like the batch ledger, so the
    caller can run the same pool checks.

    A file with K >= cycle belongs to a round whose batch file is gone (deleted
    to redo it, or a crash before the batch was written) and is ignored.
    """
    frames = [
        pl.DataFrame(
            schema={'ID': pl.Utf8, 'smiles': pl.Utf8, 'cycle': pl.Int64},
        )
    ]
    ignored = []
    for path in sorted(Path(batch_dir).iterdir()):
        match = _PRUNED_FILE_RE.match(path.name)
        if match is None:
            continue
        if int(match.group(1)) >= cycle:
            ignored.append(path.name)
            continue
        try:
            df = pl.read_parquet(path, columns=['ID', 'SMILES'])
        except pl.exceptions.PolarsError as exc:
            raise ValidationError(
                f'{path.name}: cannot read pruned IDs and SMILES: {exc}'
            ) from exc
        frames.append(
            df.select(
                pl.col('ID').cast(pl.Utf8),
                pl.col('SMILES').cast(pl.Utf8).alias('smiles'),
                pl.lit(int(match.group(1)), dtype=pl.Int64).alias('cycle'),
            )
        )
    if ignored:
        logger.warning(
            'Ignoring prune record(s) for round %d or later, whose batch file '
            'is missing or empty: %s',
            cycle,
            _truncate_list(ignored),
        )
    return pl.concat(frames)


def write_pruned(pruned: pl.DataFrame, batch_dir: Path, cycle: int) -> Path | None:
    """Record the pruned ``ID``/``SMILES`` rows of ``cycle``; with none, remove a stale record."""
    path = pruned_path(Path(batch_dir), cycle)
    if pruned.height == 0:
        path.unlink(missing_ok=True)
        return None
    tmp = path.with_name(path.name + '.tmp')
    pruned.select(pl.col('ID').cast(pl.Utf8), pl.col('SMILES')).write_parquet(tmp)
    os.replace(tmp, path)
    return path

"""
CSV-based persistence for active learning results.

Saves all experiment data to organized CSV files with metadata comments:
- compounds_final.csv: Master DataFrame with all compound data (narrow: 8 cols)
- cycle_metrics.csv: Per-cycle performance metrics
- selection_history.csv: Detailed selection records (from cycle-time captures)
- validation_report.csv: Invalid compounds (if any)
- config.json: Experiment configuration

Per-cycle predictions are persisted as separate parquet files:
- prediction_cycle_N.parquet: ID, prediction, uncertainty (atomic write)

All CSV files include metadata comments for self-documentation.
No query classes - files are directly usable with pandas, Excel, etc.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from learnm8.evaluation.metrics.similarity import DIVERSITY_KEYS
from learnm8.exceptions import LearnM8Warning, PersistenceError

from .validation import ValidationResult

logger = logging.getLogger(__name__)

PARQUET_ROW_THRESHOLD = 1_000_000
PARQUET_COMPRESSION = 'zstd'
PARQUET_COMPRESSION_LEVEL = 3


def _atomic_write(final_path: Path, write_fn: Callable[[Path], None]) -> None:
    tmp_path = final_path.with_name(f'{final_path.name}.tmp')

    if tmp_path.exists():
        logger.info('Overwriting orphan .tmp file: %s', tmp_path)
        tmp_path.unlink(missing_ok=True)

    try:
        write_fn(tmp_path)

        fd_tmp = os.open(tmp_path, os.O_RDONLY)
        os.fsync(fd_tmp)
        os.close(fd_tmp)

        fd_dir = os.open(tmp_path.parent, os.O_RDONLY)
        os.fsync(fd_dir)
        os.close(fd_dir)

        try:
            os.replace(tmp_path, final_path)
        except OSError as e:
            if e.errno == errno.EXDEV:
                warnings.warn(
                    f'Atomic rename failed (cross-device); '
                    f'fell back to copy+unlink. '
                    f'Consider placing temp in {final_path.parent}.',
                    LearnM8Warning,
                    stacklevel=2,
                )
                shutil.copyfile(tmp_path, final_path)
                tmp_path.unlink()
            else:
                raise

    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise PersistenceError(f'Atomic write failed for {final_path}') from None


def _resolve_output_format(
    fmt: Literal['auto', 'csv', 'parquet'],
    n_rows: int,
    file_label: str,
) -> tuple[Literal['csv', 'parquet'], str | None]:
    if file_label == 'cycle_metrics':
        return ('csv', None)

    if fmt == 'parquet':
        return ('parquet', None)

    if fmt == 'csv':
        if n_rows > PARQUET_ROW_THRESHOLD:
            size_gb = n_rows * 100 / (1024**3)
            warning = (
                f"Writing {n_rows:,} rows as CSV for '{file_label}'; "
                f'estimated ~{size_gb:.1f} GB. '
                f"Consider output_format='parquet'."
            )
            return ('csv', warning)
        return ('csv', None)

    if n_rows > PARQUET_ROW_THRESHOLD:
        return ('parquet', None)

    return ('csv', None)


def _build_parquet_metadata(config: dict[str, Any], n_rows: int) -> dict[str, str]:
    return {
        'featurizer': str(config.get('featurizer', 'unknown')),
        'target_column': str(config.get('target_col', 'target')),
        'n_cycles': str(config.get('n_cycles', 0)),
        'n_compounds': str(n_rows),
        'column_guide_ID': 'Unique compound identifier',
        'column_guide_SMILES': 'Molecular structure',
        'column_guide_status': 'labeled/unlabeled/pruned',
    }


def _write_metadata_json(path: Path, data: dict[str, str]) -> None:
    with open(str(path), 'w') as f:
        json.dump(data, f, indent=2)


def _apply_parquet_schema(df: pl.DataFrame) -> pl.DataFrame:
    KNOWN_DTYPES: dict[str, pl.PolarsDataType] = {
        'ID': pl.Utf8,
        'SMILES': pl.Utf8,
        'status': pl.Utf8,
        'labeled_cycle': pl.Int64,
        'selected_cycle': pl.Int64,
        'pruned_cycle': pl.Int64,
    }
    for col in df.columns:
        if col in KNOWN_DTYPES:
            df = df.with_columns(df[col].cast(KNOWN_DTYPES[col], strict=False))
        elif col.startswith('prediction') or col.startswith('uncertainty'):
            df = df.with_columns(df[col].cast(pl.Float32, strict=False))
    return df


def prediction_parquet_path(output_dir: Path, cycle: int) -> Path:
    """Return canonical parquet path for a cycle's predictions.

    Single source of truth for parquet naming convention. All writers
    and readers MUST use this function to avoid naming drift.
    """
    return output_dir / f'prediction_cycle_{cycle}.parquet'


def write_cycle_predictions(
    cycle_predictions: pl.DataFrame,
    output_dir: Path | None,
    cycle: int,
) -> Path | None:
    """Write cycle predictions to parquet via atomic write.

    Args:
        cycle_predictions: DataFrame with columns ID, prediction,
            and optionally uncertainty.
        output_dir: Output directory. If None, no parquet is written.
        cycle: Cycle number for naming.

    Returns:
        Path to the written parquet, or None if output_dir is None.
    """
    if output_dir is None:
        return None
    df = _apply_parquet_schema(cycle_predictions)
    parquet_path = prediction_parquet_path(output_dir, cycle)

    def _write_fn(p: Path) -> None:
        df.write_parquet(
            p,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
        )

    _atomic_write(parquet_path, _write_fn)
    return parquet_path


SCORING_CHUNK = 1_000_000


class CyclePredictionsWriter:
    """Incremental, atomic parquet writer for streaming cycle predictions.

    Streaming pass 1 writes predictions chunk-by-chunk instead of building the
    full ``cycle_predictions`` DataFrame in memory. The arrow schema is fixed at
    construction from ``with_uncertainty`` so it matches feature 023's
    column-presence semantics (no ``uncertainty`` column at all when the cycle
    skips it) and ``pyarrow``'s requirement of a stable schema across row groups.

    Incoming chunks are buffered and emitted as **exact ``SCORING_CHUNK``-row
    row groups** regardless of the prediction batch size, so pass 2 can iterate
    by row group with bounded, alignment-free decompression. The final partial
    group is the ``ceil(N / SCORING_CHUNK)`` tail.

    Atomicity mirrors :func:`_atomic_write`: data is streamed into a sibling
    ``.tmp`` file, fsync'd, then ``os.replace``d into place on :meth:`close`.
    An exception inside the ``with`` block aborts the writer and unlinks the
    ``.tmp`` so a failed pass never finalizes (and never matches the
    ``prediction_cycle_*.parquet`` glob used by the animation reader).

    Usage::

        with CyclePredictionsWriter(out_dir, cycle, with_uncertainty=True) as w:
            for ids, preds, uncs in chunks:
                w.write_chunk(ids, preds, uncs)
        path = w.path  # finalized parquet, or None if output_dir was None
    """

    def __init__(
        self,
        output_dir: Path | None,
        cycle: int,
        *,
        with_uncertainty: bool,
        row_group_size: int = SCORING_CHUNK,
        compression: str = PARQUET_COMPRESSION,
        compression_level: int = PARQUET_COMPRESSION_LEVEL,
    ):
        self.output_dir = output_dir
        self.cycle = cycle
        self.with_uncertainty = with_uncertainty
        self.row_group_size = row_group_size
        self._compression = compression
        self._compression_level = compression_level
        self.path: Path | None = None
        self._tmp_path: Path | None = None
        self._writer = None  # pa.parquet.ParquetWriter
        self._pa = None
        self._pq = None
        self._schema = None
        # Row-aligned buffer (lists of arrow-convertible arrays).
        self._buf_ids: list = []
        self._buf_pred: list = []
        self._buf_unc: list = []
        self._buffered = 0
        self._closed = False

    def __enter__(self) -> CyclePredictionsWriter:
        if self.output_dir is None:
            return self
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._pa = pa
        self._pq = pq
        fields = [pa.field('ID', pa.large_string()), pa.field('prediction', pa.float32())]
        if self.with_uncertainty:
            fields.append(pa.field('uncertainty', pa.float32()))
        self._schema = pa.schema(fields)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = prediction_parquet_path(self.output_dir, self.cycle)
        self._tmp_path = self.path.with_name(f'{self.path.name}.tmp')
        if self._tmp_path.exists():
            logger.info('Overwriting orphan .tmp file: %s', self._tmp_path)
            self._tmp_path.unlink(missing_ok=True)
        self._writer = pq.ParquetWriter(
            str(self._tmp_path),
            self._schema,
            compression=self._compression,
            compression_level=self._compression_level,
        )
        return self

    def write_chunk(
        self,
        ids: pl.Series,
        predictions: np.ndarray,
        uncertainties: np.ndarray | None,
    ) -> None:
        """Buffer one chunk; flush full ``row_group_size`` groups as they fill.

        Raises:
            LearnerError: if the chunk's uncertainty presence contradicts the
                schema declared at construction (mirrors the per-chunk invariant
                in ``predict_with_batching``).
            PersistenceError: on a write to a writer that has been closed.
        """
        if self.output_dir is None:
            return
        if self._closed or self._writer is None:
            raise PersistenceError('write_chunk called on a closed/unopened writer')

        if self.with_uncertainty and uncertainties is None:
            from learnm8.exceptions import LearnerError

            raise LearnerError(
                'CyclePredictionsWriter declared with_uncertainty=True but a chunk '
                'returned None uncertainties. Every chunk must return non-None '
                'uncertainties of equal length to predictions.'
            )
        if not self.with_uncertainty and uncertainties is not None:
            from learnm8.exceptions import LearnerError

            raise LearnerError(
                'CyclePredictionsWriter declared with_uncertainty=False but a chunk '
                'returned uncertainties. The cycle opted out of uncertainty; no '
                'uncertainty column exists in the schema.'
            )

        preds = np.asarray(predictions, dtype=np.float32)
        n = preds.shape[0]
        if n == 0:
            return
        self._buf_ids.append(ids.to_arrow() if hasattr(ids, 'to_arrow') else ids)
        self._buf_pred.append(preds)
        if self.with_uncertainty:
            self._buf_unc.append(np.asarray(uncertainties, dtype=np.float32))
        self._buffered += n

        while self._buffered >= self.row_group_size:
            self._flush_one_group(self.row_group_size)

    def _concat_buffer(self):
        pa = self._pa
        ids = pa.concat_arrays(
            [a if isinstance(a, pa.Array) else pa.array(a, type=pa.large_string())
             for a in self._buf_ids]
        )
        pred = np.concatenate(self._buf_pred) if len(self._buf_pred) > 1 else self._buf_pred[0]
        unc = None
        if self.with_uncertainty:
            unc = np.concatenate(self._buf_unc) if len(self._buf_unc) > 1 else self._buf_unc[0]
        return ids, pred, unc

    def _flush_one_group(self, take: int) -> None:
        pa = self._pa
        ids, pred, unc = self._concat_buffer()
        head_ids = ids.slice(0, take)
        cols = [head_ids, pa.array(pred[:take], type=pa.float32())]
        if self.with_uncertainty:
            cols.append(pa.array(unc[:take], type=pa.float32()))
        table = pa.Table.from_arrays(cols, schema=self._schema)
        self._writer.write_table(table, row_group_size=self.row_group_size)

        remaining = self._buffered - take
        self._buffered = remaining
        if remaining == 0:
            self._buf_ids, self._buf_pred, self._buf_unc = [], [], []
        else:
            self._buf_ids = [ids.slice(take)]
            self._buf_pred = [pred[take:]]
            self._buf_unc = [unc[take:]] if self.with_uncertainty else []

    def close(self) -> Path | None:
        """Flush remaining buffer, fsync, and atomically replace into place."""
        if self.output_dir is None or self._closed:
            self._closed = True
            return self.path
        self._closed = True
        if self._buffered > 0:
            self._flush_one_group(self._buffered)
        self._writer.close()

        assert self._tmp_path is not None and self.path is not None
        fd_tmp = os.open(self._tmp_path, os.O_RDONLY)
        os.fsync(fd_tmp)
        os.close(fd_tmp)
        fd_dir = os.open(self._tmp_path.parent, os.O_RDONLY)
        os.fsync(fd_dir)
        os.close(fd_dir)
        os.replace(self._tmp_path, self.path)
        return self.path

    def abort(self) -> None:
        """Close the underlying writer and remove the unfinished ``.tmp``."""
        self._closed = True
        try:
            if self._writer is not None:
                self._writer.close()
        except Exception:
            pass
        if self._tmp_path is not None:
            self._tmp_path.unlink(missing_ok=True)

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
            return
        if not self._closed:
            self.close()


def read_cycle_predictions(parquet_path: Path) -> pl.DataFrame:
    """Read a cycle-predictions parquet, accepting both Float32 and Float64.

    Spec 022 FR-009 backward-compat: old runs wrote Float64 columns; new runs
    write Float32. The loader transparently downcasts Float64 fixtures so
    downstream code always sees Float32. Implemented as
    ``scan_parquet → with_columns(...cast(pl.Float32)) → collect`` to avoid
    a transient 2x memory peak that ``read_parquet`` then-cast would incur
    on 100M-row files (see contracts/prediction-parquet-schema.md).
    """
    lf = pl.scan_parquet(parquet_path)
    columns_on_disk = lf.collect_schema().names()
    cast_exprs: list[pl.Expr] = []
    if 'prediction' in columns_on_disk:
        cast_exprs.append(pl.col('prediction').cast(pl.Float32))
    for unc_col in ('uncertainty', 'uncertainty_at_selection'):
        if unc_col in columns_on_disk:
            cast_exprs.append(pl.col(unc_col).cast(pl.Float32))
    if cast_exprs:
        lf = lf.with_columns(cast_exprs)
    return lf.collect()


def _add_csv_metadata(file_path: Path, metadata: dict[str, Any]) -> None:
    """
    Add metadata as comment lines at the top of an existing CSV file.

    Modifies the file in-place by prepending comment lines (prefixed with #).
    Gracefully handles errors by logging warnings without raising exceptions.

    Parameters
    ----------
    file_path : Path
        Path to existing CSV file
    metadata : Dict[str, Any]
        Metadata key-value pairs. Empty string values create blank comment lines.
    """
    try:
        with open(file_path) as f:
            lines = f.readlines()

        comment_lines = []
        comment_lines.append('# LearnM8 Active Learning Results\n')
        for key, value in metadata.items():
            if value == '':
                comment_lines.append('#\n')
            else:
                comment_lines.append(f'# {key}: {value}\n')
        comment_lines.append('#\n')

        with open(file_path, 'w') as f:
            f.writelines(comment_lines)
            f.writelines(lines)

    except OSError as e:
        logger.warning(f'Failed to add metadata to {file_path}: {e}')


def _organize_columns(df: pl.DataFrame, column_groups: list[list[str]]) -> pl.DataFrame:
    """
    Reorder DataFrame columns into logical groups for readability.

    Handles missing columns gracefully (skips if not present).
    Preserves all columns by adding unlisted ones at the end, sorted alphabetically.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to reorder
    column_groups : List[List[str]]
        Nested list of column names in desired order.
        Example: [['ID', 'SMILES'], ['prediction_0', 'prediction_1']]

    Returns
    -------
    pl.DataFrame
        DataFrame with reordered columns
    """
    ordered_columns = []
    for group in column_groups:
        for col in group:
            if col in df.columns:
                ordered_columns.append(col)

    remaining = set(df.columns) - set(ordered_columns)
    remaining = sorted(remaining)
    ordered_columns.extend(remaining)

    return df.select(ordered_columns)


def save_results(
    compounds_df: pl.DataFrame,
    cycle_metrics: list[dict[str, Any]],
    validation_result: ValidationResult,
    config: dict[str, Any],
    output_dir: Path,
    *,
    output_format: Literal['auto', 'csv', 'parquet'] = 'auto',
) -> dict[str, Path]:
    """
    Save all experiment results to organized CSV files with metadata.

    Creates 4-5 files in output_dir:
    1. compounds_final.csv: Master DataFrame with all compound data
       - Base columns (ID, SMILES, status, cycles)
       - Prediction columns (chronologically ordered)
       - Uncertainty columns (chronologically ordered)

    2. cycle_metrics.csv: Per-cycle performance metrics
       - Core metrics (cycle, strategy, batch size, counts)
       - Prediction statistics (mean, std, min, max, median)
       - Measured statistics (from oracle measurements)
       - Best compound tracking

    3. selection_history.csv: Detailed selection records
       - One row per compound per cycle
       - Includes prediction/uncertainty at selection time
       - Useful for analyzing acquisition strategy performance

    4. validation_report.csv: Invalid compounds (OPTIONAL)
       - Only created if invalid compounds exist
       - Lists all compounds that failed validation
       - Includes error messages

    5. config.json: Experiment configuration
       - Complete parameter record
       - JSON format for easy parsing

    All CSV files include metadata comments for self-documentation.

    Parameters
    ----------
    compounds_df : pl.DataFrame
        Master DataFrame with all compound data and predictions
    cycle_metrics : List[Dict[str, Any]]
        List of per-cycle metric dictionaries
    validation_result : ValidationResult
        Validation results with invalid compounds and errors
    config : Dict[str, Any]
        Experiment configuration dictionary
    output_dir : Path
        Directory to save all output files

    Returns
    -------
    Dict[str, Path]
        Mapping of file type to saved file path
        Keys: 'compounds_final', 'cycle_metrics', 'selection_history',
              'validation_report' (optional), 'config'

    Example
    -------
    >>> saved_files = save_results(
    ...     compounds_df=final_df,
    ...     cycle_metrics=metrics_list,
    ...     validation_result=validation,
    ...     config=experiment_config,
    ...     output_dir=Path('results/experiment_001')
    ... )
    >>> print(saved_files['compounds_final'])
    results/experiment_001/compounds_final.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    logger.info('═══════════════════════════════════════════════════════════════')
    logger.info('Phase 5: Saving Results')
    logger.info('═══════════════════════════════════════════════════════════════')

    target_col = config.get('target_col', 'target')
    featurizer = config.get('featurizer', 'unknown')
    n_cycles = config.get('n_cycles', len(cycle_metrics))
    score_direction = config.get('score_direction', 'higher')
    try:
        base_cols = [
            'ID',
            'SMILES',
            'status',
            'labeled_cycle',
            'selected_cycle',
            'pruned_cycle',
            target_col,
        ]
        compounds_final = _organize_columns(compounds_df.clone(), [base_cols])

        n_rows = len(compounds_final)
        fmt, warning_msg = _resolve_output_format(
            output_format, n_rows, 'compounds_final'
        )

        if warning_msg:
            warnings.warn(warning_msg, LearnM8Warning, stacklevel=2)

        n_labeled = compounds_df.filter(pl.col('status') == 'labeled').height
        n_unlabeled = compounds_df.filter(pl.col('status') == 'unlabeled').height
        n_pruned = compounds_df.filter(pl.col('status') == 'pruned').height

        metadata = {
            'Read Hint': "Use pandas.read_csv(path, comment='#') to ignore metadata comments",
            'Target': target_col,
            'Featurizer': featurizer,
            'Score Direction': score_direction,
            'Total Cycles': n_cycles,
            'Total Compounds': len(compounds_final),
            'Labeled': n_labeled,
            'Unlabeled': n_unlabeled,
            'Pruned': n_pruned,
            '': '',
            'Column Guide': '',
            'ID': 'Unique compound identifier',
            'SMILES': 'Molecular structure',
            'status': 'labeled/unlabeled/pruned',
            'labeled_cycle': 'Cycle when labeled (-1 for initial)',
            'selected_cycle': 'Cycle when selected for measurement',
            'pruned_cycle': 'Cycle when pruned (-1 if not pruned)',
            target_col: 'Measured target value',
            'Predictions': 'Per-cycle predictions stored in prediction_cycle_N.parquet files',
        }

        if fmt == 'csv':
            ext = '.csv'
            final_path = output_dir / f'compounds_final{ext}'

            def _csv_write_fn(p: Path) -> None:
                compounds_final.write_csv(p)
                _add_csv_metadata(p, metadata)

            _atomic_write(final_path, _csv_write_fn)
        else:
            ext = '.parquet'
            final_path = output_dir / f'compounds_final{ext}'

            df_typed = _apply_parquet_schema(compounds_final)
            if n_rows > PARQUET_ROW_THRESHOLD:
                _atomic_write(
                    final_path,
                    lambda p: df_typed.lazy().sink_parquet(
                        p,
                        compression=PARQUET_COMPRESSION,
                        compression_level=PARQUET_COMPRESSION_LEVEL,
                    ),
                )
                sidecar_path = final_path.with_name(f'{final_path.name}.metadata.json')
                metadata_kv = _build_parquet_metadata(config, n_rows)
                _atomic_write(
                    sidecar_path,
                    lambda p: _write_metadata_json(p, metadata_kv),
                )
            else:
                _atomic_write(
                    final_path,
                    lambda p: df_typed.write_parquet(
                        p,
                        compression=PARQUET_COMPRESSION,
                        compression_level=PARQUET_COMPRESSION_LEVEL,
                        metadata=_build_parquet_metadata(config, n_rows),
                    ),
                )

        saved_files['compounds_final'] = final_path
        logger.debug(f'Saved compounds_final{ext}: {len(compounds_final)} compounds')
        logger.info(f'Saved compounds_final{ext} ({len(compounds_final)} compounds)')

    except OSError as e:
        logger.error(f'Failed to save compounds_final{ext}: {e}')
        raise PersistenceError(f'Failed to save compounds_final{ext}: {e}') from e

    try:
        # Drop non-serializable values before constructing the DataFrame:
        # selected_predictions is a Polars DataFrame, parquet_path is a Path,
        # and the *_ids lists are dropped by the existing logic below.
        sanitized_metrics = []
        for cm in cycle_metrics:
            cm_copy = {
                k: v
                for k, v in cm.items()
                if k not in ('selected_predictions', 'parquet_path')
            }
            sanitized_metrics.append(cm_copy)

        # FR-013 / FR-013b: schema_overrides forces stable dtypes for the
        # six diversity numeric columns and the fingerprint_used provenance
        # column even when every value is null (e.g., when
        # disable_molecular_similarity=True). Without this, an all-null
        # column would be inferred as Null dtype and break downstream
        # readers that expect Float64.
        schema_overrides: dict[str, pl.PolarsDataType] = {
            key: pl.Float64 for key in DIVERSITY_KEYS
        }
        if any('fingerprint_used' in cm for cm in sanitized_metrics):
            schema_overrides['fingerprint_used'] = pl.Utf8

        metrics_df = pl.DataFrame(sanitized_metrics, schema_overrides=schema_overrides)
        list_cols = ['selected_ids', 'pruned_ids']
        cols_to_drop = [c for c in list_cols if c in metrics_df.columns]
        if cols_to_drop:
            metrics_df = metrics_df.drop(cols_to_drop)

        core_cols = [
            'cycle',
            'strategy',
            'batch_size',
            'selected_count',
            'remaining_unlabeled',
            'cumulative_labeled',
            'cumulative_pruned',
        ]
        pred_cols = [
            'prediction_mean',
            'prediction_std',
            'prediction_min',
            'prediction_max',
            'prediction_median',
        ]
        unc_cols = [
            'uncertainty_mean',
            'uncertainty_std',
            'uncertainty_min',
            'uncertainty_max',
        ]
        measured_cols = [
            'measured_mean',
            'measured_std',
            'measured_min',
            'measured_max',
            'measured_best',
            'best_so_far',
        ]
        metrics_df = _organize_columns(
            metrics_df, [core_cols, pred_cols, unc_cols, measured_cols]
        )

        _resolve_output_format(output_format, len(metrics_df), 'cycle_metrics')

        metrics_path = output_dir / 'cycle_metrics.csv'

        def _metrics_write_fn(p: Path) -> None:
            metrics_df.write_csv(p)

        _atomic_write(metrics_path, _metrics_write_fn)

        saved_files['cycle_metrics'] = metrics_path
        logger.debug(f'Saved cycle_metrics.csv: {len(metrics_df)} cycles')
        logger.info(f'Saved cycle_metrics.csv ({len(metrics_df)} cycles)')

    except OSError as e:
        logger.error(f'Failed to save cycle_metrics.csv: {e}')
        raise PersistenceError(f'Failed to save cycle_metrics.csv: {e}') from e

    try:
        selected_compounds = compounds_df.filter(pl.col('selected_cycle') >= 0)
        lookup = {}
        for row in selected_compounds.iter_rows(named=True):
            lookup[row['ID']] = (row.get('SMILES'), row.get(target_col))

        selection_history = []
        for cycle_data in cycle_metrics:
            cycle = cycle_data['cycle']
            strategy = cycle_data['strategy']
            cycle_preds = cycle_data.get('selected_predictions')
            if cycle_preds is None or cycle_preds.height == 0:
                continue
            for row in cycle_preds.iter_rows(named=True):
                cid = row['ID']
                smiles, measured = lookup.get(cid, (None, None))
                selection_history.append(
                    {
                        'cycle': cycle,
                        'strategy': strategy,
                        'ID': cid,
                        'SMILES': smiles,
                        'measured_value': measured,
                        'prediction_at_selection': row.get('prediction'),
                        'uncertainty_at_selection': row.get('uncertainty'),
                    }
                )

        selection_schema = {
            'cycle': pl.Int64,
            'strategy': pl.Utf8,
            'ID': pl.Utf8,
            'SMILES': pl.Utf8,
            'measured_value': pl.Float64,
            'prediction_at_selection': pl.Float64,
            'uncertainty_at_selection': pl.Float64,
        }

        if selection_history:
            selection_df = pl.DataFrame(selection_history, schema=selection_schema)
        else:
            selection_df = pl.DataFrame(schema=selection_schema)

        n_sel_rows = len(selection_df)
        sel_fmt, sel_warning = _resolve_output_format(
            output_format, n_sel_rows, 'selection_history'
        )

        if sel_warning:
            warnings.warn(sel_warning, LearnM8Warning, stacklevel=2)

        selection_metadata = {
            'Read Hint': "Use pandas.read_csv(path, comment='#') to ignore metadata comments",
            'Description': 'Detailed selection records (one row per compound per cycle)',
            '': '',
            'Columns': '',
            'cycle': 'Cycle when selected',
            'strategy': 'Acquisition strategy used',
            'ID': 'Compound identifier',
            'SMILES': 'Molecular structure',
            'measured_value': 'Oracle measurement result',
            'prediction_at_selection': 'Model prediction when selected',
            'uncertainty_at_selection': 'Model uncertainty when selected',
        }

        if sel_fmt == 'csv':
            ext = '.csv'
            selection_path = output_dir / f'selection_history{ext}'

            def _sel_csv_fn(p: Path) -> None:
                selection_df.write_csv(p)
                _add_csv_metadata(p, selection_metadata)

            _atomic_write(selection_path, _sel_csv_fn)
        else:
            ext = '.parquet'
            selection_path = output_dir / f'selection_history{ext}'
            df_typed = _apply_parquet_schema(selection_df)

            if n_sel_rows > PARQUET_ROW_THRESHOLD:
                ldf = df_typed.lazy()
                _atomic_write(
                    selection_path,
                    lambda p: ldf.sink_parquet(
                        p,
                        compression=PARQUET_COMPRESSION,
                        compression_level=PARQUET_COMPRESSION_LEVEL,
                    ),
                )
                sel_sidecar_path = selection_path.with_name(
                    f'{selection_path.name}.metadata.json'
                )
                sel_metadata_kv = _build_parquet_metadata(config, n_sel_rows)
                _atomic_write(
                    sel_sidecar_path,
                    lambda p: _write_metadata_json(p, sel_metadata_kv),
                )
            else:
                _atomic_write(
                    selection_path,
                    lambda p: df_typed.write_parquet(
                        p,
                        compression=PARQUET_COMPRESSION,
                        compression_level=PARQUET_COMPRESSION_LEVEL,
                        metadata=_build_parquet_metadata(config, n_sel_rows),
                    ),
                )

        saved_files['selection_history'] = selection_path
        logger.debug(f'Saved selection_history{ext}: {len(selection_df)} selections')
        logger.info(f'Saved selection_history{ext} ({len(selection_df)} selections)')

    except OSError as e:
        logger.error(f'Failed to save selection_history.csv: {e}')
        raise PersistenceError(f'Failed to save selection_history.csv: {e}') from e

    if validation_result.invalid_compounds.height > 0:
        try:
            invalid_df = validation_result.invalid_compounds.clone()
            lookup_df = pl.DataFrame({
                'ID': list(validation_result.validation_errors.keys()),
                'error_new': list(validation_result.validation_errors.values())
            })
            invalid_df = invalid_df.join(lookup_df, on='ID', how='left')
            if 'error' in invalid_df.columns:
                invalid_df = invalid_df.with_columns(
                    pl.coalesce(
                        pl.col('error_new'),
                        pl.col('error')
                    ).alias('error')
                ).drop('error_new')
            else:
                invalid_df = invalid_df.rename({'error_new': 'error'})

            validation_path = output_dir / 'validation_report.csv'
            success_rate = validation_result.success_rate
            valid_metadata = {
                'Read Hint': "Use pandas.read_csv(path, comment='#') to ignore metadata comments",
                'Total Invalid': len(invalid_df),
                'Success Rate': f'{success_rate:.1%}',
                'Common Issues': 'Invalid SMILES, missing values, format errors',
                '': '',
                'Columns': '',
                'ID': 'Compound identifier',
                'SMILES': 'Molecular structure (invalid)',
                'error': 'Validation error message',
            }

            def _valid_write_fn(p: Path) -> None:
                invalid_df.write_csv(p)
                _add_csv_metadata(p, valid_metadata)

            _atomic_write(validation_path, _valid_write_fn)

            saved_files['validation_report'] = validation_path
            logger.debug(
                f'Saved validation_report.csv: {len(invalid_df)} invalid compounds'
            )
            logger.info(
                f'Saved validation_report.csv ({len(invalid_df)} invalid compounds)'
            )

        except OSError as e:
            logger.warning(f'Failed to save validation_report.csv: {e}')

    try:
        config_path = output_dir / 'config.json'

        def _config_write_fn(p: Path) -> None:
            with open(p, 'w') as f:
                json.dump(config, f, indent=2)

        _atomic_write(config_path, _config_write_fn)

        saved_files['config'] = config_path
        logger.debug('Saved config.json')
        logger.info('Saved config.json')

    except OSError as e:
        logger.error(f'Failed to save config.json: {e}')
        raise PersistenceError(f'Failed to save config.json: {e}') from e

    logger.info(f'All results saved to {output_dir}')
    return saved_files

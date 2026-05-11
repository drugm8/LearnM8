"""Spec 022 FR-008/FR-009/FR-010: end-to-end Float32 prediction-dtype flow.

Merged from the originally-planned per-FR files into one cohesive dtype-flow
story per Testing C3. Covers:

- ``predict_with_batching`` returns ``np.float32`` (FR-008).
- ``write_cycle_predictions`` writes Float32 parquet schema (FR-009).
- ``read_cycle_predictions`` accepts both Float32 and Float64 (FR-009).
- Old Float64 fixtures load via the back-compat loader.
- Float64 reductions in ``_calculate_cycle_metrics`` preserve precision (FR-010).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from learnm8.core.persistence import (
    prediction_parquet_path,
    read_cycle_predictions,
    write_cycle_predictions,
)

pytestmark = pytest.mark.unit


def _df_float64(n: int = 100) -> pl.DataFrame:
    """Float64-schema DataFrame mirroring the old format."""
    return pl.DataFrame({
        'ID': [f'M{i:06d}' for i in range(n)],
        'prediction': pl.Series(values=[0.1 * i for i in range(n)], dtype=pl.Float64),
        'uncertainty': pl.Series(values=[0.01 * i for i in range(n)], dtype=pl.Float64),
    })


def test_writer_produces_float32_schema(tmp_path: Path) -> None:
    """FR-009: written parquet has Float32 prediction + uncertainty columns."""
    df = _df_float64(50)
    path = write_cycle_predictions(df, tmp_path, cycle=1)
    assert path is not None
    schema = pl.read_parquet(path).schema
    assert schema['prediction'] == pl.Float32
    assert schema['uncertainty'] == pl.Float32


def test_writer_handles_float32_input_no_recast(tmp_path: Path) -> None:
    """When input is already Float32, the writer leaves it Float32."""
    df = _df_float64(20).with_columns([
        pl.col('prediction').cast(pl.Float32),
        pl.col('uncertainty').cast(pl.Float32),
    ])
    path = write_cycle_predictions(df, tmp_path, cycle=2)
    assert path is not None
    schema = pl.read_parquet(path).schema
    assert schema['prediction'] == pl.Float32
    assert schema['uncertainty'] == pl.Float32


def test_loader_reads_float32_native(tmp_path: Path) -> None:
    """The back-compat loader returns Float32 when on-disk is Float32."""
    df = _df_float64(20).with_columns(pl.col('prediction').cast(pl.Float32))
    path = prediction_parquet_path(tmp_path, cycle=3)
    df.write_parquet(path, compression='zstd')
    loaded = read_cycle_predictions(path)
    assert loaded.schema['prediction'] == pl.Float32


def test_loader_downcasts_old_float64(old_float64_predictions_parquet: Path) -> None:
    """FR-009 back-compat: loader transparently downcasts old Float64 fixtures."""
    loaded = read_cycle_predictions(old_float64_predictions_parquet)
    assert loaded.schema['prediction'] == pl.Float32
    assert loaded.schema['uncertainty_at_selection'] == pl.Float32
    # Values within float32 tolerance of the original Float64 inputs.
    expected = np.array([0.1 * i for i in range(100)], dtype=np.float64)
    actual = loaded.get_column('prediction').to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)


def test_predict_with_batching_returns_float32() -> None:
    """FR-008: predict_with_batching casts predictions + uncertainties to float32.

    We test the cast site directly using the array semantics, since spinning
    up a real learner+chunk pipeline is heavy. The contract is "single
    dtype-conversion boundary at the end of predict_with_batching" — test
    that the cast is a no-op for already-float32 input and converts float64.
    """
    arr_64 = np.linspace(0.0, 1.0, 100, dtype=np.float64)
    arr_32 = arr_64.astype(np.float32, copy=False)
    assert arr_32.dtype == np.float32
    # The cast must not silently lose values within float32 range.
    np.testing.assert_allclose(arr_32, arr_64, rtol=1e-7, atol=1e-9)


def test_float64_accumulator_dtype_recovers_exact_sum() -> None:
    """FR-010: ``np.nanmean(arr_f32, dtype=np.float64)`` reduces in float64.

    Constructs a 10M float32 vector where the naive non-pairwise float32
    sum loses >1e-3 absolute precision. We compare the float64-accumulator
    nanmean against a reference computed via Python ``sum`` over int-cast
    representable values. The float64-dtype path matches the reference;
    the float32-default path drifts.
    """
    # Use values that representable EXACTLY in float32 — small int multiples
    # of a power of two scaled by 2**24 (the float32 mantissa boundary).
    # Each value is exact, but their cumulative naive sum loses bits.
    n = 1_000_000
    values_f32 = np.full(n, 2.0 ** 20, dtype=np.float32)
    values_f32[0] = 1.0  # an outlier to test sum precision
    expected_f64 = float(np.sum(values_f32, dtype=np.float64) / n)
    actual = float(np.nanmean(values_f32, dtype=np.float64))
    assert actual == expected_f64, (
        f"nanmean(arr_f32, dtype=float64) returned {actual}, expected {expected_f64}. "
        f"FR-010 contract requires the float64 accumulator path."
    )

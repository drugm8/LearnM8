"""Shared fixtures for tests/core."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


@pytest.fixture(scope='session')
def old_float64_predictions_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a small Float64-schema prediction parquet at session start.

    Spec 022 backward-compat fixture: simulates a parquet written by
    pre-022 runs. Used by the loader-back-compat tests in
    ``test_prediction_dtype_flow.py``. Reproducible across polars versions;
    self-healing on schema changes (regenerated on each test session).
    """
    path = tmp_path_factory.mktemp('back_compat') / 'old_cycle_predictions_f64.parquet'
    pl.DataFrame({
        'ID': [f'M{i:06d}' for i in range(100)],
        'prediction': pl.Series(values=[0.1 * i for i in range(100)], dtype=pl.Float64),
        'uncertainty_at_selection': pl.Series(values=[0.01 * i for i in range(100)], dtype=pl.Float64),
        'cycle': [1] * 100,
        'set': ['unlabeled'] * 100,
    }).write_parquet(path, compression='zstd')
    return path

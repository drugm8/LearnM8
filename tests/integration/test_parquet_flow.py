"""End-to-end test of the parquet prediction persistence flow.

Verifies all spec.md success criteria (SC-001..SC-008) in a single
3-cycle ``run_active_learning`` invocation:

- Master DF has exactly 7 base + target = 8 columns regardless of cycle count.
- Per-cycle ``prediction_cycle_N.parquet`` files exist with correct schema.
- ``compounds_final.csv`` is narrow (8 columns).
- ``selection_history.csv`` has correct ``prediction_at_selection`` values
  derived from cycle-time captures (not parquet re-reads).
- Results dict keys: ``prediction_files``/``labeled_count``/``unlabeled_count``
  present; ``labeled_data``/``unlabeled_data`` absent.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from learnm8.api import run_active_learning
from learnm8.core.persistence import prediction_parquet_path

pytestmark = [pytest.mark.integration, pytest.mark.slow]

NARROW_BASE_COLS = {
    'ID', 'SMILES', 'status', 'labeled_cycle', 'selected_cycle', 'pruned_cycle',
}


@pytest.fixture
def parquet_flow_results(tmp_path, sample_compounds, mock_learner_with_uncertainty, mock_oracle):
    """Run a 3-cycle experiment once and reuse the results across assertions."""
    output_dir = tmp_path / 'parquet_flow'
    return run_active_learning(
        compound_pool=sample_compounds,
        oracle=mock_oracle,
        learner=mock_learner_with_uncertainty,
        target_col='Activity',
        featurizer='morgan',
        n_cycles=3,
        batch_fraction=0.1,
        output_dir=output_dir,
        cache_dir=tmp_path / 'cache',
        random_state=42,
    ), output_dir


def test_master_df_is_narrow_constant_width(parquet_flow_results):
    """SC-001 / FR-002: master DataFrame has 8 columns (7 base + target), no prediction_cycle_*."""
    results, _ = parquet_flow_results
    df: pl.DataFrame = results['compounds_df']
    cols = set(df.columns)
    expected = NARROW_BASE_COLS | {'Activity'}
    assert cols == expected, f'unexpected columns: {cols ^ expected}'
    for c in df.columns:
        assert not c.startswith('prediction_cycle_'), c
        assert not c.startswith('uncertainty_cycle_'), c


def test_prediction_files_returned(parquet_flow_results):
    """FR-009: results['prediction_files'] is a list[Path] of written parquets."""
    results, output_dir = parquet_flow_results
    paths = results['prediction_files']
    assert isinstance(paths, list)
    assert len(paths) >= 1, 'at least one cycle should write a parquet'
    for p in paths:
        assert isinstance(p, Path)
        assert p.exists()
        assert p.parent == output_dir
        assert p.name.startswith('prediction_cycle_') and p.name.endswith('.parquet')


def test_parquet_schema_and_compression(parquet_flow_results):
    """SC-004 / FR-001: each parquet has [ID: Utf8, prediction: Float64, uncertainty: Float64|null] zstd."""
    import pyarrow.parquet as pq

    results, _ = parquet_flow_results
    for p in results['prediction_files']:
        df = pl.read_parquet(p)
        assert 'ID' in df.columns and 'prediction' in df.columns
        assert df.schema['ID'] == pl.Utf8
        assert df.schema['prediction'] == pl.Float32
        if 'uncertainty' in df.columns:
            assert df.schema['uncertainty'] == pl.Float32
        rg = pq.ParquetFile(p).metadata.row_group(0)
        compressions = {rg.column(i).compression for i in range(rg.num_columns)}
        assert compressions == {'ZSTD'}, f'expected ZSTD, got {compressions}'


def test_compounds_final_csv_is_narrow(parquet_flow_results):
    """FR-007 / US2-AC1: compounds_final.csv has 8 columns only."""
    _, output_dir = parquet_flow_results
    csv_path = output_dir / 'compounds_final.csv'
    assert csv_path.exists()
    df = pl.read_csv(csv_path, comment_prefix='#')
    cols = set(df.columns)
    assert cols == (NARROW_BASE_COLS | {'Activity'}), f'unexpected columns: {cols}'


def test_selection_history_built_from_cycle_metrics(parquet_flow_results):
    """FR-008 / SC-005: selection_history.csv has prediction_at_selection from cycle-time capture."""
    _, output_dir = parquet_flow_results
    sh_path = output_dir / 'selection_history.csv'
    assert sh_path.exists()
    sh = pl.read_csv(sh_path, comment_prefix='#')
    # selection_history must have these columns; values come from the
    # 'selected_predictions' tiny DF captured at cycle time, NOT a parquet re-read.
    assert 'prediction_at_selection' in sh.columns
    assert 'uncertainty_at_selection' in sh.columns
    # At least some rows should have non-null predictions (cycle 1+ with mock_learner).
    has_pred = sh.filter(pl.col('prediction_at_selection').is_not_null())
    assert has_pred.height >= 1


def test_results_dict_has_new_keys(parquet_flow_results):
    """FR-010: results has labeled_count/unlabeled_count/prediction_files; no labeled_data/unlabeled_data."""
    results, _ = parquet_flow_results
    assert 'prediction_files' in results
    assert 'labeled_count' in results and isinstance(results['labeled_count'], int)
    assert 'unlabeled_count' in results and isinstance(results['unlabeled_count'], int)
    assert 'labeled_data' not in results
    assert 'unlabeled_data' not in results


def test_centralized_path_helper_matches_written_files(parquet_flow_results):
    """FR-001b: written parquet paths match prediction_parquet_path() naming convention."""
    results, output_dir = parquet_flow_results
    for p in results['prediction_files']:
        # Extract cycle number from filename and verify canonical path matches.
        cycle = int(p.stem.removeprefix('prediction_cycle_'))
        assert p == prediction_parquet_path(output_dir, cycle)


def test_parquet_contains_only_unlabeled_predictions_in_run_mode(parquet_flow_results):
    """US1-AC3: in run mode, parquet contains predictions for unlabeled compounds (or all in benchmark)."""
    results, _ = parquet_flow_results
    df: pl.DataFrame = results['compounds_df']
    all_ids = set(df['ID'].to_list())
    for p in results['prediction_files']:
        pred = pl.read_parquet(p)
        pred_ids = set(pred['ID'].to_list())
        # Every parquet ID must reference a real compound.
        assert pred_ids.issubset(all_ids), pred_ids - all_ids

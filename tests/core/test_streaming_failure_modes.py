"""Failure-mode and edge-case tests for the streaming cycle path."""

import numpy as np
import polars as pl
import pytest

from learnm8.core.persistence import (
    CyclePredictionsWriter,
    prediction_parquet_path,
    read_cycle_predictions,
)

pytestmark = pytest.mark.unit


class TestWriterOOMSubchunkOrdering:
    """A logical chunk split by OOM retry into sub-chunks (multiple write_chunk
    calls) must preserve global row order and exact count in the parquet."""

    def test_multiple_write_chunks_preserve_order(self, tmp_path):
        n = 250
        ids_all = [f'c{i}' for i in range(n)]
        preds_all = np.arange(n, dtype=np.float32)
        # Simulate OOM sub-chunking: write in irregular pieces.
        pieces = [(0, 40), (40, 1), (41, 9), (50, 200)]
        with CyclePredictionsWriter(tmp_path, 1, with_uncertainty=False,
                                    row_group_size=64) as w:
            for start, ln in pieces:
                w.write_chunk(
                    pl.Series('ID', ids_all[start:start + ln]),
                    preds_all[start:start + ln],
                    None,
                )
        df = read_cycle_predictions(w.path)
        assert df['ID'].to_list() == ids_all
        np.testing.assert_array_equal(df['prediction'].to_numpy(), preds_all)


class TestCrashAfterPass1:
    """A finalized parquet from a crashed cycle (pass 2 never ran) must be
    overwritten atomically on the next attempt, not silently reused."""

    def test_rerun_overwrites_atomically(self, tmp_path):
        path = prediction_parquet_path(tmp_path, 1)
        # First "crashed" run leaves a finalized parquet with stale content.
        with CyclePredictionsWriter(tmp_path, 1, with_uncertainty=False) as w:
            w.write_chunk(pl.Series('ID', ['stale']), np.array([9.0], np.float32), None)
        assert path.exists()
        stale = read_cycle_predictions(path)
        assert stale['ID'].to_list() == ['stale']

        # Re-run for the same cycle overwrites cleanly (os.replace), no .tmp left.
        with CyclePredictionsWriter(tmp_path, 1, with_uncertainty=False) as w:
            w.write_chunk(pl.Series('ID', ['fresh']), np.array([1.0], np.float32), None)
        fresh = read_cycle_predictions(path)
        assert fresh['ID'].to_list() == ['fresh']
        assert not path.with_name(path.name + '.tmp').exists()


class TestStreamingTopKMaskInteraction:
    """The -inf masking used to exclude pruned rows must never let a masked row
    win, and must keep correct global indices for survivors."""

    def test_neg_inf_rows_never_selected(self):
        from learnm8.acquisition.streaming import StreamingTopK

        scores = np.array([5.0, -np.inf, 4.0, -np.inf, 3.0, 6.0])
        topk = StreamingTopK(3, descending=True)
        topk.update(scores, 0)
        idx, sc = topk.result()
        assert set(idx.tolist()) == {5, 0, 2}  # 6.0, 5.0, 4.0
        assert -np.inf not in sc.tolist()

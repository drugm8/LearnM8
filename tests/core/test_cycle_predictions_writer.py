"""Tests for CyclePredictionsWriter (streaming incremental parquet)."""

import numpy as np
import polars as pl
import pytest

from learnm8.core.persistence import (
    CyclePredictionsWriter,
    prediction_parquet_path,
    read_cycle_predictions,
)
from learnm8.exceptions import LearnerError, PersistenceError

pytestmark = pytest.mark.unit


def _chunks(n, chunk, with_unc, seed=0):
    rng = np.random.default_rng(seed)
    preds_all = rng.normal(size=n).astype(np.float32)
    unc_all = rng.uniform(0, 1, size=n).astype(np.float32) if with_unc else None
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ids = pl.Series('ID', [f'c{i}' for i in range(s, e)])
        unc = unc_all[s:e] if with_unc else None
        yield ids, preds_all[s:e], unc
    yield None, preds_all, unc_all  # sentinel for full-array comparison


def _write(tmp_path, n, chunk, with_unc, rg=1_000_000):
    gen = _chunks(n, chunk, with_unc)
    *parts, full = list(gen)
    _ids_full, preds_full, unc_full = full
    with CyclePredictionsWriter(
        tmp_path, cycle=1, with_uncertainty=with_unc, row_group_size=rg
    ) as w:
        for ids, preds, unc in parts:
            w.write_chunk(ids, preds, unc)
    return w.path, preds_full, unc_full


class TestRoundTrip:
    @pytest.mark.parametrize('chunk', [1, 7, 64, 1000])
    @pytest.mark.parametrize('with_unc', [True, False])
    def test_content_matches_full_arrays_any_chunking(self, tmp_path, chunk, with_unc):
        n = 1000
        path, preds_full, unc_full = _write(tmp_path, n, chunk, with_unc)
        df = read_cycle_predictions(path)
        assert df['ID'].to_list() == [f'c{i}' for i in range(n)]
        np.testing.assert_allclose(df['prediction'].to_numpy(), preds_full, rtol=0, atol=0)
        if with_unc:
            np.testing.assert_allclose(df['uncertainty'].to_numpy(), unc_full)
        else:
            assert 'uncertainty' not in df.columns

    def test_atomic_replace_no_tmp_left(self, tmp_path):
        path, _, _ = _write(tmp_path, 100, 10, True)
        assert path.exists()
        assert not path.with_name(path.name + '.tmp').exists()


class TestRowGroupAlignment:
    def test_row_group_count_equals_ceil_n_over_rg(self, tmp_path):
        import pyarrow.parquet as pq

        n, rg = 25, 10  # small row_group_size to exercise alignment
        path, _, _ = _write(tmp_path, n, 3, True, rg=rg)
        pf = pq.ParquetFile(str(path))
        assert pf.num_row_groups == 3  # ceil(25/10)
        # first two groups exactly rg rows, last is the tail
        sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
        assert sizes == [10, 10, 5]

    def test_small_chunks_still_aligned_groups(self, tmp_path):
        import pyarrow.parquet as pq

        # chunk smaller than rg must NOT create one tiny group per chunk
        n, rg = 100, 50
        path, _, _ = _write(tmp_path, n, 1, False, rg=rg)
        pf = pq.ParquetFile(str(path))
        assert pf.num_row_groups == 2


class TestSchemaInvariant:
    def test_uncertainty_none_with_schema_true_raises(self, tmp_path):
        with pytest.raises(LearnerError):
            with CyclePredictionsWriter(tmp_path, 1, with_uncertainty=True) as w:
                w.write_chunk(pl.Series('ID', ['a']), np.array([1.0], np.float32), None)

    def test_uncertainty_present_with_schema_false_raises(self, tmp_path):
        with pytest.raises(LearnerError):
            with CyclePredictionsWriter(tmp_path, 1, with_uncertainty=False) as w:
                w.write_chunk(
                    pl.Series('ID', ['a']),
                    np.array([1.0], np.float32),
                    np.array([0.1], np.float32),
                )


class TestFailureModes:
    def test_exception_in_block_leaves_no_tmp_or_final(self, tmp_path):
        path = prediction_parquet_path(tmp_path, 1)
        with pytest.raises(RuntimeError):
            with CyclePredictionsWriter(tmp_path, 1, with_uncertainty=False) as w:
                w.write_chunk(pl.Series('ID', ['a']), np.array([1.0], np.float32), None)
                raise RuntimeError('boom')
        assert not path.exists()
        assert not path.with_name(path.name + '.tmp').exists()

    def test_close_with_zero_chunks(self, tmp_path):
        with CyclePredictionsWriter(tmp_path, 1, with_uncertainty=False) as w:
            pass
        df = read_cycle_predictions(w.path)
        assert df.height == 0

    def test_write_after_close_raises(self, tmp_path):
        w = CyclePredictionsWriter(tmp_path, 1, with_uncertainty=False)
        w.__enter__()
        w.close()
        with pytest.raises(PersistenceError):
            w.write_chunk(pl.Series('ID', ['a']), np.array([1.0], np.float32), None)

    def test_output_dir_none_is_noop(self):
        with CyclePredictionsWriter(None, 1, with_uncertainty=True) as w:
            w.write_chunk(pl.Series('ID', ['a']), np.array([1.0], np.float32), None)
        assert w.path is None

    def test_creates_missing_output_dir(self, tmp_path):
        out = tmp_path / 'does' / 'not' / 'exist'
        with CyclePredictionsWriter(out, 1, with_uncertainty=False) as w:
            w.write_chunk(pl.Series('ID', ['a']), np.array([1.0], np.float32), None)
        assert w.path.exists()

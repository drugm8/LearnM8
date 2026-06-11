"""Tests for streaming additions to dataframe_ops: iter_status_chunks, mark_pruned_by_ids."""

import polars as pl
import pytest

from learnm8.core.dataframe_ops import (
    iter_status_chunks,
    mark_pruned_by_ids,
    update_status,
)
from learnm8.core.initialization import initialize_master_dataframe_empty

pytestmark = pytest.mark.unit


def _master(n):
    df = pl.DataFrame({'ID': [f'c{i}' for i in range(n)],
                       'SMILES': [f'C{"C" * (i % 5)}' for i in range(n)]})
    return initialize_master_dataframe_empty(df, 'Activity')


class TestIterStatusChunks:
    def test_covers_all_unlabeled_in_order(self):
        df = _master(100)
        chunks = list(iter_status_chunks(df, 'unlabeled', chunk_size=30))
        offsets = [o for o, _ in chunks]
        assert offsets == [0, 30, 60, 90]
        all_ids = []
        for _, c in chunks:
            all_ids.extend(c['ID'].to_list())
        assert all_ids == [f'c{i}' for i in range(100)]

    def test_global_offset_matches_filtered_position(self):
        df = _master(50)
        # label first 10 → unlabeled sequence starts at c10
        df = update_status(df, [f'c{i}' for i in range(10)], 'labeled', 0, 'Activity',
                           pl.Series([0.0] * 10))
        chunks = list(iter_status_chunks(df, 'unlabeled', chunk_size=20))
        assert chunks[0][0] == 0
        assert chunks[0][1]['ID'].to_list()[0] == 'c10'  # first unlabeled
        # offsets index the filtered (unlabeled) sequence, not master rows
        assert [o for o, _ in chunks] == [0, 20]

    def test_columns_projected(self):
        df = _master(10)
        _, c = next(iter_status_chunks(df, 'unlabeled', 5, columns=('ID',)))
        assert c.columns == ['ID']

    def test_empty_when_no_match(self):
        df = _master(5)
        df = update_status(df, [f'c{i}' for i in range(5)], 'labeled', 0, 'Activity',
                           pl.Series([0.0] * 5))
        assert list(iter_status_chunks(df, 'unlabeled', 10)) == []

    def test_bad_chunk_size(self):
        with pytest.raises(ValueError):
            list(iter_status_chunks(_master(3), 'unlabeled', 0))


class TestMarkPrunedByIds:
    def test_matches_update_status_pruned(self):
        df = _master(20)
        ids = [f'c{i}' for i in range(5, 12)]
        via_list = update_status(df, ids, 'pruned', 3, 'Activity')
        via_join = mark_pruned_by_ids(df, pl.Series('ID', ids), 3)

        for col in ('status', 'pruned_cycle'):
            assert via_join[col].to_list() == via_list[col].to_list()

    def test_only_targeted_rows_change(self):
        df = _master(10)
        out = mark_pruned_by_ids(df, pl.Series('ID', ['c2', 'c7']), 1)
        pruned = out.filter(pl.col('status') == 'pruned')['ID'].to_list()
        assert sorted(pruned) == ['c2', 'c7']
        assert out.filter(pl.col('ID') == 'c2')['pruned_cycle'].item() == 1
        assert out.filter(pl.col('ID') == 'c0')['status'].item() == 'unlabeled'

    def test_empty_ids_noop(self):
        df = _master(5)
        out = mark_pruned_by_ids(df, pl.Series('ID', [], dtype=pl.Utf8), 1)
        assert out['status'].to_list() == df['status'].to_list()

    def test_no_extra_columns(self):
        df = _master(5)
        out = mark_pruned_by_ids(df, pl.Series('ID', ['c1']), 0)
        assert set(out.columns) == set(df.columns)

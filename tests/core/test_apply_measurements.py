import math

import polars as pl
import pytest

from learnm8.core.dataframe_ops import apply_measurements
from learnm8.core.initialization import initialize_master_dataframe_empty

pytestmark = pytest.mark.unit

TARGET = 'Activity'
IDS = [f'c{i}' for i in range(6)]


@pytest.fixture
def master():
    pool = pl.DataFrame({'ID': IDS, 'SMILES': ['C' * (i + 1) for i in range(6)]})
    return initialize_master_dataframe_empty(pool, TARGET)


def _row(df, cid):
    return df.filter(pl.col('ID') == cid).row(0, named=True)


def test_finite_values_are_labeled(master):
    meas = pl.DataFrame({'ID': ['c0', 'c1'], TARGET: [1.5, -2.0]})
    out = apply_measurements(master, ['c0', 'c1'], meas, 3, TARGET)
    for cid, val in (('c0', 1.5), ('c1', -2.0)):
        row = _row(out, cid)
        assert row['status'] == 'labeled'
        assert row[TARGET] == val
        assert row['labeled_cycle'] == 3
        assert row['selected_cycle'] == 3


@pytest.mark.parametrize('bad', [None, math.nan, math.inf, -math.inf, 'abc'], ids=repr)
def test_unmeasurable_value_is_failed(master, bad):
    meas = pl.DataFrame(
        {'ID': ['c0', 'c1'], TARGET: [1.0, bad]},
        schema={'ID': pl.Utf8, TARGET: pl.Utf8 if bad == 'abc' else pl.Float64},
        strict=False,
    )
    if bad == 'abc':
        meas = pl.DataFrame({'ID': ['c0', 'c1'], TARGET: ['1.0', 'abc']})
    out = apply_measurements(master, ['c0', 'c1'], meas, 2, TARGET)
    ok, failed = _row(out, 'c0'), _row(out, 'c1')
    assert ok['status'] == 'labeled'
    assert ok[TARGET] == 1.0
    assert failed['status'] == 'failed'
    assert failed['labeled_cycle'] == 2
    assert failed['selected_cycle'] == 2
    assert failed[TARGET] is None


def test_missing_id_is_failed(master):
    meas = pl.DataFrame({'ID': ['c0'], TARGET: [1.0]})
    out = apply_measurements(master, ['c0', 'c1'], meas, 1, TARGET)
    row = _row(out, 'c1')
    assert row['status'] == 'failed'
    assert row['labeled_cycle'] == 1
    assert row['selected_cycle'] == 1
    assert row[TARGET] is None


def test_empty_measurements_all_failed(master):
    meas = pl.DataFrame(
        {'ID': [], TARGET: []}, schema={'ID': pl.Utf8, TARGET: pl.Float64}
    )
    out = apply_measurements(master, ['c0', 'c1'], meas, 0, TARGET)
    assert out.filter(pl.col('status') == 'failed').height == 2


def test_value_alignment_when_measurements_reordered(master):
    selected = ['c0', 'c1', 'c2', 'c3']
    meas = pl.DataFrame(
        {'ID': ['c3', 'c1', 'c0'], TARGET: [3.0, math.nan, 0.5]},
    )
    out = apply_measurements(master, selected, meas, 4, TARGET)
    assert _row(out, 'c0')['status'] == 'labeled'
    assert _row(out, 'c0')[TARGET] == 0.5
    assert _row(out, 'c1')['status'] == 'failed'
    assert _row(out, 'c2')['status'] == 'failed'
    assert _row(out, 'c3')['status'] == 'labeled'
    assert _row(out, 'c3')[TARGET] == 3.0
    assert _row(out, 'c4')['status'] == 'unlabeled'


def test_master_columns_unchanged_and_input_not_mutated(master):
    cols = list(master.columns)
    meas = pl.DataFrame(
        {'ID': ['c0'], TARGET: [None]}, schema={'ID': pl.Utf8, TARGET: pl.Float64}
    )
    out = apply_measurements(master, ['c0', 'c1'], meas, 1, TARGET)
    assert out.columns == cols
    assert len(out.columns) == 7
    assert (master['status'] == 'unlabeled').all()
    assert out['status'].dtype == master['status'].dtype

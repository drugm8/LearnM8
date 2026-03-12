"""Tests for Polars utility functions."""

import pytest
import polars as pl
import numpy as np
from learnm8.utils.polars_utils import map_values_via_join

pytestmark = pytest.mark.unit


def test_map_values_via_join_basic():
    """Test basic dictionary mapping via join."""
    df = pl.DataFrame({
        'ID': ['C1', 'C2', 'C3'],
        'prediction': [None, None, None]
    })

    lookup = {'C1': 0.5, 'C2': 0.6, 'C3': 0.7}
    result = map_values_via_join(df, lookup, 'ID', 'prediction')

    assert result.filter(pl.col('ID') == 'C1').get_column('prediction')[0] == 0.5
    assert result.filter(pl.col('ID') == 'C2').get_column('prediction')[0] == 0.6
    assert result.filter(pl.col('ID') == 'C3').get_column('prediction')[0] == 0.7


def test_map_values_via_join_partial():
    """Test mapping when some keys are missing."""
    df = pl.DataFrame({
        'ID': ['C1', 'C2', 'C3'],
        'prediction': [0.1, 0.2, 0.3]
    })

    lookup = {'C1': 0.5, 'C2': 0.6}
    result = map_values_via_join(df, lookup, 'ID', 'prediction')

    assert result.filter(pl.col('ID') == 'C1').get_column('prediction')[0] == 0.5
    assert result.filter(pl.col('ID') == 'C2').get_column('prediction')[0] == 0.6
    assert result.filter(pl.col('ID') == 'C3').get_column('prediction')[0] == 0.3


def test_map_values_via_join_performance():
    """Test performance with large dictionary."""
    import time

    n = 100_000
    df = pl.DataFrame({
        'ID': [f'C{i}' for i in range(n)],
        'prediction': [None] * n
    })

    mapping = {f'C{i}': float(i) for i in range(n)}

    start = time.time()
    result = map_values_via_join(df, mapping, 'ID', 'prediction')
    duration = time.time() - start

    print(f"Join-based mapping for {n} items: {duration:.4f}s")

    assert result.filter(pl.col('ID') == 'C0').get_column('prediction')[0] == 0.0
    assert result.filter(pl.col('ID') == 'C999').get_column('prediction')[0] == 999.0

    assert duration < 1.0, f"Performance regression: {duration:.4f}s > 1.0s"


def test_map_values_via_join_types():
    """Test mapping with different data types."""
    df = pl.DataFrame({
        'status': ['a', 'b', 'c'],
        'code': [None, None, None]
    })
    mapping = {'a': 1, 'b': 2, 'c': 3}
    result = map_values_via_join(df, mapping, 'status', 'code')
    assert result.get_column('code').to_list() == [1, 2, 3]

    df = pl.DataFrame({
        'id': [1, 2, 3],
        'label': [None, None, None]
    })
    mapping = {1: 'one', 2: 'two', 3: 'three'}
    result = map_values_via_join(df, mapping, 'id', 'label')
    assert result.get_column('label').to_list() == ['one', 'two', 'three']

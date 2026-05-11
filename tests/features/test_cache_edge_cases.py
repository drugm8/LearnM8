"""Edge-case tests for the v2 cache (T033)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learnm8.exceptions import FeatureExtractionError
from learnm8.features import create_featurizer
from learnm8.features.cache import (
    _cache_keys_uint64,
    _merge_sorted_indices,
    from_storage,
    to_storage,
)
from learnm8.features.extraction import extract_features


@pytest.mark.unit
def test_empty_smiles_no_io(tmp_path: Path):
    out = extract_features(
        [], create_featurizer('morgan', n_jobs=1), cache_dir=tmp_path
    )
    assert out.shape == (0, 2048)
    assert out.dtype == np.float32
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_single_smiles_cold_then_warm(tmp_path: Path):
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    a = extract_features(['CCO'], feat, cache_dir=tmp_path)
    b = extract_features(['CCO'], feat, cache_dir=tmp_path)
    assert np.array_equal(a, b)


@pytest.mark.unit
def test_to_from_storage_roundtrip_packed():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 2, size=(10, 2048)).astype(np.float32)
    encoded = to_storage(arr, 'packed_uint8')
    assert encoded.shape == (10, 256)
    assert encoded.dtype == np.uint8
    decoded = from_storage(encoded, 'packed_uint8', 2048)
    assert np.array_equal(arr, decoded)


@pytest.mark.unit
def test_to_from_storage_roundtrip_float32():
    arr = np.random.default_rng(0).random((5, 100)).astype(np.float32)
    encoded = to_storage(arr, 'float32')
    decoded = from_storage(encoded, 'float32', 100)
    assert np.array_equal(arr, decoded)


@pytest.mark.unit
def test_from_storage_chunk_streaming_handles_partial_last_chunk():
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 2, size=(13, 167)).astype(np.float32)
    encoded = to_storage(arr, 'packed_uint8')
    # Force tiny chunk to exercise the streaming branch.
    decoded = from_storage(encoded, 'packed_uint8', 167, max_chunk_rows=4)
    assert np.array_equal(arr, decoded)


@pytest.mark.unit
def test_merge_sorted_indices_concat_path_small():
    h_old = np.array([1, 5, 10], dtype=np.uint64)
    r_old = np.array([0, 1, 2], dtype=np.uint64)
    h_new = np.array([7, 3], dtype=np.uint64)
    r_new = np.array([3, 4], dtype=np.uint64)
    h, r = _merge_sorted_indices(h_old, r_old, h_new, r_new)
    assert np.array_equal(h, np.array([1, 3, 5, 7, 10], dtype=np.uint64))
    assert np.array_equal(r, np.array([0, 4, 1, 3, 2], dtype=np.uint64))


@pytest.mark.unit
def test_merge_sorted_indices_rejects_duplicate():
    h_old = np.array([1, 5], dtype=np.uint64)
    r_old = np.array([0, 1], dtype=np.uint64)
    h_new = np.array([5, 7], dtype=np.uint64)  # 5 collides with old
    r_new = np.array([2, 3], dtype=np.uint64)
    with pytest.raises(FeatureExtractionError, match=r'Duplicate'):
        _merge_sorted_indices(h_old, r_old, h_new, r_new)


@pytest.mark.unit
def test_cache_key_truncation_stable_across_calls():
    feat = create_featurizer('morgan', n_jobs=1)
    keys_a = _cache_keys_uint64(['CCO', 'CCC', 'CCN'], feat)
    keys_b = _cache_keys_uint64(['CCO', 'CCC', 'CCN'], feat)
    assert np.array_equal(keys_a, keys_b)
    assert keys_a.dtype == np.uint64


@pytest.mark.unit
def test_cache_directory_auto_created(tmp_path: Path):
    new_dir = tmp_path / 'fresh' / 'cache'
    assert not new_dir.exists()
    extract_features(['CCO'], create_featurizer('morgan', n_jobs=1), cache_dir=new_dir)
    assert new_dir.exists()


@pytest.mark.unit
def test_warm_read_after_partial_write_returns_consistent_data(tmp_path: Path):
    """Two interleaved cold-then-warm calls produce consistent output."""
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    a = extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)
    b = extract_features(['CCN', 'CCO'], feat, cache_dir=tmp_path)
    c = extract_features(['CCO', 'CCC', 'CCN'], feat, cache_dir=tmp_path)
    assert np.array_equal(a[0], c[0])
    assert np.array_equal(a[1], c[1])
    assert np.array_equal(b[0], c[2])

"""Targeted branch-coverage tests pushing cache.py over the 95% NFR-005 line."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.exceptions import FeatureExtractionError
from learnm8.features import create_featurizer
from learnm8.features.cache import (
    _hash_index_cache_get,
    _hash_index_cache_put,
    _load_hash_index,
    _validate_cache_integrity,
    from_storage,
)
from learnm8.features.extraction import extract_features


@pytest.mark.unit
def test_from_storage_empty_packed_returns_empty():
    empty = np.empty((0, 256), dtype=np.uint8)
    out = from_storage(empty, 'packed_uint8', 2048)
    assert out.shape == (0, 2048)
    assert out.dtype == np.float32


@pytest.mark.unit
def test_hash_index_cache_invalidates_on_write_epoch_bump(tmp_path: Path):
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)
    cache_path = tmp_path / 'features_morgan_packed_uint8.h5'

    with h5py.File(cache_path, 'r') as f:
        idx_a = _load_hash_index(f, cache_path)
    extract_features(['CCNCC'], feat, cache_dir=tmp_path)
    with h5py.File(cache_path, 'r') as f:
        idx_b = _load_hash_index(f, cache_path)

    assert idx_a.size == 2
    assert idx_b.size == 3


@pytest.mark.unit
def test_hash_index_cache_get_returns_none_when_missing():
    fake_path = Path('/tmp/nonexistent_cache_v2.h5')
    assert _hash_index_cache_get(fake_path, 12345, 0) is None


@pytest.mark.unit
def test_hash_index_cache_put_typeerror_silenced():
    """Non-arr input cannot be weakref'd; the put helper must swallow that."""
    fake_path = Path('/tmp/x.h5')
    _hash_index_cache_put(fake_path, 0, 0, 'not-an-array')  # type: ignore[arg-type]


@pytest.mark.unit
def test_validate_cache_integrity_featurizer_name_mismatch(tmp_path: Path):
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    extract_features(['CCO'], feat, cache_dir=tmp_path)
    cache_path = tmp_path / 'features_morgan_packed_uint8.h5'
    with h5py.File(cache_path, 'r+') as f:
        f.attrs['featurizer_name'] = 'wrongname'

    with (
        pytest.raises(FeatureExtractionError, match=r'featurizer_name mismatch'),
        h5py.File(cache_path, 'r') as f,
    ):
        _validate_cache_integrity(f, feat, cache_path)


@pytest.mark.unit
def test_validate_cache_integrity_schema_version_mismatch(tmp_path: Path):
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    extract_features(['CCO'], feat, cache_dir=tmp_path)
    cache_path = tmp_path / 'features_morgan_packed_uint8.h5'
    with h5py.File(cache_path, 'r+') as f:
        f.attrs['schema_version'] = np.uint8(99)

    with (
        pytest.raises(FeatureExtractionError, match=r'schema_version mismatch'),
        h5py.File(cache_path, 'r') as f,
    ):
        _validate_cache_integrity(f, feat, cache_path)

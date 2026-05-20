"""Phase 1 — cache validation split: CacheMetadata + cheap/full validators."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.core.interfaces import Featurizer
from learnm8.exceptions import FeatureExtractionError
from learnm8.features import create_featurizer
from learnm8.features.cache import (
    CacheMetadata,
    _hash_index_cache_clear,
    _validate_cache_cheap,
    _validate_cache_full,
)
from learnm8.features.extraction import extract_features


@pytest.mark.unit
def test_cache_metadata_dense_construction():
    meta = CacheMetadata(
        schema_version=3,
        bit_count=2048,
        storage_dtype='packed_uint8',
        storage_layout='dense',
        featurizer_name='morgan',
        write_epoch=0,
        n_rows=10,
        feature_width=256,
    )
    assert meta.storage_layout == 'dense'
    assert meta.feature_width == 256


@pytest.mark.unit
def test_cache_metadata_csr_construction():
    meta = CacheMetadata(
        schema_version=3,
        bit_count=2048,
        storage_dtype='csr_uint16',
        storage_layout='csr',
        featurizer_name='erg',
        write_epoch=2,
        n_rows=5,
        feature_width=None,
    )
    assert meta.storage_layout == 'csr'
    assert meta.feature_width is None


@pytest.mark.unit
def test_cache_metadata_is_frozen():
    meta = CacheMetadata(
        schema_version=3,
        bit_count=2048,
        storage_dtype='packed_uint8',
        storage_layout='dense',
        featurizer_name='morgan',
        write_epoch=0,
        n_rows=10,
        feature_width=256,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.n_rows = 11  # type: ignore[misc]


@pytest.mark.unit
def test_cache_metadata_dense_requires_feature_width():
    """Dense layout with feature_width=None violates the invariant."""
    with pytest.raises(ValueError, match='feature_width'):
        CacheMetadata(
            schema_version=3,
            bit_count=2048,
            storage_dtype='packed_uint8',
            storage_layout='dense',
            featurizer_name='morgan',
            write_epoch=0,
            n_rows=10,
            feature_width=None,
        )


@pytest.mark.unit
def test_cache_metadata_csr_rejects_feature_width():
    """CSR layout carries no dense feature_width; a non-None value is invalid."""
    with pytest.raises(ValueError, match='feature_width'):
        CacheMetadata(
            schema_version=3,
            bit_count=2048,
            storage_dtype='csr_uint16',
            storage_layout='csr',
            featurizer_name='erg',
            write_epoch=0,
            n_rows=5,
            feature_width=256,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeCsrFeaturizer(Featurizer):
    """Synthetic featurizer declaring ``csr_uint16`` storage (sparse path)."""

    def __init__(self, values: np.ndarray, name: str = 'fake_csr') -> None:
        self._values = np.asarray(values, dtype=np.float32)
        self._name = name
        self._dim = int(self._values.shape[1])

    def transform(self, smiles_list: list[str]) -> np.ndarray:
        out = np.zeros((len(smiles_list), self._dim), dtype=np.float32)
        for i, _ in enumerate(smiles_list):
            out[i] = self._values[i % self._values.shape[0]]
        return out

    def get_dimension(self) -> int:
        return self._dim

    def get_name(self) -> str:
        return self._name

    def get_storage_dtype(self) -> str:
        return 'csr_uint16'


@pytest.fixture
def dense_cache(tmp_path: Path) -> tuple[Path, object]:
    """A valid dense (packed_uint8) Morgan cache and its featurizer."""
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    extract_features(['CCO', 'CCC', 'CCN'], feat, cache_dir=tmp_path)
    _hash_index_cache_clear()
    return tmp_path / 'features_morgan.h5', feat


@pytest.fixture
def csr_cache(tmp_path: Path) -> tuple[Path, _FakeCsrFeaturizer]:
    """A valid CSR (csr_uint16) cache and its featurizer."""
    values = np.array(
        [[1, 0, 2, 0, 0, 3, 0, 0], [0, 0, 0, 4, 0, 0, 0, 0]], dtype=np.float32
    )
    feat = _FakeCsrFeaturizer(values, name='fake_csr')
    extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)
    _hash_index_cache_clear()
    return tmp_path / 'features_fake_csr.h5', feat


# ---------------------------------------------------------------------------
# Cheap validator
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cheap_validator_accepts_valid_dense_cache(dense_cache):
    path, feat = dense_cache
    with h5py.File(path, 'r') as f:
        meta = _validate_cache_cheap(f, feat, path)
    assert isinstance(meta, CacheMetadata)
    assert meta.storage_layout == 'dense'
    assert meta.storage_dtype == 'packed_uint8'
    assert meta.n_rows == 3
    assert meta.feature_width == 256
    assert meta.bit_count == 2048


@pytest.mark.unit
def test_cheap_validator_accepts_valid_csr_cache(csr_cache):
    path, feat = csr_cache
    with h5py.File(path, 'r') as f:
        meta = _validate_cache_cheap(f, feat, path)
    assert meta.storage_layout == 'csr'
    assert meta.storage_dtype == 'csr_uint16'
    assert meta.feature_width is None
    assert meta.n_rows == 2


@pytest.mark.unit
def test_cheap_validator_rejects_missing_attr(dense_cache):
    path, feat = dense_cache
    with h5py.File(path, 'r+') as f:
        del f.attrs['featurizer_name']
    with h5py.File(path, 'r') as f, pytest.raises(
        FeatureExtractionError, match='missing required attr'
    ):
        _validate_cache_cheap(f, feat, path)


@pytest.mark.unit
def test_cheap_validator_rejects_dtype_mismatch(dense_cache):
    path, feat = dense_cache
    with h5py.File(path, 'r+') as f:
        f.attrs['storage_dtype'] = 'float32'
    with h5py.File(path, 'r') as f, pytest.raises(
        FeatureExtractionError, match='dtype mismatch'
    ):
        _validate_cache_cheap(f, feat, path)


@pytest.mark.unit
def test_cheap_validator_rejects_shape_mismatch(dense_cache):
    path, feat = dense_cache
    with h5py.File(path, 'r+') as f:
        existing = f['features'][:]
        del f['features']
        f.create_dataset(
            'features',
            data=np.zeros((existing.shape[0], 100), dtype=np.uint8),
            maxshape=(None, 100),
            chunks=(4096, 100),
        )
    with h5py.File(path, 'r') as f, pytest.raises(
        FeatureExtractionError, match='shape mismatch'
    ):
        _validate_cache_cheap(f, feat, path)


@pytest.mark.unit
def test_cheap_validator_rejects_dirty_flag(dense_cache):
    """An interrupted write leaves dirty=1; the warm path must reject it."""
    path, feat = dense_cache
    with h5py.File(path, 'r+') as f:
        f.attrs['dirty'] = np.uint8(1)
    with h5py.File(path, 'r') as f, pytest.raises(
        FeatureExtractionError, match='dirty'
    ):
        _validate_cache_cheap(f, feat, path)


@pytest.mark.unit
def test_cheap_validator_missing_dirty_attr_is_clean(dense_cache):
    """Caches written before the dirty flag existed lack the attr — treat as clean."""
    path, feat = dense_cache
    with h5py.File(path, 'r+') as f:
        if 'dirty' in f.attrs:
            del f.attrs['dirty']
    with h5py.File(path, 'r') as f:
        meta = _validate_cache_cheap(f, feat, path)
    assert meta.n_rows == 3


@pytest.mark.unit
def test_cheap_validator_rejects_row_index_out_of_bounds(dense_cache):
    """A row_index entry past /features must be caught on the warm path."""
    path, feat = dense_cache
    with h5py.File(path, 'r+') as f:
        f['row_index'][0] = 9999
    with h5py.File(path, 'r') as f, pytest.raises(
        FeatureExtractionError, match='past /features'
    ):
        _validate_cache_cheap(f, feat, path)


@pytest.mark.unit
def test_cheap_validator_does_not_scan_hash_sortedness(dense_cache):
    """Cheap validation deliberately skips the hash-sortedness scan.

    A duplicate hash is structurally valid; only the full validator (or the
    writer's merge) catches it. This pins the speed/safety tradeoff.
    """
    path, feat = dense_cache
    with h5py.File(path, 'r+') as f:
        h = f['hash_index'][:]
        h[1] = h[0]
        f['hash_index'][:] = h
    with h5py.File(path, 'r') as f:
        meta = _validate_cache_cheap(f, feat, path)
    assert meta.n_rows == 3


# ---------------------------------------------------------------------------
# Full validator
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_full_validator_accepts_valid_dense_cache(dense_cache):
    path, feat = dense_cache
    with h5py.File(path, 'r') as f:
        meta = _validate_cache_full(f, feat, path)
    assert isinstance(meta, CacheMetadata)
    assert meta.n_rows == 3


@pytest.mark.unit
def test_full_validator_accepts_valid_csr_cache(csr_cache):
    path, feat = csr_cache
    with h5py.File(path, 'r') as f:
        meta = _validate_cache_full(f, feat, path)
    assert meta.storage_layout == 'csr'


@pytest.mark.unit
def test_full_validator_rejects_duplicate_hashes(dense_cache):
    path, feat = dense_cache
    with h5py.File(path, 'r+') as f:
        h = f['hash_index'][:]
        h[1] = h[0]
        f['hash_index'][:] = h
    with h5py.File(path, 'r') as f, pytest.raises(
        FeatureExtractionError, match='not strictly increasing'
    ):
        _validate_cache_full(f, feat, path)


@pytest.mark.unit
def test_full_validator_rejects_unsorted_hashes(dense_cache):
    path, feat = dense_cache
    with h5py.File(path, 'r+') as f:
        h = f['hash_index'][:]
        h[0], h[1] = h[1].copy(), h[0].copy()
        f['hash_index'][:] = h
    with h5py.File(path, 'r') as f, pytest.raises(
        FeatureExtractionError, match='not strictly increasing'
    ):
        _validate_cache_full(f, feat, path)


@pytest.mark.unit
def test_full_validator_rejects_nonmonotone_csr_indptr(csr_cache):
    path, feat = csr_cache
    with h5py.File(path, 'r+') as f:
        ip = f['csr_indptr'][:]
        # make indptr[1] exceed indptr[2] -> a decreasing step
        ip[1] = ip[-1] + 5
        f['csr_indptr'][:] = ip
    with h5py.File(path, 'r') as f, pytest.raises(
        FeatureExtractionError, match='monotone'
    ):
        _validate_cache_full(f, feat, path)


# ---------------------------------------------------------------------------
# Safety-gap: warm-read behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_warm_read_on_dirty_cache_raises(dense_cache):
    """End-to-end: a dirty cache fails a warm extract_features read."""
    path, feat = dense_cache
    with h5py.File(path, 'r+') as f:
        f.attrs['dirty'] = np.uint8(1)
    _hash_index_cache_clear()
    with pytest.raises(FeatureExtractionError, match='dirty'):
        extract_features(['CCO'], feat, cache_dir=path.parent)


@pytest.mark.unit
def test_warm_read_on_out_of_bounds_row_index_raises(dense_cache):
    """End-to-end: an out-of-bounds row_index fails fast, never reads garbage."""
    path, feat = dense_cache
    with h5py.File(path, 'r+') as f:
        f['row_index'][0] = 9999
    _hash_index_cache_clear()
    with pytest.raises(FeatureExtractionError, match='past /features'):
        extract_features(['CCO'], feat, cache_dir=path.parent)

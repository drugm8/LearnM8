"""Phase 1 — cache validation split: CacheMetadata + cheap/full validators."""

from __future__ import annotations

import dataclasses

import pytest

from learnm8.features.cache import CacheMetadata


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

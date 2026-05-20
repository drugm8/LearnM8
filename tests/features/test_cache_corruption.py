"""Corruption detection: each invariant violation raises FeatureExtractionError (T024)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.exceptions import FeatureExtractionError
from learnm8.features import create_featurizer
from learnm8.features.cache import _HASH_INDEX_CACHE
from learnm8.features.extraction import extract_features


@pytest.fixture
def populated_cache(tmp_path: Path) -> Path:
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    extract_features(['CCO', 'CCC', 'CCN'], feat, cache_dir=tmp_path)
    _HASH_INDEX_CACHE.clear()
    return tmp_path / 'features_morgan.h5'


def _break_invariant_1_length_mismatch(cache_file: Path) -> None:
    with h5py.File(cache_file, 'r+') as f:
        new_len = len(f['row_index']) + 1
        f['row_index'].resize((new_len,))


def _break_invariant_2_duplicate_hashes(cache_file: Path) -> None:
    with h5py.File(cache_file, 'r+') as f:
        h = f['hash_index'][:]
        if len(h) >= 2:
            h[1] = h[0]
            f['hash_index'][:] = h


def _break_invariant_3_row_past_features(cache_file: Path) -> None:
    with h5py.File(cache_file, 'r+') as f:
        feat = f['features']
        if feat.shape[0] > 0:
            feat.resize((feat.shape[0] - 1, feat.shape[1]))


def _break_invariant_4_dtype_mismatch(cache_file: Path) -> None:
    with h5py.File(cache_file, 'r+') as f:
        f.attrs['storage_dtype'] = 'float32'


def _break_invariant_5_bit_count_inconsistent(cache_file: Path) -> None:
    """Bit_count mismatch with featurizer.get_dimension() triggers rename, not corruption.

    To break the data-model invariant 5 (features.shape[1] inconsistent with bit_count
    given storage_dtype) without triggering the rename, we corrupt features.shape directly:
    leave bit_count=2048 and storage_dtype='packed_uint8' (so expected width=256), but
    sneak features into shape (N, 100) by recreating the dataset.
    """
    with h5py.File(cache_file, 'r+') as f:
        existing = f['features'][:]
        del f['features']
        f.create_dataset(
            'features',
            data=np.zeros((existing.shape[0], 100), dtype=np.uint8),
            maxshape=(None, 100),
            chunks=(4096, 100),
        )


def _break_invariant_6_missing_attr(cache_file: Path) -> None:
    with h5py.File(cache_file, 'r+') as f:
        del f.attrs['featurizer_name']


def _break_invariant_7_missing_dataset(cache_file: Path) -> None:
    with h5py.File(cache_file, 'r+') as f:
        del f['hash_index']


CORRUPTORS = {
    1: _break_invariant_1_length_mismatch,
    2: _break_invariant_2_duplicate_hashes,
    3: _break_invariant_3_row_past_features,
    4: _break_invariant_4_dtype_mismatch,
    5: _break_invariant_5_bit_count_inconsistent,
    6: _break_invariant_6_missing_attr,
    7: _break_invariant_7_missing_dataset,
}


@pytest.mark.unit
@pytest.mark.parametrize('invariant_id', sorted(CORRUPTORS))
def test_each_invariant_violation_raises(populated_cache: Path, invariant_id: int):
    """Every invariant violation surfaces as FeatureExtractionError end-to-end.

    Invariants 1,3-7 are caught by cheap validation on the warm read path.
    Invariant 2 (duplicate hashes) is deliberately NOT scanned by cheap
    validation — it is caught by ``_merge_sorted_indices`` on the write path,
    which is why the probe SMILES must be valid (a miss that reaches the
    writer). Full-validator coverage of invariant 2 lives in
    ``test_cache_validation.py``.
    """
    CORRUPTORS[invariant_id](populated_cache)
    cache_dir = populated_cache.parent

    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    with pytest.raises(FeatureExtractionError, match=r'[Dd]elete the cache file'):
        extract_features(['c1ccccc1'], feat, cache_dir=cache_dir)


@pytest.mark.unit
def test_truncated_file_fails_fast(populated_cache: Path):
    file_size = populated_cache.stat().st_size
    with open(populated_cache, 'r+b') as f:
        f.truncate(file_size // 2)

    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    with pytest.raises(FeatureExtractionError):
        extract_features(['CCO', 'XXXNEW'], feat, cache_dir=populated_cache.parent)


@pytest.mark.unit
def test_no_silent_fallback_on_corruption(populated_cache: Path):
    """FR-011: corrupt cache must NEVER silently re-fingerprint."""
    _break_invariant_4_dtype_mismatch(populated_cache)

    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    with pytest.raises(FeatureExtractionError):
        extract_features(['CCO'], feat, cache_dir=populated_cache.parent)

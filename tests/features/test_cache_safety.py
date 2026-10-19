"""Cache safety: the cache never deletes, renames or overwrites an existing file."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.exceptions import PersistenceError
from learnm8.features import cache as cache_mod
from learnm8.features import create_featurizer
from learnm8.features.cache import _HASH_INDEX_CACHE
from learnm8.features.extraction import extract_features

SMILES = ['CCO', 'CCC', 'CCN']


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _morgan(fp_size: int):
    return create_featurizer('morgan', radius=2, fp_size=fp_size, n_jobs=1)


def _path(tmp_path: Path, dim: int) -> Path:
    return tmp_path / f'features_morgan_{dim}_packed_uint8.h5'


@pytest.mark.unit
def test_different_dimensions_use_distinct_files(tmp_path: Path):
    f2048, f1024 = _morgan(2048), _morgan(1024)
    cold_2048 = extract_features(SMILES, f2048, cache_dir=tmp_path)
    cold_1024 = extract_features(SMILES, f1024, cache_dir=tmp_path)

    assert _path(tmp_path, 2048).exists()
    assert _path(tmp_path, 1024).exists()
    assert len(list(tmp_path.glob('*.h5'))) == 2

    warm_2048 = extract_features(SMILES, f2048, cache_dir=tmp_path)
    warm_1024 = extract_features(SMILES, f1024, cache_dir=tmp_path)
    assert warm_2048.shape == (3, 2048)
    assert warm_1024.shape == (3, 1024)
    np.testing.assert_array_equal(warm_2048, cold_2048)
    np.testing.assert_array_equal(warm_1024, cold_1024)


@pytest.mark.unit
def test_wrong_schema_version_raises_and_leaves_file_untouched(tmp_path: Path):
    feat = _morgan(2048)
    extract_features(SMILES, feat, cache_dir=tmp_path)
    path = _path(tmp_path, 2048)
    with h5py.File(path, 'r+') as f:
        f.attrs['schema_version'] = np.uint8(99)
    _HASH_INDEX_CACHE.clear()
    before = _sha(path)

    with pytest.raises(PersistenceError, match=r'schema_version=99'):
        extract_features(SMILES, feat, cache_dir=tmp_path)
    with pytest.raises(PersistenceError, match=r'schema_version=99'):
        extract_features(['CCCC'], feat, cache_dir=tmp_path)
    assert _sha(path) == before


@pytest.mark.unit
def test_garbage_file_raises_and_leaves_file_untouched(tmp_path: Path):
    path = _path(tmp_path, 2048)
    path.write_bytes(os.urandom(512))
    before = _sha(path)

    with pytest.raises(PersistenceError, match=r'left untouched'):
        extract_features(SMILES, _morgan(2048), cache_dir=tmp_path)
    assert _sha(path) == before


@pytest.mark.unit
def test_truncated_file_raises_and_leaves_file_untouched(tmp_path: Path):
    feat = _morgan(2048)
    extract_features(SMILES, feat, cache_dir=tmp_path)
    path = _path(tmp_path, 2048)
    path.write_bytes(path.read_bytes()[:100])
    _HASH_INDEX_CACHE.clear()
    before = _sha(path)

    with pytest.raises(PersistenceError, match=str(path)):
        extract_features(SMILES, feat, cache_dir=tmp_path)
    assert _sha(path) == before


@pytest.mark.unit
def test_bit_count_mismatch_raises_and_leaves_file_untouched(tmp_path: Path):
    extract_features(SMILES, _morgan(2048), cache_dir=tmp_path)
    wrong = _path(tmp_path, 1024)
    shutil.copy(_path(tmp_path, 2048), wrong)
    before = _sha(wrong)

    with pytest.raises(PersistenceError, match=r'bit_count=2048, expected 1024'):
        extract_features(SMILES, _morgan(1024), cache_dir=tmp_path)
    assert _sha(wrong) == before


@pytest.mark.unit
@pytest.mark.skipif(
    'LEARNM8_LOCK_TIMEOUT_S' in os.environ,
    reason='LEARNM8_LOCK_TIMEOUT_S overrides the default',
)
def test_default_lock_timeout_is_one_hour():
    assert cache_mod.LOCK_TIMEOUT_S == 3600.0

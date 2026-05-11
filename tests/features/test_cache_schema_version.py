"""Schema version & dimension migration tests (T030, FR-008, post-review FR-008 tightening)."""

from __future__ import annotations

import errno
import logging
from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.exceptions import PersistenceError
from learnm8.features import create_featurizer
from learnm8.features.extraction import extract_features


def _make_v1_file(path: Path, dim: int = 2048) -> None:
    with h5py.File(path, 'w') as f:
        grp = f.create_group('features')
        grp.create_dataset('abc123', data=np.zeros(dim, dtype=np.float32))


def _make_v2_file(tmp_path: Path) -> Path:
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)
    return tmp_path / 'features_morgan.h5'


@pytest.mark.unit
def test_fresh_cache_writes_v2(tmp_path: Path):
    feat = create_featurizer('morgan', n_jobs=1)
    extract_features(['CCO'], feat, cache_dir=tmp_path)

    cache_file = tmp_path / 'features_morgan.h5'
    with h5py.File(cache_file, 'r') as f:
        assert int(f.attrs['schema_version']) == 2
        assert int(f.attrs['bit_count']) == 2048
        assert int(f.attrs['write_epoch']) >= 1


@pytest.mark.unit
def test_v1_file_renamed_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    cache_file = tmp_path / 'features_morgan.h5'
    _make_v1_file(cache_file)

    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    with caplog.at_level(logging.WARNING):
        out = extract_features(['CCO'], feat, cache_dir=tmp_path)

    assert out.shape == (1, 2048)
    assert (tmp_path / 'features_morgan.h5.v1.bak').exists()
    assert cache_file.exists()
    with h5py.File(cache_file, 'r') as f:
        assert int(f.attrs['schema_version']) == 2

    rename_warnings = [r for r in caplog.records if 'Renamed non-v2 cache' in r.message]
    assert len(rename_warnings) == 1


@pytest.mark.unit
def test_v3_from_future_renamed_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    cache_file = tmp_path / 'features_morgan.h5'
    with h5py.File(cache_file, 'w') as f:
        f.attrs['schema_version'] = np.uint8(3)
        f.attrs['bit_count'] = np.uint32(2048)
        f.attrs['storage_dtype'] = 'packed_uint8'
        f.attrs['featurizer_name'] = 'morgan'
        f.attrs['write_epoch'] = np.uint64(0)

    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    with caplog.at_level(logging.WARNING):
        extract_features(['CCO'], feat, cache_dir=tmp_path)

    assert (tmp_path / 'features_morgan.h5.v3.bak').exists()
    with h5py.File(cache_file, 'r') as f:
        assert int(f.attrs['schema_version']) == 2


@pytest.mark.unit
def test_bit_count_mismatch_renames_to_dim_bak(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    cache_file = _make_v2_file(tmp_path)
    with h5py.File(cache_file, 'r+') as f:
        f.attrs['bit_count'] = np.uint32(1024)

    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    with caplog.at_level(logging.WARNING):
        extract_features(['CCO'], feat, cache_dir=tmp_path)

    assert (tmp_path / 'features_morgan.h5.dim1024.bak').exists()
    with h5py.File(cache_file, 'r') as f:
        assert int(f.attrs['bit_count']) == 2048

    rename_warnings = [r for r in caplog.records if 'mismatched bit_count' in r.message]
    assert len(rename_warnings) == 1


@pytest.mark.unit
def test_rename_failure_raises_persistence_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_file = tmp_path / 'features_morgan.h5'
    _make_v1_file(cache_file)

    def fake_replace(src, dst):
        raise OSError(errno.EXDEV, 'Invalid cross-device link')

    monkeypatch.setattr('learnm8.features.cache.os.replace', fake_replace)

    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    with pytest.raises(PersistenceError, match=r'Failed to rename'):
        extract_features(['CCO'], feat, cache_dir=tmp_path)

    assert cache_file.exists()
    assert not (tmp_path / 'features_morgan.h5.v1.bak').exists()

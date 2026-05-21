"""Dispatch tests: binary featurizers cache packed→float32 transparently (T019)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.features import create_featurizer
from learnm8.features.extraction import extract_features


@pytest.mark.unit
@pytest.mark.parametrize('name', ['morgan', 'maccs', 'avalon', 'pattern'])
def test_binary_featurizer_packed_storage_returns_float32(tmp_path: Path, name: str):
    feat = create_featurizer(name, n_jobs=1)
    out = extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)

    assert out.dtype == np.float32
    unique = set(np.unique(out).tolist())
    assert unique <= {0.0, 1.0}, (
        f'binary featurizer {name} produced non-binary values {unique}'
    )


@pytest.mark.unit
@pytest.mark.parametrize('name', ['mordred', 'rdkit_2d_descriptors', 'estate'])
def test_continuous_featurizer_float32_storage(tmp_path: Path, name: str):
    feat = create_featurizer(name, n_jobs=1)
    out = extract_features(['CCO'], feat, cache_dir=tmp_path)

    assert out.dtype == np.float32

    cache_file = tmp_path / f'features_{feat.get_name()}_{feat.get_storage_dtype()}.h5'
    with h5py.File(cache_file, 'r') as f:
        assert str(f.attrs['storage_dtype']) == 'float32'
        assert f['features'].dtype == np.float32


@pytest.mark.unit
def test_mqns_uses_uint8_storage(tmp_path: Path):
    """016: mqns opts into raw uint8 storage (small-int counts, max ~26)."""
    feat = create_featurizer('mqns', n_jobs=1)
    out = extract_features(['CCO'], feat, cache_dir=tmp_path)
    assert out.dtype == np.float32

    cache_file = tmp_path / 'features_mqns_uint8.h5'
    with h5py.File(cache_file, 'r') as f:
        assert str(f.attrs['storage_dtype']) == 'uint8'
        assert f['features'].dtype == np.uint8


@pytest.mark.unit
def test_empty_smiles_no_file_io(tmp_path: Path):
    feat = create_featurizer('morgan', n_jobs=1)
    out = extract_features([], feat, cache_dir=tmp_path)

    assert out.shape == (0, 2048)
    assert out.dtype == np.float32
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_warm_read_returns_same_dtype_as_cold(tmp_path: Path):
    for name in ('morgan', 'maccs', 'mordred'):
        feat = create_featurizer(name, n_jobs=1)
        cold = extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)
        warm = extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)
        assert cold.dtype == warm.dtype == np.float32


@pytest.mark.unit
def test_packed_storage_uses_correct_packed_width(tmp_path: Path):
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    extract_features(['CCO'], feat, cache_dir=tmp_path)

    cache_file = tmp_path / 'features_morgan_packed_uint8.h5'
    with h5py.File(cache_file, 'r') as f:
        assert f['features'].shape[1] == 256

    feat_maccs = create_featurizer('maccs', n_jobs=1)
    extract_features(['CCO'], feat_maccs, cache_dir=tmp_path)
    cache_file_maccs = tmp_path / 'features_maccs_packed_uint8.h5'
    bit_count = feat_maccs.get_dimension()
    expected_width = (bit_count + 7) // 8
    with h5py.File(cache_file_maccs, 'r') as f:
        assert f['features'].shape[1] == expected_width


@pytest.mark.unit
def test_uint8_csr_fallback_warns_once(tmp_path: Path, caplog):
    import logging
    from learnm8.features import cache as cache_module
    cache_module._uint8_fallback_warned.clear()

    feat = create_featurizer('morgan', count=True, n_jobs=1)
    caplog.set_level(logging.WARNING, logger='learnm8.features.cache')

    extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path, preferred_dtype='uint8')
    assert feat._storage_dtype == 'csr_uint16'

    warn_records = [r for r in caplog.records if '4x working-set' in r.message]
    assert len(warn_records) == 1, f'Expected exactly 1 warning, got {len(warn_records)}'

    caplog.clear()
    extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path, preferred_dtype='uint8')
    warn_records_2 = [r for r in caplog.records if '4x working-set' in r.message]
    assert len(warn_records_2) == 0, 'Warning should not fire twice'

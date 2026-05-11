"""Round-trip tests for v2 cache: cold→write→read→unpack matches cold output (T017, SC-006)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.features import create_featurizer
from learnm8.features.extraction import extract_features


@pytest.mark.unit
def test_morgan_packed_uint8_roundtrip(tmp_path: Path):
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    cold = extract_features(['CCO', 'CCC', 'CCN'], feat, cache_dir=tmp_path)

    assert cold.shape == (3, 2048)
    assert cold.dtype == np.float32
    assert set(np.unique(cold).tolist()) <= {0.0, 1.0}

    cache_file = tmp_path / 'features_morgan.h5'
    with h5py.File(cache_file, 'r') as f:
        assert int(f.attrs['schema_version']) == 2
        assert str(f.attrs['storage_dtype']) == 'packed_uint8'
        assert int(f.attrs['bit_count']) == 2048
        assert f['features'].shape == (3, 256)
        assert f['features'].dtype == np.uint8
        assert f['hash_index'].shape == (3,)
        assert f['hash_index'].dtype == np.uint64
        assert f['row_index'].shape == (3,)
        assert f['row_index'].dtype == np.uint64
        assert np.all(np.diff(f['hash_index'][:]) > 0)

    warm = extract_features(['CCO', 'CCC', 'CCN'], feat, cache_dir=tmp_path)
    assert np.array_equal(cold, warm)
    assert warm.dtype == np.float32


@pytest.mark.unit
def test_maccs_166_non_multiple_of_8_roundtrip(tmp_path: Path):
    feat = create_featurizer('maccs', n_jobs=1)
    cold = extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)

    bit_count = feat.get_dimension()
    assert cold.shape == (2, bit_count)
    assert cold.dtype == np.float32

    cache_file = tmp_path / 'features_maccs.h5'
    expected_packed_width = (bit_count + 7) // 8
    with h5py.File(cache_file, 'r') as f:
        assert int(f.attrs['bit_count']) == bit_count
        assert f['features'].shape == (2, expected_packed_width)

    warm = extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)
    assert np.array_equal(cold, warm)


@pytest.mark.unit
def test_mordred_float32_roundtrip(tmp_path: Path):
    feat = create_featurizer('mordred', n_jobs=1)
    cold = extract_features(['CCO'], feat, cache_dir=tmp_path)

    assert cold.dtype == np.float32

    cache_file = tmp_path / 'features_mordred.h5'
    with h5py.File(cache_file, 'r') as f:
        assert str(f.attrs['storage_dtype']) == 'float32'
        assert f['features'].dtype == np.float32

    warm = extract_features(['CCO'], feat, cache_dir=tmp_path)
    assert np.allclose(cold, warm, equal_nan=True)


@pytest.mark.unit
def test_mixed_hit_miss_preserves_order(tmp_path: Path):
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)
    direct = feat.transform(['CCO', 'CCCC', 'CCC'])

    mixed = extract_features(['CCO', 'CCCC', 'CCC'], feat, cache_dir=tmp_path)
    assert np.array_equal(mixed, direct.astype(np.float32))


@pytest.mark.unit
def test_single_smiles_roundtrip(tmp_path: Path):
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    cold = extract_features(['CCO'], feat, cache_dir=tmp_path)
    warm = extract_features(['CCO'], feat, cache_dir=tmp_path)
    assert np.array_equal(cold, warm)
    assert cold.shape == (1, 2048)

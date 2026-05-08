"""Dispatch tests: binary featurizers cache packed→float32 transparently (T019)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.features import (
    FEATURIZER_REGISTRY,
    MACCSFeaturizer,
    MordredFeaturizer,
    MorganFeaturizer,
)
from learnm8.features.extraction import extract_features


@pytest.mark.unit
@pytest.mark.parametrize('name', ['morgan', 'maccs', 'avalon', 'pattern'])
def test_binary_featurizer_packed_storage_returns_float32(tmp_path: Path, name: str):
    cls = FEATURIZER_REGISTRY[name]
    feat = cls(n_jobs=1)
    out = extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)

    assert out.dtype == np.float32
    unique = set(np.unique(out).tolist())
    assert unique <= {0.0, 1.0}, f"binary featurizer {name} produced non-binary values {unique}"


@pytest.mark.unit
@pytest.mark.parametrize('name', ['mordred', 'rdkit_2d_descriptors', 'estate', 'mqns'])
def test_continuous_featurizer_float32_storage(tmp_path: Path, name: str):
    cls = FEATURIZER_REGISTRY[name]
    feat = cls(n_jobs=1)
    out = extract_features(['CCO'], feat, cache_dir=tmp_path)

    assert out.dtype == np.float32

    cache_file = tmp_path / f'features_{feat.get_name()}.h5'
    with h5py.File(cache_file, 'r') as f:
        assert str(f.attrs['storage_dtype']) == 'float32'
        assert f['features'].dtype == np.float32


@pytest.mark.unit
def test_empty_smiles_no_file_io(tmp_path: Path):
    feat = MorganFeaturizer(n_jobs=1)
    out = extract_features([], feat, cache_dir=tmp_path)

    assert out.shape == (0, 2048)
    assert out.dtype == np.float32
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_warm_read_returns_same_dtype_as_cold(tmp_path: Path):
    for cls in (MorganFeaturizer, MACCSFeaturizer, MordredFeaturizer):
        feat = cls(n_jobs=1)
        cold = extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)
        warm = extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)
        assert cold.dtype == warm.dtype == np.float32


@pytest.mark.unit
def test_packed_storage_uses_correct_packed_width(tmp_path: Path):
    feat = MorganFeaturizer(radius=2, fp_size=2048, n_jobs=1)
    extract_features(['CCO'], feat, cache_dir=tmp_path)

    cache_file = tmp_path / 'features_morgan.h5'
    with h5py.File(cache_file, 'r') as f:
        assert f['features'].shape[1] == 256

    feat_maccs = MACCSFeaturizer(n_jobs=1)
    extract_features(['CCO'], feat_maccs, cache_dir=tmp_path)
    cache_file_maccs = tmp_path / 'features_maccs.h5'
    bit_count = feat_maccs.get_dimension()
    expected_width = (bit_count + 7) // 8
    with h5py.File(cache_file_maccs, 'r') as f:
        assert f['features'].shape[1] == expected_width

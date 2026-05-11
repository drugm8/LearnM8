"""Round-trip + bounds-check tests for the v2 ``uint8`` raw-storage path (016, T008/T020)."""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.core.interfaces import Featurizer
from learnm8.exceptions import FeatureExtractionError
from learnm8.features.extraction import extract_features


class _FakeUint8Featurizer(Featurizer):
    """Synthetic featurizer with declared `'uint8'` storage and tunable values.

    Bypasses rdkit by returning a precomputed `(n, dim)` float matrix.
    """

    def __init__(
        self,
        values: np.ndarray,
        name: str = 'fake_uint8',
        dim: int | None = None,
    ) -> None:
        self._values = np.asarray(values, dtype=np.float32)
        self._name = name
        self._dim = int(self._values.shape[1]) if dim is None else dim

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
        return 'uint8'


@pytest.mark.unit
def test_uint8_cold_then_warm_round_trip_bit_exact(tmp_path: Path):
    rng = np.random.default_rng(0)
    values = rng.integers(0, 27, size=(4, 42)).astype(np.float32)
    feat = _FakeUint8Featurizer(values)
    smiles = ['CCO', 'CCC', 'CCN', 'CCCl']

    cold = extract_features(smiles, feat, cache_dir=tmp_path)
    warm = extract_features(smiles, feat, cache_dir=tmp_path)

    expected = values
    np.testing.assert_array_equal(cold, expected)
    np.testing.assert_array_equal(warm, expected)
    assert cold.dtype == np.float32
    assert warm.dtype == np.float32


@pytest.mark.unit
def test_uint8_warm_read_dtype_is_float32(tmp_path: Path):
    values = np.array([[0, 1, 2, 3, 4]], dtype=np.float32)
    feat = _FakeUint8Featurizer(values, dim=5, name='fake_uint8_dtype')

    extract_features(['CCO'], feat, cache_dir=tmp_path)
    warm = extract_features(['CCO'], feat, cache_dir=tmp_path)
    assert warm.dtype == np.float32
    np.testing.assert_array_equal(warm, values)


@pytest.mark.unit
def test_uint8_storage_dtype_attr_written(tmp_path: Path):
    values = np.array([[10, 20, 30]], dtype=np.float32)
    feat = _FakeUint8Featurizer(values, dim=3, name='fake_uint8_attr')
    extract_features(['CCO'], feat, cache_dir=tmp_path)

    cache_file = tmp_path / 'features_fake_uint8_attr.h5'
    with h5py.File(cache_file, 'r') as f:
        assert str(f.attrs['storage_dtype']) == 'uint8'
        layout = f.attrs.get('storage_layout', 'dense')
        assert str(layout) == 'dense'
        assert f['features'].dtype == np.uint8
        assert f['features'].shape[1] == 3


@pytest.mark.unit
def test_uint8_value_above_255_raises_feature_extraction_error(tmp_path: Path):
    bad = np.array([[260.0, 1.0, 2.0]], dtype=np.float32)
    feat = _FakeUint8Featurizer(bad, dim=3, name='fake_uint8_overflow')

    extract_features(
        ['CCO'],
        _FakeUint8Featurizer(
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            dim=3,
            name='fake_uint8_overflow',
        ),
        cache_dir=tmp_path,
    )
    cache_file = tmp_path / 'features_fake_uint8_overflow.h5'
    with h5py.File(cache_file, 'r') as f:
        pre_features = np.asarray(f['features'][:])
        pre_hash_index = np.asarray(f['hash_index'][:])
        pre_row_index = np.asarray(f['row_index'][:])
        pre_write_epoch = int(f.attrs['write_epoch'])

    with pytest.raises(FeatureExtractionError, match=r'260|out of range|exceeds'):
        extract_features(['CCNew'], feat, cache_dir=tmp_path)

    with h5py.File(cache_file, 'r') as f:
        post_features = np.asarray(f['features'][:])
        post_hash_index = np.asarray(f['hash_index'][:])
        post_row_index = np.asarray(f['row_index'][:])
        post_write_epoch = int(f.attrs['write_epoch'])
    np.testing.assert_array_equal(pre_features, post_features)
    np.testing.assert_array_equal(pre_hash_index, post_hash_index)
    np.testing.assert_array_equal(pre_row_index, post_row_index)
    assert pre_write_epoch == post_write_epoch, 'write_epoch advanced despite raise'


@pytest.mark.unit
def test_uint8_negative_value_raises(tmp_path: Path):
    bad = np.array([[-1.0, 2.0, 3.0]], dtype=np.float32)
    feat = _FakeUint8Featurizer(bad, dim=3, name='fake_uint8_negative')
    with pytest.raises(FeatureExtractionError, match=r'-1|out of range|negative|range'):
        extract_features(['CCO'], feat, cache_dir=tmp_path)


@pytest.mark.unit
def test_uint8_boundary_255_succeeds(tmp_path: Path):
    boundary = np.array([[0.0, 255.0, 128.0]], dtype=np.float32)
    feat = _FakeUint8Featurizer(boundary, dim=3, name='fake_uint8_boundary')
    cold = extract_features(['CCO'], feat, cache_dir=tmp_path)
    warm = extract_features(['CCO'], feat, cache_dir=tmp_path)
    np.testing.assert_array_equal(cold, boundary)
    np.testing.assert_array_equal(warm, boundary)


@pytest.mark.slow
@pytest.mark.molecular
def test_uint8_100k_warm_read_under_half_second(tmp_path: Path):
    """T020: 100k-mqns warm-read of all rows in <0.5 s on one CPU thread."""
    rng = np.random.default_rng(42)
    n_unique = 5000
    dim = 42
    values = rng.integers(0, 27, size=(n_unique, dim)).astype(np.float32)
    feat = _FakeUint8Featurizer(values, dim=dim, name='perf_uint8_100k')

    smiles_unique = [f'C{"C" * i}O' for i in range(n_unique)]
    smiles_all = (smiles_unique * (100_000 // n_unique + 1))[:100_000]

    extract_features(smiles_unique, feat, cache_dir=tmp_path)

    t0 = time.perf_counter()
    warm = extract_features(smiles_all, feat, cache_dir=tmp_path)
    elapsed = time.perf_counter() - t0
    assert warm.shape == (100_000, dim)
    assert elapsed < 0.5, f'100k uint8 warm read took {elapsed:.3f}s (target <0.5s)'

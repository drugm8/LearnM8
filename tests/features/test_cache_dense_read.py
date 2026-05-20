"""Phase 3 — slabbed contiguous-run dense reads.

The runs reader must be bit-identical to the fancy-index reader (kept as the
oracle) across every target-row pattern, and must bound peak memory.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.exceptions import FeatureExtractionError
from learnm8.features.cache import (
    DSET_FEATURES,
    MAX_UNPACK_CHUNK_ROWS,
    _count_dense_runs,
    _read_cache_hits_dense_fancy,
    _read_cache_hits_dense_runs,
    _should_use_dense_runs,
)

BIT_COUNT = 64
N_ROWS = 500


def _make_cache(
    path: Path,
    n_rows: int,
    storage_dtype: str,
    seed: int = 0,
    bit_count: int = BIT_COUNT,
) -> None:
    """Write an HDF5 file with a /features dataset for the given storage dtype."""
    rng = np.random.default_rng(seed)
    if storage_dtype == 'packed_uint8':
        width = (bit_count + 7) // 8
        data = rng.integers(0, 256, size=(n_rows, width), dtype=np.uint8)
    elif storage_dtype == 'uint8':
        data = rng.integers(0, 256, size=(n_rows, bit_count), dtype=np.uint8)
    elif storage_dtype == 'float32':
        data = rng.standard_normal((n_rows, bit_count)).astype(np.float32)
    else:  # pragma: no cover
        raise ValueError(storage_dtype)
    with h5py.File(path, 'w') as f:
        f.create_dataset(DSET_FEATURES, data=data, maxshape=(None, data.shape[1]))


@pytest.fixture
def packed_cache(tmp_path: Path) -> Path:
    path = tmp_path / 'packed.h5'
    _make_cache(path, N_ROWS, 'packed_uint8')
    return path


def _diff(
    path: Path, target_rows: np.ndarray, storage_dtype: str, output_dtype: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run both readers on the same input; return (fancy, runs)."""
    with h5py.File(path, 'r') as f:
        u, inv = np.unique(target_rows, return_inverse=True)
        fancy = _read_cache_hits_dense_fancy(
            f, u, inv, storage_dtype, BIT_COUNT, output_dtype
        )
        runs = _read_cache_hits_dense_runs(
            f, u, inv, storage_dtype, BIT_COUNT, output_dtype, MAX_UNPACK_CHUNK_ROWS
        )
    return fancy, runs


def _assert_identical(fancy: np.ndarray, runs: np.ndarray) -> None:
    assert runs.dtype == fancy.dtype
    assert runs.shape == fancy.shape
    assert np.array_equal(runs, fancy)


# ---------------------------------------------------------------------------
# Run detection + dispatch policy
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ('rows', 'expected'),
    [
        ([], 0),
        ([5], 1),
        ([0, 1, 2, 3], 1),
        ([0, 1, 2, 10, 11, 20], 3),
        ([0, 2, 4, 6], 4),
    ],
)
def test_count_dense_runs(rows: list[int], expected: int) -> None:
    assert _count_dense_runs(np.array(rows, dtype=np.int64)) == expected


@pytest.mark.unit
def test_should_use_dense_runs_contiguous_true() -> None:
    assert _should_use_dense_runs(n_unique=100_000, n_runs=1) is True


@pytest.mark.unit
def test_should_use_dense_runs_sparse_false() -> None:
    # 90k isolated rows -> 90k runs, far above ceil(90k/1000)=90 fancy chunks.
    assert _should_use_dense_runs(n_unique=90_000, n_runs=90_000) is False


@pytest.mark.unit
def test_should_use_dense_runs_empty_false() -> None:
    assert _should_use_dense_runs(n_unique=0, n_runs=0) is False


# ---------------------------------------------------------------------------
# Differential: runs reader == fancy reader, every pattern
# ---------------------------------------------------------------------------


def _patterns() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    k = 120
    return {
        'contiguous_all': np.arange(N_ROWS, dtype=np.int64),
        'contiguous_subset': np.arange(40, 40 + k, dtype=np.int64),
        'feature_order_sparse': np.sort(rng.choice(N_ROWS, k, replace=False)),
        'random_sparse_unsorted': rng.choice(N_ROWS, k, replace=False),
        'duplicate_heavy': rng.integers(0, 12, size=k, dtype=np.int64),
        'mixed_dup_random': np.concatenate(
            [rng.choice(N_ROWS, 60, replace=False), rng.integers(0, 30, 60)]
        ).astype(np.int64),
        'out_of_order': rng.permutation(np.arange(k, dtype=np.int64)),
        'single_row': np.array([7], dtype=np.int64),
        'zero_rows': np.empty(0, dtype=np.int64),
    }


@pytest.mark.unit
@pytest.mark.parametrize('pattern', list(_patterns()))
@pytest.mark.parametrize('output_dtype', ['uint8', 'float32'])
def test_runs_matches_fancy_packed(
    packed_cache: Path, pattern: str, output_dtype: str
) -> None:
    target_rows = _patterns()[pattern]
    fancy, runs = _diff(packed_cache, target_rows, 'packed_uint8', output_dtype)
    _assert_identical(fancy, runs)


@pytest.mark.unit
@pytest.mark.parametrize('storage_dtype', ['uint8', 'float32'])
def test_runs_matches_fancy_other_storage(
    tmp_path: Path, storage_dtype: str
) -> None:
    path = tmp_path / f'{storage_dtype}.h5'
    _make_cache(path, N_ROWS, storage_dtype)
    target_rows = _patterns()['mixed_dup_random']
    fancy, runs = _diff(path, target_rows, storage_dtype, 'float32')
    _assert_identical(fancy, runs)


@pytest.mark.unit
def test_runs_matches_fancy_randomized(packed_cache: Path) -> None:
    """Randomized differential sweep — fancy and runs must always agree."""
    rng = np.random.default_rng(7)
    for _ in range(30):
        size = int(rng.integers(0, 250))
        # with-replacement -> duplicates; unsorted -> reorder.
        target_rows = rng.integers(0, N_ROWS, size=size, dtype=np.int64)
        for output_dtype in ('uint8', 'float32'):
            fancy, runs = _diff(packed_cache, target_rows, 'packed_uint8', output_dtype)
            _assert_identical(fancy, runs)


@pytest.mark.unit
def test_runs_reader_forces_multiple_slabs(packed_cache: Path) -> None:
    """A small slab size must still produce the correct result over many slabs."""
    target_rows = np.arange(N_ROWS, dtype=np.int64)
    with h5py.File(packed_cache, 'r') as f:
        u, inv = np.unique(target_rows, return_inverse=True)
        fancy = _read_cache_hits_dense_fancy(f, u, inv, 'packed_uint8', BIT_COUNT, 'uint8')
        runs = _read_cache_hits_dense_runs(
            f, u, inv, 'packed_uint8', BIT_COUNT, 'uint8', slab_rows=32
        )
    _assert_identical(fancy, runs)


# ---------------------------------------------------------------------------
# Out-of-bounds: raise, never silently clamp
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runs_reader_raises_on_row_past_features(packed_cache: Path) -> None:
    target_rows = np.array([0, 1, N_ROWS + 50], dtype=np.int64)
    with h5py.File(packed_cache, 'r') as f:
        u, inv = np.unique(target_rows, return_inverse=True)
        with pytest.raises(FeatureExtractionError, match='past /features'):
            _read_cache_hits_dense_runs(
                f, u, inv, 'packed_uint8', BIT_COUNT, 'uint8', MAX_UNPACK_CHUNK_ROWS
            )


# ---------------------------------------------------------------------------
# Memory: the slab loop bounds peak RSS to one output array + a bounded slab
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runs_reader_peak_memory_bounded(tmp_path: Path) -> None:
    """A full contiguous read must not allocate a second full-size matrix.

    Uses a realistic 2048-bit width so the output dwarfs the int64 index
    arrays — the regime the slab loop is designed for.
    """
    bit_count = 2048
    n_rows = 40_000
    path = tmp_path / 'big.h5'
    _make_cache(path, n_rows, 'packed_uint8', bit_count=bit_count)
    target_rows = np.arange(n_rows, dtype=np.int64)
    out_bytes = n_rows * bit_count  # uint8 output

    with h5py.File(path, 'r') as f:
        u, inv = np.unique(target_rows, return_inverse=True)
        tracemalloc.start()
        try:
            result = _read_cache_hits_dense_runs(
                f, u, inv, 'packed_uint8', bit_count, 'uint8', MAX_UNPACK_CHUNK_ROWS
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    assert result.shape == (n_rows, bit_count)
    # One full-size output + a bounded decoded slab + index arrays — never a
    # second full-size matrix (the fancy path holds two, ~2.1x output).
    assert peak < out_bytes * 1.6, f'peak {peak} >= 1.6x output {out_bytes}'

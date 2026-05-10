"""Tests for the process-level /hash_index LRU cache (Feature 020 T027/T028/T029).

Covers:
- T027: Core LRU functionality (FR-005, FR-006, FR-007, SC-002).
- T028: write_epoch invalidation (review ShouldFix #6).
- T029: env-var validation for LEARNM8_HASH_INDEX_LRU_MAX (review ShouldFix #8).
"""

from __future__ import annotations

import gc
import threading
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.exceptions import LearnM8Warning
from learnm8.features.cache import (
    _HASH_INDEX_CACHE,
    DEFAULT_HASH_INDEX_LRU_MAX,
    ENV_HASH_INDEX_LRU_MAX,
    HASH_DTYPE,
    _hash_index_cache_clear,
    _hash_index_cache_get,
    _hash_index_cache_put,
    _load_hash_index,
    _max_lru_entries,
)


@pytest.fixture(autouse=True)
def _isolate_lru_cache() -> None:
    """Ensure each test starts with a clean cache and tidy LRU env state."""
    _hash_index_cache_clear()
    yield
    _hash_index_cache_clear()


def _make_hash_array(seed: int, n: int = 4) -> np.ndarray:
    """Build a deterministic S16 ndarray for tests."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    return np.ascontiguousarray(raw).view(HASH_DTYPE).reshape(n)


def _write_minimal_v3_cache(path: Path, n_rows: int = 3) -> None:
    """Write a minimal but valid v3 cache file for the LRU integration test."""
    with h5py.File(path, 'w') as f:
        f.attrs['schema_version'] = np.uint8(3)
        f.attrs['hash_width_bits'] = np.uint8(128)
        f.attrs['bit_count'] = np.uint32(2048)
        f.attrs['storage_dtype'] = 'packed_uint8'
        f.attrs['featurizer_name'] = 'morgan'
        f.attrs['write_epoch'] = np.uint64(0)
        hashes = _make_hash_array(seed=12345, n=n_rows)
        hashes_sorted = np.sort(hashes)
        f.create_dataset(
            'hash_index',
            data=hashes_sorted,
            maxshape=(None,),
            dtype=HASH_DTYPE,
            chunks=(2 ** 16,),
        )
        f.create_dataset(
            'row_index',
            data=np.arange(n_rows, dtype=np.uint64),
            maxshape=(None,),
            dtype=np.uint64,
        )


# ---------------------------------------------------------------------------
# T027: core LRU functionality
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lru_starts_empty_after_clear() -> None:
    _hash_index_cache_put(Path('/tmp/x.h5'), 1, 0, _make_hash_array(0))
    _hash_index_cache_clear()
    assert len(_HASH_INDEX_CACHE) == 0


@pytest.mark.unit
def test_put_get_round_trip(tmp_path: Path) -> None:
    path = tmp_path / 'cache.h5'
    path.touch()
    arr1 = _make_hash_array(1)
    _hash_index_cache_put(path, 100, 1, arr1)
    arr2 = _hash_index_cache_get(path, 100, 1)
    assert arr2 is arr1
    assert arr2.dtype == HASH_DTYPE


@pytest.mark.unit
def test_get_misses_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / 'never_inserted.h5'
    path.touch()
    assert _hash_index_cache_get(path, 1, 0) is None


@pytest.mark.unit
def test_lru_hit_moves_entry_to_mru(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_HASH_INDEX_LRU_MAX, '2')
    path_a = tmp_path / 'a.h5'
    path_b = tmp_path / 'b.h5'
    path_c = tmp_path / 'c.h5'
    for p in (path_a, path_b, path_c):
        p.touch()

    arr_a = _make_hash_array(10)
    arr_b = _make_hash_array(20)
    arr_c = _make_hash_array(30)

    _hash_index_cache_put(path_a, 1, 0, arr_a)
    _hash_index_cache_put(path_b, 1, 0, arr_b)

    hit_a = _hash_index_cache_get(path_a, 1, 0)
    assert hit_a is arr_a

    _hash_index_cache_put(path_c, 1, 0, arr_c)

    assert _hash_index_cache_get(path_a, 1, 0) is arr_a
    assert _hash_index_cache_get(path_c, 1, 0) is arr_c
    assert _hash_index_cache_get(path_b, 1, 0) is None


@pytest.mark.unit
def test_eviction_at_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_HASH_INDEX_LRU_MAX, '2')
    paths = []
    for name in ('a.h5', 'b.h5', 'c.h5'):
        p = tmp_path / name
        p.touch()
        paths.append(p)

    arr_a = _make_hash_array(1)
    arr_b = _make_hash_array(2)
    arr_c = _make_hash_array(3)

    _hash_index_cache_put(paths[0], 1, 0, arr_a)
    _hash_index_cache_put(paths[1], 1, 0, arr_b)
    _hash_index_cache_put(paths[2], 1, 0, arr_c)

    assert len(_HASH_INDEX_CACHE) == 2
    assert _hash_index_cache_get(paths[0], 1, 0) is None
    assert _hash_index_cache_get(paths[1], 1, 0) is arr_b
    assert _hash_index_cache_get(paths[2], 1, 0) is arr_c


@pytest.mark.unit
def test_mtime_invalidation(tmp_path: Path) -> None:
    path = tmp_path / 'cache.h5'
    path.touch()
    arr = _make_hash_array(7)
    _hash_index_cache_put(path, 100, 1, arr)
    assert _hash_index_cache_get(path, 200, 1) is None
    assert _hash_index_cache_get(path, 100, 1) is arr


@pytest.mark.unit
def test_strong_ref_survives_caller_scope(tmp_path: Path) -> None:
    """Regression: previous weakref(np.ndarray) impl let the array be GC'd
    as soon as the caller's local went out of scope. Strong-ref impl must keep it.
    """
    path = tmp_path / 'cache.h5'
    path.touch()

    def _put_and_drop_local() -> None:
        local_arr = _make_hash_array(999)
        _hash_index_cache_put(path, 100, 1, local_arr)

    _put_and_drop_local()
    gc.collect()

    survived = _hash_index_cache_get(path, 100, 1)
    assert survived is not None
    assert survived.dtype == HASH_DTYPE
    assert survived.shape == (4,)


@pytest.mark.unit
def test_load_hash_index_uses_cache(tmp_path: Path) -> None:
    """SC-002 acceptance: second `_load_hash_index` returns the cached strong ref."""
    cache_path = tmp_path / 'features_morgan.h5'
    _write_minimal_v3_cache(cache_path)

    with h5py.File(cache_path, 'r') as f:
        arr1 = _load_hash_index(f, cache_path)
    with h5py.File(cache_path, 'r') as f:
        arr2 = _load_hash_index(f, cache_path)

    assert arr1 is arr2
    assert arr1.dtype == HASH_DTYPE
    assert len(_HASH_INDEX_CACHE) == 1


@pytest.mark.unit
def test_thread_safety(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_HASH_INDEX_LRU_MAX, '8')
    n_threads = 4
    paths: list[Path] = []
    arrays: list[np.ndarray] = []
    for i in range(n_threads):
        p = tmp_path / f'cache_{i}.h5'
        p.touch()
        paths.append(p)
        arrays.append(_make_hash_array(i + 1))

    errors: list[BaseException] = []

    def _worker(idx: int) -> None:
        try:
            _hash_index_cache_put(paths[idx], 100 + idx, 1, arrays[idx])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(_HASH_INDEX_CACHE) == n_threads
    for i, p in enumerate(paths):
        assert _hash_index_cache_get(p, 100 + i, 1) is arrays[i]


# ---------------------------------------------------------------------------
# T028: write_epoch invalidation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_epoch_invalidation(tmp_path: Path) -> None:
    path = tmp_path / 'cache.h5'
    path.touch()
    arr = _make_hash_array(42)
    _hash_index_cache_put(path, 100, 1, arr)

    assert _hash_index_cache_get(path, 100, 2) is None
    assert _hash_index_cache_get(path, 100, 1) is arr


# ---------------------------------------------------------------------------
# T029: env-var validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize('bad_value', ['0', '-1', 'abc', '', '8.5'])
def test_max_lru_entries_invalid_env_warns(
    bad_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_HASH_INDEX_LRU_MAX, bad_value)
    with pytest.warns(LearnM8Warning):
        result = _max_lru_entries()
    assert result == DEFAULT_HASH_INDEX_LRU_MAX


@pytest.mark.unit
def test_max_lru_entries_valid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_HASH_INDEX_LRU_MAX, '7')
    with warnings.catch_warnings():
        warnings.simplefilter('error', LearnM8Warning)
        result = _max_lru_entries()
    assert result == 7


@pytest.mark.unit
def test_max_lru_entries_default_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_HASH_INDEX_LRU_MAX, raising=False)
    assert _max_lru_entries() == DEFAULT_HASH_INDEX_LRU_MAX

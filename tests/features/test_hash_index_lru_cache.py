"""Tests for the consolidated process-level index LRU cache (Phase 2).

One ``_INDEX_CACHE`` holds both ``/hash_index`` and ``/row_index`` per cache
file, keyed on ``(path, st_ino, mtime_ns, write_epoch)``. The cap is the
module constant ``DEFAULT_INDEX_LRU_MAX``; eviction tests monkeypatch it.
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.features import cache as cache_module
from learnm8.features.cache import (
    _INDEX_CACHE,
    HASH_DTYPE,
    _index_cache_clear,
    _index_cache_get,
    _index_cache_put,
    _IndexEntry,
    _load_hash_index,
    _load_row_index,
)


@pytest.fixture(autouse=True)
def _isolate_lru_cache():
    """Each test starts and ends with a clean cache."""
    _index_cache_clear()
    yield
    _index_cache_clear()


def _make_hash_array(seed: int, n: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    return np.ascontiguousarray(raw).view(HASH_DTYPE).reshape(n)


def _key(
    path: Path, st_ino: int = 1, mtime_ns: int = 100, write_epoch: int = 1
) -> tuple[str, int, int, int]:
    return (str(path.resolve()), st_ino, mtime_ns, write_epoch)


def _write_minimal_v3_cache(path: Path, n_rows: int = 3) -> None:
    """Write a minimal v3 cache file sufficient for _load_* (not full validation)."""
    with h5py.File(path, 'w') as f:
        f.attrs['schema_version'] = np.uint8(3)
        f.attrs['hash_width_bits'] = np.uint8(128)
        f.attrs['bit_count'] = np.uint32(2048)
        f.attrs['storage_dtype'] = 'packed_uint8'
        f.attrs['featurizer_name'] = 'morgan'
        f.attrs['write_epoch'] = np.uint64(0)
        hashes_sorted = np.sort(_make_hash_array(seed=12345, n=n_rows))
        f.create_dataset(
            'hash_index', data=hashes_sorted, maxshape=(None,), dtype=HASH_DTYPE
        )
        f.create_dataset(
            'row_index',
            data=np.arange(n_rows, dtype=np.uint64),
            maxshape=(None,),
            dtype=np.uint64,
        )


# ---------------------------------------------------------------------------
# Core LRU mechanics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lru_starts_empty_after_clear(tmp_path: Path) -> None:
    _index_cache_put(_key(tmp_path / 'x.h5'), _IndexEntry(hash_index=_make_hash_array(0)))
    _index_cache_clear()
    assert len(_INDEX_CACHE) == 0


@pytest.mark.unit
def test_put_get_round_trip(tmp_path: Path) -> None:
    path = tmp_path / 'cache.h5'
    entry = _IndexEntry(hash_index=_make_hash_array(1))
    _index_cache_put(_key(path), entry)
    got = _index_cache_get(_key(path))
    assert got is entry


@pytest.mark.unit
def test_get_misses_unknown_key(tmp_path: Path) -> None:
    assert _index_cache_get(_key(tmp_path / 'never.h5')) is None


@pytest.mark.unit
def test_lru_hit_moves_entry_to_mru(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache_module, 'DEFAULT_INDEX_LRU_MAX', 2)
    ka, kb, kc = (_key(tmp_path / f'{n}.h5') for n in 'abc')
    ea, eb, ec = (_IndexEntry(hash_index=_make_hash_array(s)) for s in (10, 20, 30))

    _index_cache_put(ka, ea)
    _index_cache_put(kb, eb)
    assert _index_cache_get(ka) is ea  # touch a -> a is MRU, b is LRU
    _index_cache_put(kc, ec)  # evicts b

    assert _index_cache_get(ka) is ea
    assert _index_cache_get(kc) is ec
    assert _index_cache_get(kb) is None


@pytest.mark.unit
def test_eviction_at_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_module, 'DEFAULT_INDEX_LRU_MAX', 2)
    keys = [_key(tmp_path / f'{n}.h5') for n in 'abc']
    entries = [_IndexEntry(hash_index=_make_hash_array(s)) for s in (1, 2, 3)]
    for k, e in zip(keys, entries, strict=True):
        _index_cache_put(k, e)

    assert len(_INDEX_CACHE) == 2
    assert _index_cache_get(keys[0]) is None
    assert _index_cache_get(keys[1]) is entries[1]
    assert _index_cache_get(keys[2]) is entries[2]


@pytest.mark.unit
def test_mtime_invalidation(tmp_path: Path) -> None:
    path = tmp_path / 'cache.h5'
    entry = _IndexEntry(hash_index=_make_hash_array(7))
    _index_cache_put(_key(path, mtime_ns=100), entry)
    assert _index_cache_get(_key(path, mtime_ns=200)) is None
    assert _index_cache_get(_key(path, mtime_ns=100)) is entry


@pytest.mark.unit
def test_st_ino_invalidation(tmp_path: Path) -> None:
    """A delete-and-recreate at the same path changes st_ino — must miss."""
    path = tmp_path / 'cache.h5'
    entry = _IndexEntry(hash_index=_make_hash_array(7))
    _index_cache_put(_key(path, st_ino=111), entry)
    assert _index_cache_get(_key(path, st_ino=222)) is None
    assert _index_cache_get(_key(path, st_ino=111)) is entry


@pytest.mark.unit
def test_write_epoch_invalidation(tmp_path: Path) -> None:
    path = tmp_path / 'cache.h5'
    entry = _IndexEntry(hash_index=_make_hash_array(42))
    _index_cache_put(_key(path, write_epoch=1), entry)
    assert _index_cache_get(_key(path, write_epoch=2)) is None
    assert _index_cache_get(_key(path, write_epoch=1)) is entry


@pytest.mark.unit
def test_stale_same_path_entries_evicted_on_put(tmp_path: Path) -> None:
    """Putting a new (path, epoch) drops the old entry for the same path."""
    path = tmp_path / 'cache.h5'
    _index_cache_put(_key(path, write_epoch=1), _IndexEntry(hash_index=_make_hash_array(1)))
    _index_cache_put(_key(path, write_epoch=2), _IndexEntry(hash_index=_make_hash_array(2)))
    assert len(_INDEX_CACHE) == 1
    assert _index_cache_get(_key(path, write_epoch=1)) is None


@pytest.mark.unit
def test_strong_ref_survives_caller_scope(tmp_path: Path) -> None:
    path = tmp_path / 'cache.h5'

    def _put_and_drop_local() -> None:
        _index_cache_put(_key(path), _IndexEntry(hash_index=_make_hash_array(999)))

    _put_and_drop_local()
    gc.collect()
    survived = _index_cache_get(_key(path))
    assert survived is not None
    assert survived.hash_index is not None


# ---------------------------------------------------------------------------
# _load_hash_index / _load_row_index integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_hash_index_uses_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / 'features_morgan.h5'
    _write_minimal_v3_cache(cache_path)
    with h5py.File(cache_path, 'r') as f:
        arr1 = _load_hash_index(f, cache_path)
    with h5py.File(cache_path, 'r') as f:
        arr2 = _load_hash_index(f, cache_path)
    assert arr1 is arr2
    assert arr1.dtype == HASH_DTYPE


@pytest.mark.unit
def test_load_row_index_uses_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / 'features_morgan.h5'
    _write_minimal_v3_cache(cache_path)
    with h5py.File(cache_path, 'r') as f:
        arr1 = _load_row_index(f, cache_path)
    with h5py.File(cache_path, 'r') as f:
        arr2 = _load_row_index(f, cache_path)
    assert arr1 is arr2
    assert arr1.dtype == np.dtype(np.uint64)


@pytest.mark.unit
def test_hash_and_row_share_one_entry(tmp_path: Path) -> None:
    """Both loaders populate a single consolidated cache entry."""
    cache_path = tmp_path / 'features_morgan.h5'
    _write_minimal_v3_cache(cache_path)
    with h5py.File(cache_path, 'r') as f:
        _load_hash_index(f, cache_path)
        _load_row_index(f, cache_path)
    assert len(_INDEX_CACHE) == 1
    entry = next(iter(_INDEX_CACHE.values()))
    assert entry.hash_index is not None
    assert entry.row_index is not None


@pytest.mark.unit
def test_cached_index_arrays_are_read_only(tmp_path: Path) -> None:
    """Shared LRU arrays must be immutable so one thread cannot corrupt another."""
    cache_path = tmp_path / 'features_morgan.h5'
    _write_minimal_v3_cache(cache_path)
    with h5py.File(cache_path, 'r') as f:
        h = _load_hash_index(f, cache_path)
        r = _load_row_index(f, cache_path)
    assert not h.flags.writeable
    assert not r.flags.writeable
    with pytest.raises(ValueError):
        r[0] = 0


@pytest.mark.unit
def test_load_helpers_invalidate_on_write_epoch_bump(tmp_path: Path) -> None:
    """A writer's write_epoch bump makes the next _load_* miss the stale array."""
    from learnm8.features import create_featurizer
    from learnm8.features.extraction import extract_features

    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)
    cache_path = tmp_path / 'features_morgan.h5'
    with h5py.File(cache_path, 'r') as f:
        idx_a = _load_hash_index(f, cache_path)
    extract_features(['CCNCC'], feat, cache_dir=tmp_path)
    with h5py.File(cache_path, 'r') as f:
        idx_b = _load_hash_index(f, cache_path)
    assert idx_a.size == 2
    assert idx_b.size == 3


@pytest.mark.unit
def test_thread_safety(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_module, 'DEFAULT_INDEX_LRU_MAX', 8)
    n_threads = 4
    keys = [_key(tmp_path / f'c{i}.h5', st_ino=i + 1) for i in range(n_threads)]
    entries = [_IndexEntry(hash_index=_make_hash_array(i + 1)) for i in range(n_threads)]
    errors: list[BaseException] = []

    def _worker(idx: int) -> None:
        try:
            _index_cache_put(keys[idx], entries[idx])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(_INDEX_CACHE) == n_threads
    for i, k in enumerate(keys):
        assert _index_cache_get(k) is entries[i]

"""Auto-migration tests for v2 (64-bit hash) caches to v3 (128-bit hash).

Covers Feature 020 tasks T030 (FR-003, FR-004, SC-005) and T031 (concurrent
migration race). The implementation lives in
``learnm8.features.cache._open_or_create_h5``: any cache file with
``schema_version < 3`` is renamed to ``<name>.h5.hash64.bak`` (atomic rename
under LOCK_EX) and a fresh v3 file is created in its place. An existing
``.hash64.bak`` is never clobbered — migration is refused with a
``PersistenceError``.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.exceptions import PersistenceError
from learnm8.features.cache import (
    ATTR_BIT_COUNT,
    ATTR_FEATURIZER_NAME,
    ATTR_SCHEMA_VERSION,
    ATTR_STORAGE_DTYPE,
    ATTR_STORAGE_LAYOUT,
    ATTR_WRITE_EPOCH,
    DSET_FEATURES,
    DSET_HASH_INDEX,
    DSET_ROW_INDEX,
    HASH_DTYPE,
    LAYOUT_DENSE,
    SCHEMA_VERSION,
    STORAGE_PACKED,
    _acquire_lock,
    _open_or_create_h5,
)
from learnm8.features.skfp_2d.morgan import MorganFeaturizer

# ---------------------------------------------------------------------------
# Helper: synthesize a v2 (64-bit hash) cache file
# ---------------------------------------------------------------------------


def _create_v2_cache(
    path: Path,
    n_rows: int = 10,
    smiles_list: list[str] | None = None,
) -> None:
    """Fabricate a synthetic v2 HDF5 cache file with 64-bit ``hash_index``.

    The file carries ``schema_version=2`` plus all REQUIRED_ATTRS so the new
    v3 code recognises it as a legitimate v2 cache and triggers the
    ``hash64.bak`` migration branch (rather than the generic ``v0.bak`` /
    ``v1.bak`` paths).

    Args:
        path: target ``.h5`` path (parent directory must already exist).
        n_rows: number of rows to populate in ``/hash_index``, ``/row_index``
            and ``/features``. Use ``0`` for the empty edge case.
        smiles_list: optional — present only for forward-compat with the test
            spec; the helper does not actually consume it (we only need
            structural validity, not fingerprint correctness).
    """
    del smiles_list  # unused; kept for API parity with the test spec

    bit_count = 2048
    packed_width = (bit_count + 7) // 8
    with h5py.File(path, 'w') as f:
        f.attrs[ATTR_SCHEMA_VERSION] = np.uint8(2)
        f.attrs[ATTR_BIT_COUNT] = np.uint32(bit_count)
        f.attrs[ATTR_STORAGE_DTYPE] = STORAGE_PACKED
        f.attrs[ATTR_FEATURIZER_NAME] = 'morgan'
        f.attrs[ATTR_WRITE_EPOCH] = np.uint64(0)
        f.attrs[ATTR_STORAGE_LAYOUT] = LAYOUT_DENSE
        f.create_dataset(
            DSET_FEATURES,
            data=np.zeros((n_rows, packed_width), dtype=np.uint8),
            maxshape=(None, packed_width),
            dtype=np.uint8,
        )
        f.create_dataset(
            DSET_HASH_INDEX,
            data=np.arange(n_rows, dtype=np.uint64),
            maxshape=(None,),
            dtype=np.uint64,
        )
        f.create_dataset(
            DSET_ROW_INDEX,
            data=np.arange(n_rows, dtype=np.uint64),
            maxshape=(None,),
            dtype=np.uint64,
        )


def _backup_path(cache_path: Path) -> Path:
    """Compute the expected ``.hash64.bak`` sibling path for a given cache."""
    return cache_path.with_suffix(cache_path.suffix + '.hash64.bak')


# ---------------------------------------------------------------------------
# T030: v2 -> v3 migration (FR-003, FR-004, SC-005)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_v2_cache_renamed_to_hash64_bak_on_open(tmp_path: Path) -> None:
    """A v2 cache is renamed to <name>.h5.hash64.bak and a fresh v3 file is created."""
    feat = MorganFeaturizer()
    cache_path = tmp_path / f'features_{feat.get_name()}.h5'
    _create_v2_cache(cache_path, n_rows=10)

    f = _open_or_create_h5(cache_path, feat)
    try:
        assert int(f.attrs[ATTR_SCHEMA_VERSION]) == SCHEMA_VERSION
        assert f[DSET_HASH_INDEX].dtype == HASH_DTYPE
    finally:
        f.close()

    backup = _backup_path(cache_path)
    assert backup.exists(), (
        f"Expected v2 cache to be renamed to {backup.name}; "
        f"directory contents: {sorted(p.name for p in tmp_path.iterdir())}"
    )

    # The legacy backup must NOT be deleted: confirm it is still a valid v2 file.
    with h5py.File(backup, 'r') as g:
        assert int(g.attrs[ATTR_SCHEMA_VERSION]) == 2
        assert g[DSET_HASH_INDEX].dtype == np.dtype(np.uint64)


@pytest.mark.unit
def test_migration_emits_info_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """v2 -> v3 migration emits a single INFO log naming the backup path."""
    feat = MorganFeaturizer()
    cache_path = tmp_path / f'features_{feat.get_name()}.h5'
    _create_v2_cache(cache_path, n_rows=4)

    backup = _backup_path(cache_path)

    caplog.set_level(logging.INFO, logger='learnm8.features.cache')
    f = _open_or_create_h5(cache_path, feat)
    f.close()

    matching = [
        rec for rec in caplog.records
        if rec.levelno == logging.INFO
        and 'Renamed v2 (64-bit hash) cache' in rec.getMessage()
    ]
    assert matching, (
        f"Expected an INFO log mentioning 'Renamed v2 (64-bit hash) cache'; "
        f"got: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
    assert any(str(backup) in rec.getMessage() for rec in matching), (
        f"Expected the backup path {backup} to appear in the INFO log; "
        f"messages: {[r.getMessage() for r in matching]}"
    )


@pytest.mark.unit
def test_refuse_to_clobber_existing_bak(tmp_path: Path) -> None:
    """If <name>.h5.hash64.bak already exists, migration raises PersistenceError."""
    feat = MorganFeaturizer()
    cache_path = tmp_path / f'features_{feat.get_name()}.h5'
    _create_v2_cache(cache_path, n_rows=5)

    backup = _backup_path(cache_path)
    pre_existing_payload = b'pre-existing backup contents - DO NOT CLOBBER'
    backup.write_bytes(pre_existing_payload)

    with pytest.raises(PersistenceError, match=r'Refusing to migrate.*already exists'):
        _open_or_create_h5(cache_path, feat)

    # Original v2 file unchanged.
    assert cache_path.exists()
    with h5py.File(cache_path, 'r') as g:
        assert int(g.attrs[ATTR_SCHEMA_VERSION]) == 2
        assert g[DSET_HASH_INDEX].dtype == np.dtype(np.uint64)

    # Pre-existing backup unchanged.
    assert backup.read_bytes() == pre_existing_payload


@pytest.mark.unit
def test_v3_cache_no_migration_on_open(tmp_path: Path) -> None:
    """Opening a fresh v3 cache twice in succession does NOT create a .bak."""
    feat = MorganFeaturizer()
    cache_path = tmp_path / f'features_{feat.get_name()}.h5'

    # First open: file does not exist → fresh v3 created, no migration.
    f1 = _open_or_create_h5(cache_path, feat)
    f1.close()
    assert not _backup_path(cache_path).exists()
    assert not any(p.name.endswith('.bak') for p in tmp_path.iterdir())

    # Second open on the existing v3 file: must not trigger migration.
    f2 = _open_or_create_h5(cache_path, feat)
    try:
        assert int(f2.attrs[ATTR_SCHEMA_VERSION]) == SCHEMA_VERSION
    finally:
        f2.close()

    assert not _backup_path(cache_path).exists()
    assert not any(p.name.endswith('.bak') for p in tmp_path.iterdir())


@pytest.mark.unit
def test_idempotency_after_migration(tmp_path: Path) -> None:
    """Migration runs at most once: re-opening the post-migration v3 file is a no-op."""
    feat = MorganFeaturizer()
    cache_path = tmp_path / f'features_{feat.get_name()}.h5'
    _create_v2_cache(cache_path, n_rows=3)

    # First open: migrate v2 -> v3 + .hash64.bak.
    f1 = _open_or_create_h5(cache_path, feat)
    f1.close()
    backup = _backup_path(cache_path)
    assert backup.exists()
    backup_size_after_first = backup.stat().st_size

    # Second open on the new v3 file: no second migration, no double-backup.
    f2 = _open_or_create_h5(cache_path, feat)
    try:
        assert int(f2.attrs[ATTR_SCHEMA_VERSION]) == SCHEMA_VERSION
    finally:
        f2.close()

    bak_files = sorted(p.name for p in tmp_path.glob('*.bak'))
    assert bak_files == [backup.name], (
        f"Expected exactly one .bak file ({backup.name}); got {bak_files}"
    )
    assert backup.stat().st_size == backup_size_after_first


@pytest.mark.unit
def test_v2_with_empty_hash_index_migrates(tmp_path: Path) -> None:
    """Edge case: a v2 cache with N=0 rows still migrates cleanly."""
    feat = MorganFeaturizer()
    cache_path = tmp_path / f'features_{feat.get_name()}.h5'
    _create_v2_cache(cache_path, n_rows=0)

    f = _open_or_create_h5(cache_path, feat)
    try:
        assert int(f.attrs[ATTR_SCHEMA_VERSION]) == SCHEMA_VERSION
        assert f[DSET_HASH_INDEX].dtype == HASH_DTYPE
        assert f[DSET_HASH_INDEX].shape == (0,)
    finally:
        f.close()

    backup = _backup_path(cache_path)
    assert backup.exists()
    with h5py.File(backup, 'r') as g:
        assert int(g.attrs[ATTR_SCHEMA_VERSION]) == 2
        assert g[DSET_HASH_INDEX].shape == (0,)
        assert g[DSET_HASH_INDEX].dtype == np.dtype(np.uint64)


# ---------------------------------------------------------------------------
# T031: concurrent migration race (review ShouldFix #5)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_concurrent_v2_migration_race(tmp_path: Path) -> None:
    """Four threads racing to migrate the same v2 cache: exactly one .bak is produced.

    The migration is wrapped in ``LOCK_EX`` so the race is structurally
    protected. We verify the protection holds: at most one ``.hash64.bak``
    file is produced, the resulting v3 cache is internally consistent, and no
    exception other than ``PersistenceError`` (lock contention) ever surfaces.
    """
    feat = MorganFeaturizer()
    cache_path = tmp_path / f'features_{feat.get_name()}.h5'
    _create_v2_cache(cache_path, n_rows=8)

    n_threads = 4
    barrier = threading.Barrier(n_threads)
    results: list[dict[str, object]] = []
    results_lock = threading.Lock()

    def worker(idx: int) -> None:
        outcome: dict[str, object] = {'idx': idx}
        try:
            barrier.wait(timeout=5.0)
            with _acquire_lock(cache_path, mode='exclusive'):
                f = _open_or_create_h5(cache_path, feat)
                try:
                    outcome['schema_version'] = int(f.attrs[ATTR_SCHEMA_VERSION])
                finally:
                    f.close()
            outcome['error'] = None
        except PersistenceError as exc:
            outcome['error'] = ('PersistenceError', str(exc))
        except Exception as exc:
            outcome['error'] = (type(exc).__name__, str(exc))
        with results_lock:
            results.append(outcome)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(worker, i) for i in range(n_threads)]
        for fut in as_completed(futures):
            fut.result()

    # Exactly one .hash64.bak exists.
    bak_files = sorted(p.name for p in tmp_path.glob('*.hash64.bak'))
    assert bak_files == [_backup_path(cache_path).name], (
        f"Expected exactly one .hash64.bak file; got {bak_files}"
    )

    # The fresh v3 cache exists and is well-formed.
    assert cache_path.exists()
    with h5py.File(cache_path, 'r') as g:
        assert int(g.attrs[ATTR_SCHEMA_VERSION]) == SCHEMA_VERSION
        assert g[DSET_HASH_INDEX].dtype == HASH_DTYPE

    # All threads either succeeded with v3 or raised PersistenceError (lock contention).
    # No data corruption errors are tolerated.
    successes = [r for r in results if r.get('error') is None]
    persistence_errors = [
        r for r in results
        if isinstance(r.get('error'), tuple) and r['error'][0] == 'PersistenceError'
    ]
    other_errors = [
        r for r in results
        if isinstance(r.get('error'), tuple) and r['error'][0] != 'PersistenceError'
    ]

    assert not other_errors, (
        f"Unexpected non-PersistenceError exceptions in concurrent migration: "
        f"{other_errors}"
    )
    assert len(results) == n_threads
    assert successes, (
        "At least one thread must successfully open the v3 cache; "
        f"results={results}"
    )
    for r in successes:
        assert r['schema_version'] == SCHEMA_VERSION

    # Diagnostic: stash race outcome on the test for visibility on `-v --tb=long`.
    print(
        f"\n[concurrent_v2_migration_race] threads={n_threads} "
        f"successes={len(successes)} persistence_errors={len(persistence_errors)} "
        f"other_errors={len(other_errors)}"
    )

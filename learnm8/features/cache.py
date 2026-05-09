"""HDF5 v2 bit-packed cache for molecular feature extraction.

Schema v2: one consolidated 2-D ``/features`` dataset per featurizer plus
``/hash_index`` and ``/row_index`` for O(log N) lookup. Binary featurizers
round-trip through :func:`numpy.packbits` for 32x storage reduction; continuous
featurizers store float32 in place. Public ``extract_features`` signature is
unchanged. Failure mode is fail-fast: corruption or persistent OS errors raise
:class:`FeatureExtractionError` with a remediation message rather than silently
re-fingerprinting (which at 100M scale wastes ~10 hours).

Cache-key recipe (FR-004) preserves the v1 contract:
``MD5(smiles + '_' + name + '_' + config_hash)`` truncated to 8 big-endian
bytes interpreted as ``uint64``. ``hashlib.md5(..., usedforsecurity=False)`` so
LearnM8 stays operational on FIPS-locked hosts. The 64-bit truncation residual
collision probability at N=10**8 is ~0.027% — accepted as documented risk;
recoverable by deleting the cache file.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import logging
import os
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import h5py
import hdf5plugin
import numpy as np
import scipy.sparse

from learnm8.core.interfaces import Featurizer
from learnm8.exceptions import (
    ConfigurationError,
    FeatureExtractionError,
    PersistenceError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants (T005)
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 2
ATTR_SCHEMA_VERSION = 'schema_version'
ATTR_BIT_COUNT = 'bit_count'
ATTR_STORAGE_DTYPE = 'storage_dtype'
ATTR_FEATURIZER_NAME = 'featurizer_name'
ATTR_WRITE_EPOCH = 'write_epoch'
ATTR_STORAGE_LAYOUT = 'storage_layout'

LAYOUT_DENSE = 'dense'
LAYOUT_CSR = 'csr'

REQUIRED_ATTRS: tuple[str, ...] = (
    ATTR_SCHEMA_VERSION,
    ATTR_BIT_COUNT,
    ATTR_STORAGE_DTYPE,
    ATTR_FEATURIZER_NAME,
    ATTR_WRITE_EPOCH,
)

DSET_FEATURES = 'features'
DSET_HASH_INDEX = 'hash_index'
DSET_ROW_INDEX = 'row_index'
DSET_CSR_DATA = 'csr_data'
DSET_CSR_INDICES = 'csr_indices'
DSET_CSR_INDPTR = 'csr_indptr'

STORAGE_PACKED = 'packed_uint8'
STORAGE_FLOAT32 = 'float32'
STORAGE_UINT8 = 'uint8'
STORAGE_CSR_UINT16 = 'csr_uint16'
ALLOWED_STORAGE_DTYPES = frozenset(
    {STORAGE_FLOAT32, STORAGE_PACKED, STORAGE_UINT8, STORAGE_CSR_UINT16}
)

# Chunk shapes target ~1 MiB chunks (HDF5 sweet spot per microbench).
CHUNK_ROWS_PACKED = 4096
CHUNK_ROWS_FLOAT32 = 1024
CHUNK_ROWS_UINT8 = 4096
CHUNK_ROWS_CSR_DATA = 65536
CHUNK_ROWS_CSR_INDPTR = 4096

# Above this density, packed_uint8 is strictly better than CSR (3 B/nonzero).
CSR_DENSITY_HARD_LIMIT = 0.50

# uint16 column indices cap the per-row dimension at 2**16.
CSR_MAX_DIM = 65536

# HDF5 chunk cache: 16 MiB / ~1M slots — microbench winner at 1M rows.
RDCC_NBYTES = 16 * 1024 * 1024
RDCC_NSLOTS = 1_000_003

# Bound transient memory during unpackbits 8x expansion (10k rows ~ 240 MB at dim=2048).
MAX_UNPACK_CHUNK_ROWS = 10_000

# Large-merge threshold (T014): vectorized merge mandatory above this.
VECTORIZED_MERGE_THRESHOLD = 10_000_000

# h5py docs: >1000 fancy-index elements may degrade; chunk the gather.
H5_FANCY_INDEX_CHUNK = 1000

# Default lock timeout (FR-009); env-overridable.
LOCK_TIMEOUT_S: float = float(os.environ.get('LEARNM8_LOCK_TIMEOUT_S', '5.0'))

# Blosc-LZ4-5 + byte-shuffle (microbench winner; see research/blosc-microbenchmark.md).
_BLOSC_KWARGS = hdf5plugin.Blosc(
    cname='lz4',
    clevel=5,
    shuffle=hdf5plugin.Blosc.SHUFFLE,
)


# ---------------------------------------------------------------------------
# Process-level hash_index cache (post-review addendum)
# ---------------------------------------------------------------------------
# WHY: NFR-003 (<1 s open at 100M) requires that we don't re-read the 800 MB
# hash_index on every reader open. Cache key (path, mtime_ns, write_epoch)
# invalidates automatically on writer commit (write_epoch bumps under LOCK_EX).

_HASH_INDEX_CACHE: dict[tuple[str, int, int], Any] = {}


def _hash_index_cache_get(path: Path, mtime_ns: int, write_epoch: int) -> np.ndarray | None:
    key = (str(path.resolve()), mtime_ns, write_epoch)
    weak = _HASH_INDEX_CACHE.get(key)
    if weak is None:
        return None
    cached = weak()
    if cached is None:
        _HASH_INDEX_CACHE.pop(key, None)
        return None
    return cached


def _hash_index_cache_put(path: Path, mtime_ns: int, write_epoch: int, arr: np.ndarray) -> None:
    key = (str(path.resolve()), mtime_ns, write_epoch)
    with contextlib.suppress(TypeError):
        _HASH_INDEX_CACHE[key] = weakref.ref(arr)


# ---------------------------------------------------------------------------
# Storage dtype helpers (T006, T007, T008)
# ---------------------------------------------------------------------------


def _normalize_storage_dtype(dtype_str: str) -> str:
    """Validate and normalize a storage dtype string (FR-015)."""
    if dtype_str not in ALLOWED_STORAGE_DTYPES:
        raise ConfigurationError(
            f"unknown storage_dtype: {dtype_str!r}; supported: "
            f"{sorted(ALLOWED_STORAGE_DTYPES)}. "
            f"Override Featurizer.get_storage_dtype() to one of these."
        )
    return dtype_str


def _validate_uint8_range(features: np.ndarray) -> None:
    if features.size == 0:
        return
    arr_min = float(features.min())
    arr_max = float(features.max())
    if arr_min < 0.0 or arr_max > 255.0:
        offenders = np.argwhere((features < 0) | (features > 255))
        row, col = (int(offenders[0, 0]), int(offenders[0, 1])) if offenders.size else (-1, -1)
        bad_value = float(features[row, col]) if row >= 0 else arr_max
        raise FeatureExtractionError(
            f"uint8 storage out of range: row={row} col={col} value={bad_value} "
            f"(observed min={arr_min}, max={arr_max}; uint8 admits [0, 255]). "
            f"Either set this featurizer's get_storage_dtype() to 'float32' or "
            f"contact maintainers about a wider int dtype."
        )


def to_storage(features: np.ndarray, storage_dtype: str) -> np.ndarray:
    """Encode a feature batch for on-disk storage."""
    storage_dtype = _normalize_storage_dtype(storage_dtype)
    if storage_dtype == STORAGE_PACKED:
        return np.packbits(
            np.ascontiguousarray(features, dtype=np.uint8),
            axis=1,
            bitorder='big',
        )
    if storage_dtype == STORAGE_UINT8:
        _validate_uint8_range(features)
        return np.ascontiguousarray(features, dtype=np.uint8)
    if storage_dtype == STORAGE_CSR_UINT16:
        raise FeatureExtractionError(
            f"to_storage() does not encode {STORAGE_CSR_UINT16!r}; "
            f"the cache writer dispatches CSR rows directly. "
            f"This call indicates a bug in the cache write path."
        )
    return np.ascontiguousarray(features, dtype=np.float32)


def from_storage(
    stored: np.ndarray,
    storage_dtype: str,
    bit_count: int,
    max_chunk_rows: int = MAX_UNPACK_CHUNK_ROWS,
) -> np.ndarray:
    """Decode an on-disk batch to ``float32``.

    Chunk-streaming bounds transient memory: unpackbits + cast for an N-row
    batch otherwise allocates ~12*N*bit_count bytes simultaneously
    (uint8 unpacked + float32 cast both alive). Streaming caps that at
    ~12*max_chunk_rows*bit_count.
    """
    storage_dtype = _normalize_storage_dtype(storage_dtype)
    if storage_dtype == STORAGE_FLOAT32:
        return stored if stored.dtype == np.float32 else stored.astype(np.float32, copy=False)
    if storage_dtype == STORAGE_UINT8:
        return stored.astype(np.float32, copy=False)
    if storage_dtype == STORAGE_CSR_UINT16:
        raise FeatureExtractionError(
            f"from_storage() does not decode {STORAGE_CSR_UINT16!r}; "
            f"the cache reader dispatches CSR rows directly. "
            f"This call indicates a bug in the cache read path."
        )

    n = stored.shape[0]
    out = np.empty((n, bit_count), dtype=np.float32)
    if n == 0:
        return out
    chunk = max(1, max_chunk_rows)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        unpacked = np.unpackbits(stored[start:end], axis=1, count=bit_count, bitorder='big')
        out[start:end] = unpacked.astype(np.float32, copy=False)
    return out


# ---------------------------------------------------------------------------
# Locking (T026)
# ---------------------------------------------------------------------------


def _lock_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + '.lock')


def _read_lock_holder_pid(lock_path: Path) -> str:
    """Best-effort PID lookup via /proc/locks; degrade gracefully."""
    proc_locks = Path('/proc/locks')
    if not proc_locks.exists():
        return 'PID unknown'
    try:
        st = lock_path.stat()
        target_inode = st.st_ino
        target_dev = st.st_dev
        # /proc/locks format: id: type kind access pid major:minor:inode start end
        major = os.major(target_dev)
        minor = os.minor(target_dev)
        with proc_locks.open('r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 8:
                    continue
                pid = parts[4]
                location = parts[5]
                try:
                    maj_str, min_str, inode_str = location.split(':')
                except ValueError:
                    continue
                if (
                    int(inode_str) == target_inode
                    and int(maj_str, 16) == major
                    and int(min_str, 16) == minor
                ):
                    return pid
    except (OSError, ValueError):
        return 'PID unknown'
    return 'PID unknown'


@contextmanager
def _acquire_lock(
    cache_path: Path,
    mode: str,
    timeout_s: float | None = None,
) -> Iterator[None]:
    """Acquire ``fcntl.flock`` on the sidecar ``.lock`` file.

    Args:
        cache_path: target ``.h5`` path; lock is held on ``<path>.lock``.
        mode: ``'exclusive'`` (LOCK_EX) for writers, ``'shared'`` (LOCK_SH) for readers.
        timeout_s: max seconds to retry; ``None`` uses :data:`LOCK_TIMEOUT_S`.

    Raises:
        FeatureExtractionError: timeout exceeded.
    """
    if mode == 'exclusive':
        flag = fcntl.LOCK_EX | fcntl.LOCK_NB
    elif mode == 'shared':
        flag = fcntl.LOCK_SH | fcntl.LOCK_NB
    else:
        raise ValueError(f"lock mode must be 'exclusive' or 'shared', got {mode!r}")

    timeout = LOCK_TIMEOUT_S if timeout_s is None else timeout_s
    lock_path = _lock_path(cache_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        deadline = time.monotonic() + max(0.0, timeout)
        backoff = 0.01
        while True:
            try:
                fcntl.flock(fd, flag)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    holder = _read_lock_holder_pid(lock_path)
                    raise FeatureExtractionError(
                        f"Failed to acquire {mode} lock on {lock_path} within "
                        f"{timeout:.2f}s (holder PID: {holder}). "
                        f"Another LearnM8 process is using this cache; wait or "
                        f"set LEARNM8_LOCK_TIMEOUT_S to a larger value."
                    ) from None
                time.sleep(backoff)
                backoff = min(backoff * 2, 0.5)
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Retry seam (T013)
# ---------------------------------------------------------------------------

T = TypeVar('T')

_TRANSIENT_ERRNOS = frozenset({errno.EAGAIN, errno.EINTR, errno.EIO})


def _read_with_one_retry(reader_callable: Callable[[], T]) -> T:
    """Call ``reader_callable``; on transient ``OSError`` retry exactly once.

    FR-010: transient errnos (EAGAIN, EINTR, EIO) get one retry with a single
    WARNING; persistent or non-transient errors propagate immediately.
    """
    try:
        return reader_callable()
    except OSError as exc:
        if getattr(exc, 'errno', None) not in _TRANSIENT_ERRNOS:
            raise
        logger.warning(
            "Transient OSError (errno=%s) on cache read; retrying once: %s",
            exc.errno,
            exc,
        )
        return reader_callable()


# ---------------------------------------------------------------------------
# Cache-key recipe (FR-004)
# ---------------------------------------------------------------------------


def _generate_cache_key(smiles: str, featurizer: Featurizer) -> str:
    """Generate the v1-compatible 32-char hex MD5 cache key (kept for tests)."""
    config_hash = featurizer.get_config_hash()
    featurizer_name = featurizer.get_name()
    cache_string = f"{smiles}_{featurizer_name}_{config_hash}"
    return hashlib.md5(cache_string.encode('utf-8'), usedforsecurity=False).hexdigest()


def _cache_keys_uint64(smiles_list: list[str], featurizer: Featurizer) -> np.ndarray:
    """Vectorized 8-byte big-endian truncation of MD5 cache keys to ``uint64``."""
    name = featurizer.get_name()
    config_hash = featurizer.get_config_hash()
    out = np.empty(len(smiles_list), dtype=np.uint64)
    for i, s in enumerate(smiles_list):
        digest = hashlib.md5(
            f"{s}_{name}_{config_hash}".encode(),
            usedforsecurity=False,
        ).digest()
        out[i] = int.from_bytes(digest[:8], byteorder='big', signed=False)
    return out


def get_smiles_hash(smiles: str) -> str:
    """Legacy SMILES-only MD5 (kept for backward compatibility in tests)."""
    return hashlib.md5(smiles.encode('utf-8'), usedforsecurity=False).hexdigest()


# ---------------------------------------------------------------------------
# Vectorized merge (T014)
# ---------------------------------------------------------------------------


def _merge_sorted_indices(
    h_old: np.ndarray,
    r_old: np.ndarray,
    h_new: np.ndarray,
    r_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge two parallel ``(hash, row)`` arrays preserving sort order.

    For ``len(h_old) > VECTORIZED_MERGE_THRESHOLD`` uses vectorized
    ``np.searchsorted``+``np.insert``. Smaller inputs use ``np.argsort`` on the
    concatenation (TimSort detects sorted runs). Both paths reject duplicate
    hashes (FR-010 tightening).
    """
    if h_old.size > VECTORIZED_MERGE_THRESHOLD:
        sort_perm = np.argsort(h_new, kind='stable')
        h_new_sorted = h_new[sort_perm]
        r_new_sorted = r_new[sort_perm]
        pos = np.searchsorted(h_old, h_new_sorted, side='left')
        out_h = np.insert(h_old, pos, h_new_sorted)
        out_r = np.insert(r_old, pos, r_new_sorted)
    else:
        out_h_unsorted = np.concatenate([h_old, h_new])
        out_r_unsorted = np.concatenate([r_old, r_new])
        order = np.argsort(out_h_unsorted, kind='stable')
        out_h = out_h_unsorted[order]
        out_r = out_r_unsorted[order]

    if out_h.size > 1 and not np.all(np.diff(out_h) > 0):
        raise FeatureExtractionError(
            "Duplicate cache_key detected during merge — cache file is "
            "corrupt. Delete the cache file and retry."
        )
    return out_h, out_r


# ---------------------------------------------------------------------------
# v2 file open / init (T009, T010, T029)
# ---------------------------------------------------------------------------


def _packed_width(bit_count: int, storage_dtype: str) -> int:
    if storage_dtype == STORAGE_PACKED:
        return (bit_count + 7) // 8
    return bit_count


def _chunk_shape(bit_count: int, storage_dtype: str) -> tuple[int, int]:
    width = _packed_width(bit_count, storage_dtype)
    if storage_dtype == STORAGE_PACKED:
        return (CHUNK_ROWS_PACKED, max(1, width))
    if storage_dtype == STORAGE_UINT8:
        return (CHUNK_ROWS_UINT8, max(1, width))
    return (CHUNK_ROWS_FLOAT32, max(1, width))


def _h5_open(path: Path, mode: str) -> h5py.File:
    return h5py.File(
        path,
        mode,
        libver='latest',
        rdcc_nbytes=RDCC_NBYTES,
        rdcc_nslots=RDCC_NSLOTS,
    )


def _dense_np_dtype(storage_dtype: str) -> Any:
    if storage_dtype == STORAGE_PACKED:
        return np.uint8
    if storage_dtype == STORAGE_UINT8:
        return np.uint8
    return np.float32


def _initialize_v2_csr_datasets(
    f: h5py.File, featurizer: Featurizer, storage_dtype: str
) -> None:
    """Create CSR-layout datasets (csr_data/csr_indices/csr_indptr + hash/row indices)."""
    bit_count = int(featurizer.get_dimension())
    if bit_count > CSR_MAX_DIM - 1:
        raise ConfigurationError(
            f"csr_uint16 storage requires dim <= {CSR_MAX_DIM - 1} (uint16 column "
            f"indices), got dim={bit_count} for featurizer "
            f"{featurizer.get_name()!r}. Either reduce dim or set "
            f"get_storage_dtype() to 'packed_uint8' / 'float32'."
        )

    f.create_dataset(
        DSET_CSR_DATA,
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint8,
        chunks=(CHUNK_ROWS_CSR_DATA,),
        **_BLOSC_KWARGS,
    )
    f.create_dataset(
        DSET_CSR_INDICES,
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint16,
        chunks=(CHUNK_ROWS_CSR_DATA,),
        **_BLOSC_KWARGS,
    )
    indptr = f.create_dataset(
        DSET_CSR_INDPTR,
        shape=(1,),
        maxshape=(None,),
        dtype=np.uint64,
        chunks=(CHUNK_ROWS_CSR_INDPTR,),
        **_BLOSC_KWARGS,
    )
    indptr[0] = 0
    f.create_dataset(
        DSET_HASH_INDEX,
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint64,
    )
    f.create_dataset(
        DSET_ROW_INDEX,
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint64,
    )

    f.attrs[ATTR_SCHEMA_VERSION] = np.uint8(SCHEMA_VERSION)
    f.attrs[ATTR_BIT_COUNT] = np.uint32(bit_count)
    f.attrs[ATTR_STORAGE_DTYPE] = storage_dtype
    f.attrs[ATTR_FEATURIZER_NAME] = featurizer.get_name()
    f.attrs[ATTR_WRITE_EPOCH] = np.uint64(0)
    f.attrs[ATTR_STORAGE_LAYOUT] = LAYOUT_CSR
    f.flush()


def _initialize_v2_datasets(
    f: h5py.File, featurizer: Featurizer, storage_dtype: str
) -> None:
    """Create empty v2 datasets and write root attrs (T010, T014)."""
    storage_dtype = _normalize_storage_dtype(storage_dtype)
    if storage_dtype == STORAGE_CSR_UINT16:
        _initialize_v2_csr_datasets(f, featurizer, storage_dtype)
        return

    bit_count = int(featurizer.get_dimension())
    width = _packed_width(bit_count, storage_dtype)
    chunks = _chunk_shape(bit_count, storage_dtype)
    np_dtype = _dense_np_dtype(storage_dtype)

    f.create_dataset(
        DSET_FEATURES,
        shape=(0, width),
        maxshape=(None, width),
        dtype=np_dtype,
        chunks=chunks,
        **_BLOSC_KWARGS,
    )
    f.create_dataset(
        DSET_HASH_INDEX,
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint64,
    )
    f.create_dataset(
        DSET_ROW_INDEX,
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint64,
    )

    f.attrs[ATTR_SCHEMA_VERSION] = np.uint8(SCHEMA_VERSION)
    f.attrs[ATTR_BIT_COUNT] = np.uint32(bit_count)
    f.attrs[ATTR_STORAGE_DTYPE] = storage_dtype
    f.attrs[ATTR_FEATURIZER_NAME] = featurizer.get_name()
    f.attrs[ATTR_WRITE_EPOCH] = np.uint64(0)
    f.attrs[ATTR_STORAGE_LAYOUT] = LAYOUT_DENSE
    f.flush()


def _create_fresh_v2(path: Path, featurizer: Featurizer, storage_dtype: str) -> None:
    with _h5_open(path, 'w') as f:
        _initialize_v2_datasets(f, featurizer, storage_dtype)


def _atomic_rename(src: Path, dst: Path) -> None:
    try:
        os.replace(str(src), str(dst))
    except OSError as exc:
        raise PersistenceError(
            f"Failed to rename cache file {src} → {dst}: {exc}. "
            f"The original file has not been modified."
        ) from exc


def _open_or_create_h5(
    cache_path: Path, featurizer: Featurizer
) -> h5py.File:
    """Probe-and-create-or-open under the writer's exclusive lock (T009 + T029).

    Caller MUST already hold ``LOCK_EX`` on ``cache_path.lock``. The probe step
    must happen under the writer lock so a concurrent writer cannot rename the
    file mid-probe.
    """
    storage_dtype = _normalize_storage_dtype(featurizer.get_storage_dtype())
    expected_dim = int(featurizer.get_dimension())

    if not cache_path.exists():
        _create_fresh_v2(cache_path, featurizer, storage_dtype)
        return _h5_open(cache_path, 'a')

    schema_version: int | None
    bit_count: int | None
    file_storage_dtype: str | None
    try:
        with _h5_open(cache_path, 'r') as probe:
            sv = probe.attrs.get(ATTR_SCHEMA_VERSION, None)
            bc = probe.attrs.get(ATTR_BIT_COUNT, None)
            sd = probe.attrs.get(ATTR_STORAGE_DTYPE, None)
            schema_version = int(sv) if sv is not None else None
            bit_count = int(bc) if bc is not None else None
            file_storage_dtype = str(sd) if sd is not None else None
    except (OSError, KeyError):
        schema_version = None
        bit_count = None
        file_storage_dtype = None

    if schema_version != SCHEMA_VERSION:
        suffix_tag = (
            f'v{schema_version}' if schema_version is not None else 'v1'
        )
        backup = cache_path.with_suffix(cache_path.suffix + f'.{suffix_tag}.bak')
        _atomic_rename(cache_path, backup)
        logger.warning(
            "Renamed non-v2 cache to %s; starting fresh v2 cache.", backup
        )
        _create_fresh_v2(cache_path, featurizer, storage_dtype)
        return _h5_open(cache_path, 'a')

    if bit_count != expected_dim:
        tag = f'dim{bit_count}' if bit_count is not None else 'dimunknown'
        backup = cache_path.with_suffix(cache_path.suffix + f'.{tag}.bak')
        _atomic_rename(cache_path, backup)
        logger.warning(
            "Renamed cache with mismatched bit_count (%s vs expected %d) "
            "to %s; starting fresh v2 cache.",
            bit_count,
            expected_dim,
            backup,
        )
        _create_fresh_v2(cache_path, featurizer, storage_dtype)
        return _h5_open(cache_path, 'a')

    if file_storage_dtype is not None and file_storage_dtype != storage_dtype:
        # Distinguish legitimate stale cache (user flipped get_storage_dtype()) from
        # genuine corruption (attr tampered or partial write). Validate against the
        # file's own recorded dtype: if the file is internally consistent, it's just
        # stale → rename to .bak. If inconsistent, propagate the validation error.
        with _h5_open(cache_path, 'r') as probe:
            _validate_v2_file_integrity(probe, featurizer, cache_path)
        tag = f'dim{bit_count}'
        backup = cache_path.with_suffix(cache_path.suffix + f'.{tag}.bak')
        _atomic_rename(cache_path, backup)
        logger.warning(
            "Renamed cache with mismatched storage_dtype (%s vs expected %s) "
            "to %s; starting fresh v2 cache.",
            file_storage_dtype,
            storage_dtype,
            backup,
        )
        _create_fresh_v2(cache_path, featurizer, storage_dtype)
        return _h5_open(cache_path, 'a')

    return _h5_open(cache_path, 'a')


# ---------------------------------------------------------------------------
# Integrity validation (T022)
# ---------------------------------------------------------------------------


_DENSE_EXPECTED_FEATURES_DTYPE: dict[str, Any] = {
    STORAGE_PACKED: np.uint8,
    STORAGE_FLOAT32: np.float32,
    STORAGE_UINT8: np.uint8,
}


def _validate_v2_file_integrity(
    f: h5py.File, featurizer: Featurizer, cache_path: Path
) -> None:
    """Check the contract invariants; on violation raise ``FeatureExtractionError``."""
    msg_suffix = (
        f"\nDelete the cache file ({cache_path}) and retry. "
        f"This file will not be used until removed."
    )

    for attr in REQUIRED_ATTRS:
        if attr not in f.attrs:
            raise FeatureExtractionError(
                f"v2 cache missing required attr {attr!r}." + msg_suffix
            )

    if int(f.attrs[ATTR_SCHEMA_VERSION]) != SCHEMA_VERSION:
        raise FeatureExtractionError(
            f"v2 cache schema_version mismatch (got {int(f.attrs[ATTR_SCHEMA_VERSION])}, "
            f"expected {SCHEMA_VERSION})." + msg_suffix
        )

    storage_dtype = _normalize_storage_dtype(str(f.attrs[ATTR_STORAGE_DTYPE]))
    layout = str(f.attrs.get(ATTR_STORAGE_LAYOUT, LAYOUT_DENSE))

    for dset in (DSET_HASH_INDEX, DSET_ROW_INDEX):
        if dset not in f:
            raise FeatureExtractionError(
                f"v2 cache missing dataset {dset!r}." + msg_suffix
            )
    hash_index = f[DSET_HASH_INDEX]
    row_index = f[DSET_ROW_INDEX]

    if layout == LAYOUT_DENSE:
        if DSET_FEATURES not in f:
            raise FeatureExtractionError(
                f"v2 cache missing dataset {DSET_FEATURES!r}." + msg_suffix
            )
        features = f[DSET_FEATURES]

        expected_features_dtype = _DENSE_EXPECTED_FEATURES_DTYPE[storage_dtype]
        if features.dtype != np.dtype(expected_features_dtype):
            raise FeatureExtractionError(
                f"v2 cache /features dtype mismatch: file says storage_dtype={storage_dtype!r} "
                f"but /features.dtype={features.dtype!r}." + msg_suffix
            )

        bit_count = int(f.attrs[ATTR_BIT_COUNT])
        expected_width = _packed_width(bit_count, storage_dtype)
        if features.ndim != 2 or features.shape[1] != expected_width:
            raise FeatureExtractionError(
                f"v2 cache /features shape mismatch: expected (*, {expected_width}) "
                f"for bit_count={bit_count} storage_dtype={storage_dtype!r}, "
                f"got {features.shape}." + msg_suffix
            )

        if len(row_index) > 0:
            max_row = int(row_index[:].max())
            if max_row >= features.shape[0]:
                raise FeatureExtractionError(
                    f"v2 cache row_index points past /features rows "
                    f"(max={max_row}, /features rows={features.shape[0]})." + msg_suffix
                )
    elif layout == LAYOUT_CSR:
        if storage_dtype != STORAGE_CSR_UINT16:
            raise FeatureExtractionError(
                f"v2 cache storage_layout='csr' requires storage_dtype='csr_uint16', "
                f"got {storage_dtype!r}." + msg_suffix
            )
        for dset, expected_dtype in (
            (DSET_CSR_DATA, np.uint8),
            (DSET_CSR_INDICES, np.uint16),
            (DSET_CSR_INDPTR, np.uint64),
        ):
            if dset not in f:
                raise FeatureExtractionError(
                    f"v2 cache missing dataset {dset!r}." + msg_suffix
                )
            if f[dset].dtype != np.dtype(expected_dtype):
                raise FeatureExtractionError(
                    f"v2 cache {dset!r} dtype mismatch: expected {np.dtype(expected_dtype)!r}, "
                    f"got {f[dset].dtype!r}." + msg_suffix
                )

        indptr = f[DSET_CSR_INDPTR]
        csr_data = f[DSET_CSR_DATA]
        csr_indices = f[DSET_CSR_INDICES]

        if indptr.shape[0] != row_index.shape[0] + 1:
            raise FeatureExtractionError(
                f"v2 cache csr_indptr length mismatch: expected {row_index.shape[0] + 1} "
                f"(row_index + 1), got {indptr.shape[0]}." + msg_suffix
            )

        if indptr.shape[0] >= 1 and int(indptr[0]) != 0:
            raise FeatureExtractionError(
                f"v2 cache csr_indptr[0] must be 0, got {int(indptr[0])}." + msg_suffix
            )

        if indptr.shape[0] > 0:
            last = int(indptr[-1])
            if last != csr_data.shape[0] or last != csr_indices.shape[0]:
                raise FeatureExtractionError(
                    f"v2 cache csr_indptr[-1]={last} disagrees with csr_data "
                    f"({csr_data.shape[0]}) / csr_indices ({csr_indices.shape[0]})." + msg_suffix
                )

        if indptr.shape[0] > 1:
            diffs = np.diff(np.asarray(indptr[:], dtype=np.int64))
            if np.any(diffs < 0):
                raise FeatureExtractionError(
                    "v2 cache csr_indptr is not monotone non-decreasing." + msg_suffix
                )
    else:
        raise FeatureExtractionError(
            f"v2 cache has unknown storage_layout {layout!r}." + msg_suffix
        )

    if len(hash_index) != len(row_index):
        raise FeatureExtractionError(
            f"v2 cache hash_index/row_index length mismatch "
            f"({len(hash_index)} vs {len(row_index)})." + msg_suffix
        )

    if len(hash_index) > 1:
        diffs = np.diff(hash_index[:])
        if np.any(diffs <= 0):
            raise FeatureExtractionError(
                "v2 cache hash_index is not strictly increasing "
                "(duplicates or out-of-order)." + msg_suffix
            )

    if str(f.attrs[ATTR_FEATURIZER_NAME]) != featurizer.get_name():
        raise FeatureExtractionError(
            f"v2 cache featurizer_name mismatch: file says "
            f"{str(f.attrs[ATTR_FEATURIZER_NAME])!r}, expected "
            f"{featurizer.get_name()!r}." + msg_suffix
        )


# ---------------------------------------------------------------------------
# Lookup, read, write (T011, T012, T015)
# ---------------------------------------------------------------------------


def _load_hash_index(f: h5py.File, cache_path: Path) -> np.ndarray:
    """Return ``/hash_index`` as a uint64 array, using the process cache."""
    try:
        mtime_ns = os.stat(cache_path).st_mtime_ns
    except OSError:
        mtime_ns = 0
    write_epoch = int(f.attrs.get(ATTR_WRITE_EPOCH, 0))
    cached = _hash_index_cache_get(cache_path, mtime_ns, write_epoch)
    if cached is not None:
        return cached
    arr = np.asarray(f[DSET_HASH_INDEX][:], dtype=np.uint64)
    _hash_index_cache_put(cache_path, mtime_ns, write_epoch, arr)
    return arr


def _lookup_cache(
    f: h5py.File, query_hashes: np.ndarray, cache_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(hits_mask, target_rows_for_hits)`` for a batch of query hashes."""
    n_query = query_hashes.shape[0]
    if len(f[DSET_HASH_INDEX]) == 0:
        return np.zeros(n_query, dtype=bool), np.empty(0, dtype=np.uint64)

    hash_index = _load_hash_index(f, cache_path)
    pos = np.searchsorted(hash_index, query_hashes)
    in_range = pos < hash_index.size
    matched_pos = np.where(in_range, pos, 0)
    found = in_range & (hash_index[matched_pos] == query_hashes)
    if np.any(found):
        row_index = np.asarray(f[DSET_ROW_INDEX][:], dtype=np.uint64)
        target_rows = row_index[matched_pos[found]]
    else:
        target_rows = np.empty(0, dtype=np.uint64)
    return found, target_rows


def _read_cache_hits_csr(
    f: h5py.File,
    target_rows: np.ndarray,
    bit_count: int,
) -> np.ndarray:
    """Gather hit rows from CSR datasets and materialize a dense float32 matrix.

    Reads csr_indptr in full (small: uint64 * (n_rows+1) ≈ 800 KB at 100k rows),
    then issues a single contiguous slab read per dense run of unique rows so
    Blosc chunk decompression is amortized across many rows rather than
    re-decompressed per row. At 0.7% density and 100k unique rows that drops
    HDF5 slice ops from 200k (one slice per row per dataset) to O(runs) — at
    least 100x for a fully-dense sweep.
    """
    if target_rows.size == 0:
        return np.empty((0, bit_count), dtype=np.float32)

    unique_rows, inverse_unique = np.unique(target_rows, return_inverse=True)
    n_unique = unique_rows.size
    out_unique = np.zeros((n_unique, bit_count), dtype=np.float32)

    indptr_dset = f[DSET_CSR_INDPTR]
    data_dset = f[DSET_CSR_DATA]
    indices_dset = f[DSET_CSR_INDICES]
    indptr_full = np.asarray(indptr_dset[:], dtype=np.uint64)

    diffs = np.diff(unique_rows.astype(np.int64))
    run_starts = np.concatenate(([0], np.where(diffs != 1)[0] + 1, [n_unique]))
    for run_idx in range(run_starts.size - 1):
        run_lo = int(run_starts[run_idx])
        run_hi = int(run_starts[run_idx + 1])
        first_row = int(unique_rows[run_lo])
        last_row = int(unique_rows[run_hi - 1])
        slab_start = int(indptr_full[first_row])
        slab_end = int(indptr_full[last_row + 1])
        if slab_end == slab_start:
            continue
        cols_slab = np.asarray(indices_dset[slab_start:slab_end], dtype=np.uint16)
        vals_slab = np.asarray(data_dset[slab_start:slab_end], dtype=np.uint8)
        run_unique = unique_rows[run_lo:run_hi].astype(np.int64)
        run_starts_local = indptr_full[run_unique].astype(np.int64) - slab_start
        run_ends_local = indptr_full[run_unique + 1].astype(np.int64) - slab_start
        per_row_counts = (run_ends_local - run_starts_local).astype(np.int64)
        row_for_each_nnz = np.repeat(
            np.arange(run_lo, run_hi, dtype=np.int64), per_row_counts
        )
        out_unique[row_for_each_nnz, cols_slab] = vals_slab.astype(np.float32, copy=False)

    return out_unique[inverse_unique]


def _read_cache_hits(
    f: h5py.File,
    target_rows: np.ndarray,
    storage_dtype: str,
    bit_count: int,
    output_dtype: str = 'float32',
) -> np.ndarray:
    """Gather + dispatch-decode hit rows. Maintains caller order.

    ``output_dtype='uint8'`` short-circuits the float32 inflation for
    ``packed_uint8`` / ``uint8`` storage (REQ-9). CSR storage always returns
    float32 regardless (REQ-10).

    h5py fancy indexing requires strictly-increasing indices; duplicates in the
    caller's query (same SMILES twice) collapse to a single read and re-expand.
    """
    out_np_dtype = np.uint8 if output_dtype == 'uint8' else np.float32
    if target_rows.size == 0:
        return np.empty((0, bit_count), dtype=out_np_dtype)

    if storage_dtype == STORAGE_CSR_UINT16:
        # CSR materialises float32; uint8 callers were diverted upstream.
        return _read_cache_hits_csr(f, target_rows, bit_count)

    unique_rows, inverse_unique = np.unique(target_rows, return_inverse=True)
    n_unique = unique_rows.size
    width = _packed_width(bit_count, storage_dtype)
    np_dtype = _dense_np_dtype(storage_dtype)
    raw = np.empty((n_unique, width), dtype=np_dtype)
    features = f[DSET_FEATURES]
    for start in range(0, n_unique, H5_FANCY_INDEX_CHUNK):
        end = min(start + H5_FANCY_INDEX_CHUNK, n_unique)
        idx = unique_rows[start:end].tolist()
        raw[start:end] = features[idx, :]

    if output_dtype == 'uint8':
        decoded_unique = from_storage_uint8(raw, storage_dtype, bit_count)
    else:
        decoded_unique = from_storage(raw, storage_dtype, bit_count)
    return decoded_unique[inverse_unique]


def _write_csr_misses(
    f: h5py.File,
    miss_features: np.ndarray,
    miss_hashes: np.ndarray,
) -> None:
    """Append CSR-encoded miss rows. Validates density + uint8 bounds BEFORE any HDF5 writes."""
    new_n = int(miss_features.shape[0])
    dim = int(miss_features.shape[1])

    density = float((miss_features != 0).mean()) if miss_features.size > 0 else 0.0
    if density > CSR_DENSITY_HARD_LIMIT:
        raise FeatureExtractionError(
            f"observed density {density:.4f} exceeds {CSR_DENSITY_HARD_LIMIT} "
            f"hard limit on csr_uint16 storage. CSR overhead exceeds dense storage "
            f"above 50% density; switch this featurizer's get_storage_dtype() to "
            f"'packed_uint8'."
        )

    if miss_features.size > 0:
        arr_min = float(miss_features.min())
        arr_max = float(miss_features.max())
        if arr_min < 0.0 or arr_max > 255.0:
            offenders = np.argwhere((miss_features < 0) | (miss_features > 255))
            row, col = (int(offenders[0, 0]), int(offenders[0, 1])) if offenders.size else (-1, -1)
            bad_value = float(miss_features[row, col]) if row >= 0 else arr_max
            raise FeatureExtractionError(
                f"csr_uint16 nonzero out of range: row={row} col={col} value={bad_value} "
                f"(observed min={arr_min}, max={arr_max}; uint8 csr_data admits [0, 255]). "
                f"Either reduce the value range or switch storage to 'float32'."
            )

    csr = scipy.sparse.csr_matrix(miss_features.astype(np.uint8))
    new_indices = np.asarray(csr.indices, dtype=np.uint16)
    new_data = np.asarray(csr.data, dtype=np.uint8)
    new_indptr = np.asarray(csr.indptr, dtype=np.uint64)

    if new_indices.size > 0 and int(new_indices.max()) >= dim:
        raise FeatureExtractionError(
            f"csr_uint16 column index {int(new_indices.max())} >= dim {dim}; "
            f"this indicates a featurizer bug or dim mismatch."
        )

    indptr_dset = f[DSET_CSR_INDPTR]
    data_dset = f[DSET_CSR_DATA]
    indices_dset = f[DSET_CSR_INDICES]

    prev_nnz = int(indptr_dset[-1])
    prev_n_rows = int(indptr_dset.shape[0]) - 1

    nnz = int(new_data.shape[0])
    data_dset.resize((prev_nnz + nnz,))
    data_dset[prev_nnz : prev_nnz + nnz] = new_data
    indices_dset.resize((prev_nnz + nnz,))
    indices_dset[prev_nnz : prev_nnz + nnz] = new_indices

    indptr_offsets = (new_indptr[1:].astype(np.uint64) + np.uint64(prev_nnz)).astype(np.uint64)
    indptr_dset.resize((prev_n_rows + 1 + new_n,))
    indptr_dset[prev_n_rows + 1 : prev_n_rows + 1 + new_n] = indptr_offsets

    new_row_index = np.arange(prev_n_rows, prev_n_rows + new_n, dtype=np.uint64)
    new_hash_index = np.asarray(miss_hashes, dtype=np.uint64)

    h_old = np.asarray(f[DSET_HASH_INDEX][:], dtype=np.uint64)
    r_old = np.asarray(f[DSET_ROW_INDEX][:], dtype=np.uint64)
    merged_h, merged_r = _merge_sorted_indices(h_old, r_old, new_hash_index, new_row_index)

    hash_dset = f[DSET_HASH_INDEX]
    row_dset = f[DSET_ROW_INDEX]
    hash_dset.resize((merged_h.size,))
    hash_dset[:] = merged_h
    row_dset.resize((merged_r.size,))
    row_dset[:] = merged_r

    f.attrs[ATTR_WRITE_EPOCH] = np.uint64(int(f.attrs.get(ATTR_WRITE_EPOCH, 0)) + 1)
    f.flush()


def _write_cache_misses(
    f: h5py.File,
    miss_smiles: list[str],
    miss_features: np.ndarray,
    miss_hashes: np.ndarray,
    storage_dtype: str,
) -> None:
    """Append miss rows; dispatches dense vs CSR by storage_dtype (T015)."""
    if len(miss_smiles) == 0:
        return

    storage_dtype = _normalize_storage_dtype(storage_dtype)
    if storage_dtype == STORAGE_CSR_UINT16:
        _write_csr_misses(f, miss_features, miss_hashes)
        return

    encoded = to_storage(miss_features, storage_dtype)

    features = f[DSET_FEATURES]
    cur_n = features.shape[0]
    new_n = encoded.shape[0]
    features.resize((cur_n + new_n, features.shape[1]))
    features[cur_n : cur_n + new_n] = encoded

    new_row_index = np.arange(cur_n, cur_n + new_n, dtype=np.uint64)
    new_hash_index = np.asarray(miss_hashes, dtype=np.uint64)

    h_old = np.asarray(f[DSET_HASH_INDEX][:], dtype=np.uint64)
    r_old = np.asarray(f[DSET_ROW_INDEX][:], dtype=np.uint64)

    merged_h, merged_r = _merge_sorted_indices(h_old, r_old, new_hash_index, new_row_index)

    hash_dset = f[DSET_HASH_INDEX]
    row_dset = f[DSET_ROW_INDEX]
    hash_dset.resize((merged_h.size,))
    hash_dset[:] = merged_h
    row_dset.resize((merged_r.size,))
    row_dset[:] = merged_r

    f.attrs[ATTR_WRITE_EPOCH] = np.uint64(int(f.attrs.get(ATTR_WRITE_EPOCH, 0)) + 1)
    f.flush()


# ---------------------------------------------------------------------------
# Decorator + main entry (T016)
# ---------------------------------------------------------------------------


def _resolve_cache_dir(default_cache_dir: Path, kwargs: dict[str, Any]) -> Path:
    cache_dir = kwargs.pop('cache_dir', None)
    if cache_dir is None:
        cache_dir = default_cache_dir
    return Path(cache_dir)


def _resolve_uint8_output(
    preferred_dtype: str,
    storage_dtype: str,
    featurizer_name: str,
) -> bool:
    """Decide whether to allocate a uint8 output buffer (REQ-9, REQ-10).

    Returns True only when caller asked for uint8 AND storage actually has the
    bits handy (packed_uint8 / uint8). For float32 / csr_uint16 storage we
    must materialise float32 — log once and fall back.
    """
    if preferred_dtype != 'uint8':
        return False
    if storage_dtype in (STORAGE_PACKED, STORAGE_UINT8):
        return True
    logger.debug(
        "preferred_dtype='uint8' incompatible with storage_dtype=%r for "
        "featurizer %r; falling back to float32 output",
        storage_dtype,
        featurizer_name,
    )
    return False


def from_storage_uint8(
    stored: np.ndarray,
    storage_dtype: str,
    bit_count: int,
    max_chunk_rows: int = MAX_UNPACK_CHUNK_ROWS,
) -> np.ndarray:
    """Decode an on-disk batch directly to ``uint8``.

    Mirrors :func:`from_storage` but skips the float32 inflation. Only valid
    for ``packed_uint8`` (np.unpackbits) and ``uint8`` (already-packed)
    storage layouts; raises :class:`FeatureExtractionError` otherwise.
    """
    if storage_dtype == STORAGE_UINT8:
        return stored if stored.dtype == np.uint8 else stored.astype(np.uint8, copy=False)
    if storage_dtype == STORAGE_PACKED:
        n = stored.shape[0]
        out = np.empty((n, bit_count), dtype=np.uint8)
        if n == 0:
            return out
        chunk = max(1, max_chunk_rows)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            out[start:end] = np.unpackbits(
                stored[start:end], axis=1, count=bit_count, bitorder='big'
            )
        return out
    raise FeatureExtractionError(
        f"from_storage_uint8() does not decode {storage_dtype!r}; "
        f"caller must use the float32 path for non-binary storage. "
        f"This call indicates a bug in the cache read dispatcher."
    )


def cache_features(default_cache_dir: Path) -> Callable[..., Any]:
    """Decorator factory: v2 bit-packed HDF5 cache around a featurizer call.

    The wrapped function MUST accept ``(smiles_list, featurizer, *args, **kwargs)``
    and return a ``(N, dim) float32`` matrix for the missed SMILES.
    """

    def decorator(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        @wraps(func)
        def wrapper(
            smiles_list: list[str], featurizer: Featurizer, *args: Any, **kwargs: Any
        ) -> np.ndarray:
            cache_dir = _resolve_cache_dir(default_cache_dir, kwargs)
            preferred_dtype = kwargs.pop('preferred_dtype', 'float32')
            dim = int(featurizer.get_dimension())

            featurizer_name = featurizer.get_name()
            storage_dtype = _normalize_storage_dtype(featurizer.get_storage_dtype())
            use_uint8_output = _resolve_uint8_output(
                preferred_dtype, storage_dtype, featurizer_name
            )
            out_np_dtype = np.uint8 if use_uint8_output else np.float32
            output_decode_dtype = 'uint8' if use_uint8_output else 'float32'

            if len(smiles_list) == 0:
                return np.empty((0, dim), dtype=out_np_dtype)

            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"features_{featurizer_name}.h5"

            query_hashes = _cache_keys_uint64(smiles_list, featurizer)

            # Phase 1: read-only attempt under LOCK_SH (lets concurrent readers run).
            found = np.zeros(len(smiles_list), dtype=bool)
            hit_features = np.empty((0, dim), dtype=out_np_dtype)
            t_lock = time.perf_counter()
            t_open = t_lock
            t_read = t_lock
            if cache_path.exists():
                with _acquire_lock(cache_path, mode='shared'):
                    t_lock = time.perf_counter()
                    try:
                        f_ro = _h5_open(cache_path, 'r')
                    except OSError:
                        f_ro = None
                    if f_ro is not None:
                        try:
                            sv = f_ro.attrs.get(ATTR_SCHEMA_VERSION, None)
                            bc = f_ro.attrs.get(ATTR_BIT_COUNT, None)
                            sd = f_ro.attrs.get(ATTR_STORAGE_DTYPE, None)
                            sd_match = sd is not None and str(sd) == storage_dtype
                            if (
                                sv is not None
                                and int(sv) == SCHEMA_VERSION
                                and bc == dim
                                and sd_match
                            ):
                                _validate_v2_file_integrity(f_ro, featurizer, cache_path)
                                t_open = time.perf_counter()

                                def _lookup_and_read() -> tuple[np.ndarray, np.ndarray]:
                                    found_local, target_rows_local = _lookup_cache(
                                        f_ro, query_hashes, cache_path
                                    )
                                    hit_features_local = _read_cache_hits(
                                        f_ro,
                                        target_rows_local,
                                        storage_dtype,
                                        dim,
                                        output_dtype=output_decode_dtype,
                                    )
                                    return found_local, hit_features_local

                                found, hit_features = _read_with_one_retry(_lookup_and_read)
                                t_read = time.perf_counter()
                        finally:
                            f_ro.close()

            miss_idx = np.where(~found)[0]
            n_hits = int(found.sum())
            n_misses = int(miss_idx.size)

            out = np.empty((len(smiles_list), dim), dtype=out_np_dtype)
            if n_hits:
                hit_idx = np.where(found)[0]
                out[hit_idx] = hit_features

            t_write_start = t_read
            t_end = t_read
            if n_misses:
                # Phase 2: misses → LOCK_EX, re-open (handles v1/dim rename),
                # re-lookup-and-read (picks up any rows committed by other
                # writers between Phase 1 release and Phase 2 acquire), then write.
                with _acquire_lock(cache_path, mode='exclusive'):
                    f = _open_or_create_h5(cache_path, featurizer)
                    try:
                        _validate_v2_file_integrity(f, featurizer, cache_path)

                        found, target_rows = _lookup_cache(f, query_hashes, cache_path)
                        hit_features = _read_cache_hits(
                            f, target_rows, storage_dtype, dim,
                            output_dtype=output_decode_dtype,
                        )
                        if found.any():
                            out[np.where(found)[0]] = hit_features
                        miss_idx = np.where(~found)[0]
                        n_hits = int(found.sum())
                        n_misses = int(miss_idx.size)

                        t_write_start = time.perf_counter()
                        if n_misses:
                            miss_smiles = [smiles_list[i] for i in miss_idx]
                            miss_features = func(miss_smiles, featurizer, *args, **kwargs)
                            miss_features = np.asarray(miss_features, dtype=np.float32)
                            if miss_features.shape != (n_misses, dim):
                                raise FeatureExtractionError(
                                    f"Featurizer returned shape {miss_features.shape}, "
                                    f"expected ({n_misses}, {dim})."
                                )
                            if use_uint8_output:
                                # Cache write still records float32 → packed/uint8 via
                                # to_storage; user-facing `out` keeps the compact dtype.
                                out[miss_idx] = miss_features.astype(np.uint8, copy=False)
                            else:
                                out[miss_idx] = miss_features
                            miss_hashes = query_hashes[miss_idx]
                            # Dedupe within the miss batch — same SMILES appearing
                            # twice produces identical hashes; the strict-ascending
                            # invariant rejects duplicate row writes.
                            _, unique_pos = np.unique(miss_hashes, return_index=True)
                            unique_pos.sort()
                            uniq_smiles = [miss_smiles[i] for i in unique_pos]
                            uniq_features = miss_features[unique_pos]
                            uniq_hashes = miss_hashes[unique_pos]
                            _write_cache_misses(
                                f,
                                uniq_smiles,
                                uniq_features,
                                uniq_hashes,
                                storage_dtype,
                            )
                        t_end = time.perf_counter()
                    finally:
                        f.close()

            logger.debug(
                "cache featurizer=%s hits=%d misses=%d "
                "open_ms=%.1f read_ms=%.1f write_ms=%.1f",
                featurizer_name,
                n_hits,
                n_misses,
                (t_open - t_lock) * 1000.0,
                (t_read - t_open) * 1000.0,
                (t_end - t_write_start) * 1000.0,
            )
            return out

        return wrapper

    return decorator

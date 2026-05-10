"""Synthetic-injection collision-resolution test (T026, SC-001).

Per quickstart.md Scenario 1, finding 100 SMILES pairs with naturally-colliding
64-bit truncated digests is computationally infeasible at test time (~5e9
candidates by birthday math). Instead we:

  1. Build a small v2-style cache with ``schema_version=2`` and a 64-bit
     ``hash_index`` dataset (manually written via h5py).
  2. Manually corrupt 100 entries in ``/hash_index`` with a duplicated value
     (the failure mode that emerges from real 64-bit collisions at 100M scale).
  3. Open the file with ``_open_or_create_h5(...)`` to trigger the v2->v3
     migration: the corrupt file is renamed to ``*.h5.hash64.bak`` and a fresh
     v3 cache (``schema_version=3``, ``S16`` ``hash_index``) is created.
  4. Recompute keys via ``_cache_keys_bytes16`` and assert all SMILES now have
     distinct 128-bit digests (i.e. BLAKE2b-128 disambiguates them).

Also pins SC-001's documented collision probability to the module docstring so
future edits cannot silently delete it.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.features import MorganFeaturizer
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
    _cache_keys_bytes16,
    _open_or_create_h5,
)


def _make_synthetic_smiles(n: int) -> list[str]:
    """Generate ``n`` distinct SMILES-like strings.

    Cheap, deterministic, and sufficient for hash-collision tests since the
    cache key digests bytes (not parsed molecules).
    """
    return [f'C{i}CCO' for i in range(n)]


def _write_legacy_v2_64bit_cache(
    cache_path: Path, featurizer: MorganFeaturizer, smiles_list: list[str]
) -> None:
    """Write a synthetic v2 cache file with a 64-bit ``uint64`` hash_index.

    Mimics the on-disk shape produced by the pre-feature-020 code path:
    ``schema_version=2`` and ``/hash_index`` dtype ``uint64`` (NOT ``S16``).
    """
    bit_count = int(featurizer.get_dimension())
    n = len(smiles_list)
    packed_width = (bit_count + 7) // 8

    # Synthetic uint64 hashes: distinct, sorted ascending so the file passes
    # v2's strictly-increasing invariant before we corrupt it.
    hash_index_u64 = np.arange(1, n + 1, dtype=np.uint64)
    row_index = np.arange(n, dtype=np.uint64)
    features = np.zeros((n, packed_width), dtype=np.uint8)

    with h5py.File(cache_path, 'w') as f:
        f.attrs[ATTR_SCHEMA_VERSION] = np.uint8(2)
        f.attrs[ATTR_BIT_COUNT] = np.uint32(bit_count)
        f.attrs[ATTR_STORAGE_DTYPE] = STORAGE_PACKED
        f.attrs[ATTR_FEATURIZER_NAME] = featurizer.get_name()
        f.attrs[ATTR_WRITE_EPOCH] = np.uint64(1)
        f.attrs[ATTR_STORAGE_LAYOUT] = LAYOUT_DENSE
        f.create_dataset(
            DSET_FEATURES,
            data=features,
            maxshape=(None, packed_width),
            dtype=np.uint8,
        )
        f.create_dataset(
            DSET_HASH_INDEX,
            data=hash_index_u64,
            maxshape=(None,),
            dtype=np.uint64,
        )
        f.create_dataset(
            DSET_ROW_INDEX,
            data=row_index,
            maxshape=(None,),
            dtype=np.uint64,
        )


@pytest.fixture
def morgan() -> MorganFeaturizer:
    return MorganFeaturizer(n_jobs=1)


@pytest.mark.unit
def test_synthetic_64bit_collision_resolution(
    tmp_path: Path, morgan: MorganFeaturizer
) -> None:
    """Synthetic v2 cache + 100-way duplicate -> migration -> 128-bit disambiguation."""
    smiles_list = _make_synthetic_smiles(n=100)
    cache_path = tmp_path / f'features_{morgan.get_name()}.h5'

    # 1. Build a small v2 cache.
    _write_legacy_v2_64bit_cache(cache_path, morgan, smiles_list)

    # 2. Corrupt 100 entries by overwriting them with a single duplicated value
    #    (the on-disk pattern that real 64-bit collisions would produce at scale).
    with h5py.File(cache_path, 'r+') as f:
        f[DSET_HASH_INDEX][:100] = f[DSET_HASH_INDEX][0]

    # 3. Open with the new v3 code -> triggers v2->v3 migration.
    f = _open_or_create_h5(cache_path, morgan)
    try:
        assert int(f.attrs[ATTR_SCHEMA_VERSION]) == SCHEMA_VERSION
        assert int(f.attrs[ATTR_SCHEMA_VERSION]) == 3
        assert f[DSET_HASH_INDEX].dtype == HASH_DTYPE
    finally:
        f.close()

    # The corrupt v2 file MUST have been preserved as the .hash64.bak sidecar.
    backup = tmp_path / f'features_{morgan.get_name()}.h5.hash64.bak'
    assert backup.exists(), (
        f"Expected v2 cache to be renamed to {backup}; tmp_path contents: "
        f"{sorted(p.name for p in tmp_path.iterdir())}"
    )

    # 4. Recompute the 128-bit keys for the original SMILES list and assert
    #    every digest is unique -- BLAKE2b-128 must resolve all 100 "collisions".
    keys = _cache_keys_bytes16(smiles_list, morgan)
    assert len(set(keys.tolist())) == len(smiles_list), (
        'BLAKE2b-128 must produce 100 distinct digests for 100 distinct SMILES'
    )


@pytest.mark.unit
def test_blake2b_collision_probability_documented() -> None:
    """SC-001 expectation is pinned in the cache module docstring.

    A trivial doc-test that future edits cannot silently delete the
    collision-probability statement that justifies the 128-bit widening.
    """
    from learnm8.features import cache as cache_mod

    docstring = cache_mod.__doc__ or ''
    assert 'blake2b' in docstring.lower(), (
        'cache.py module docstring must name the BLAKE2b hash recipe'
    )
    assert '1.5e-23' in docstring or '10^8' in docstring or 'N=10^8' in docstring, (
        "cache.py module docstring must cite the collision probability "
        "(expected '1.5e-23' or 'N=10^8' / '10^8')"
    )

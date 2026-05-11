"""Migration test for legacy packed_uint8 cache files written under count=True (T003).

Pre-existing v2 caches written by the *broken* code path used storage_dtype=
'packed_uint8' for ``Morgan(count=True)``. After this fix the same featurizer
declares ``csr_uint16`` storage. ``_open_or_create_h5`` MUST detect the
mismatch and rename the legacy file to ``<...>.h5.dim*.bak`` rather than
silently re-using corrupt data.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.features import create_featurizer
from learnm8.features.cache import (
    DSET_FEATURES,
    DSET_HASH_INDEX,
    DSET_ROW_INDEX,
    STORAGE_PACKED,
    _open_or_create_h5,
)


def _write_legacy_packed_cache(path: Path, featurizer, *, n_rows: int = 4) -> None:
    bit_count = int(featurizer.get_dimension())
    packed_width = (bit_count + 7) // 8
    with h5py.File(path, 'w') as f:
        f.attrs['schema_version'] = np.uint8(2)
        f.attrs['bit_count'] = np.uint32(bit_count)
        f.attrs['storage_dtype'] = STORAGE_PACKED
        f.attrs['featurizer_name'] = featurizer.get_name()
        f.attrs['write_epoch'] = np.uint64(1)
        f.attrs['storage_layout'] = 'dense'
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


@pytest.mark.unit
def test_legacy_packed_cache_renamed_to_bak_on_open(tmp_path: Path):
    # Reproduce a corrupted legacy cache file: written with packed_uint8 by the
    # old code path that didn't honour count=True.
    feat = create_featurizer('morgan', count=True)
    cache_path = tmp_path / f'features_{feat.get_name()}.h5'
    _write_legacy_packed_cache(cache_path, feat)

    # New code recognises this featurizer's storage_dtype as 'csr_uint16'.
    assert feat.get_storage_dtype() == 'csr_uint16'

    f = _open_or_create_h5(cache_path, feat)
    try:
        # The fresh file must record the new storage_dtype and lack the legacy
        # /features dataset (CSR layout uses csr_data/csr_indices/csr_indptr).
        assert str(f.attrs['storage_dtype']) == 'csr_uint16'
        assert (
            f.attrs.get('storage_layout', b'').decode()
            if isinstance(f.attrs['storage_layout'], bytes)
            else str(f.attrs['storage_layout']) in ('csr',)
        )
    finally:
        f.close()

    # Backup file must exist with .bak suffix.
    backups = list(tmp_path.glob('features_*.h5.*.bak'))
    assert backups, (
        f'Expected legacy cache to be renamed to *.bak, found nothing in '
        f'{tmp_path}: {list(tmp_path.iterdir())}'
    )

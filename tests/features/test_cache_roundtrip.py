"""Round-trip tests for v3 cache: cold→write→read→unpack matches cold output (T017, SC-006)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from learnm8.features import create_featurizer
from learnm8.features.cache import HASH_DTYPE
from learnm8.features.extraction import extract_features


@pytest.mark.unit
def test_morgan_packed_uint8_roundtrip(tmp_path: Path):
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    cold = extract_features(['CCO', 'CCC', 'CCN'], feat, cache_dir=tmp_path)

    assert cold.shape == (3, 2048)
    assert cold.dtype == np.float32
    assert set(np.unique(cold).tolist()) <= {0.0, 1.0}

    cache_file = tmp_path / 'features_morgan_packed_uint8.h5'
    with h5py.File(cache_file, 'r') as f:
        assert int(f.attrs['schema_version']) == 3
        assert str(f.attrs['storage_dtype']) == 'packed_uint8'
        assert int(f.attrs['bit_count']) == 2048
        assert f['features'].shape == (3, 256)
        assert f['features'].dtype == np.uint8
        assert f['hash_index'].shape == (3,)
        assert f['hash_index'].dtype == HASH_DTYPE
        assert f['row_index'].shape == (3,)
        assert f['row_index'].dtype == np.uint64
        # S16 element-wise comparison is lex byte comparison; np.diff doesn't apply.
        h_arr = f['hash_index'][:]
        assert np.all(h_arr[1:] > h_arr[:-1])

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

    cache_file = tmp_path / 'features_maccs_packed_uint8.h5'
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

    cache_file = tmp_path / 'features_mordred_float32.h5'
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


@pytest.mark.unit
def test_equivalent_smiles_dedup_in_cache(tmp_path: Path):
    """Equivalent SMILES share one cache row when routed through validation (Item 5).

    After feature 025 Item 5, raw extract_features() does NOT canonicalize SMILES
    internally — so calling it with 'CCO' then 'OCC' directly would create 2 rows.
    Equivalence collapses to a single cache entry only when SMILES are first
    canonicalized via validate_compound_pool (or canonicalize_for_cache directly).
    """
    import polars as pl

    from learnm8.core.validation import validate_compound_pool

    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)

    df_cco = pl.DataFrame({'ID': ['C1'], 'SMILES': ['CCO']})
    df_occ = pl.DataFrame({'ID': ['C2'], 'SMILES': ['OCC']})

    result_cco = validate_compound_pool(df_cco, n_jobs=1, progress=False)
    result_occ = validate_compound_pool(df_occ, n_jobs=1, progress=False)

    canon_cco = result_cco.valid_compounds['SMILES'].to_list()
    canon_occ = result_occ.valid_compounds['SMILES'].to_list()

    # Both must canonicalize to the same SMILES string.
    assert canon_cco == canon_occ, (
        f'Expected CCO and OCC to share canonical form, '
        f'got {canon_cco!r} and {canon_occ!r}'
    )

    cold = extract_features(canon_cco, feat, cache_dir=tmp_path)

    cache_file = tmp_path / 'features_morgan_packed_uint8.h5'
    with h5py.File(cache_file, 'r') as f:
        assert f['hash_index'].shape == (1,)

    # canon_occ == canon_cco, so this is a 100% cache hit — no new row.
    warm = extract_features(canon_occ, feat, cache_dir=tmp_path)
    assert np.array_equal(cold, warm)
    with h5py.File(cache_file, 'r') as f:
        assert f['hash_index'].shape == (1,), (
            'Canonical OCC must reuse the canonical CCO cache entry'
        )

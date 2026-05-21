"""Tests for feature 025 Item 5: canonicalize-once-at-validation.

Covers:
- validate_compound_pool writes RDKit-canonical SMILES into valid_compounds
- canonicalize_for_cache is byte-identical to the legacy _canonicalize_smiles
  recipe (REQ-14b golden-key test)
- raw extract_features() produces 2 hash_index rows for non-canonical equivalent
  SMILES (the accepted edge case: no in-cache dedup without prior canonicalization)
- Feature values are unchanged by canonicalization (molecule identity preserved)
- skip_validation canonicalization pass keys identically to the normal path
  (zero new hash_index rows on second run)
- Multi-featurizer sanity: 'morgan' and 'maccs' both succeed on canonical SMILES
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pytest
from rdkit import Chem

from learnm8.core.validation import canonicalize_for_cache, validate_compound_pool
from learnm8.features import create_featurizer
from learnm8.features.cache import _cache_keys_bytes16
from learnm8.features.extraction import extract_features

# ---------------------------------------------------------------------------
# Test 1: validate_compound_pool writes canonical SMILES
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_compound_pool_writes_canonical_smiles(tmp_path: Path) -> None:
    """validate_compound_pool replaces raw SMILES with RDKit-canonical forms."""
    raw_smiles = ['OCC', 'C1=CC=CC=C1', 'N#N', 'CC(O)=O']
    df = pl.DataFrame({
        'ID': [f'C{i}' for i in range(len(raw_smiles))],
        'SMILES': raw_smiles,
    })

    result = validate_compound_pool(df, n_jobs=1, progress=False)

    assert len(result.valid_compounds) == len(raw_smiles)
    returned_smiles = result.valid_compounds['SMILES'].to_list()

    for raw, canon in zip(raw_smiles, returned_smiles, strict=True):
        expected = canonicalize_for_cache(raw)
        assert canon == expected, (
            f'SMILES {raw!r}: expected canonical {expected!r}, got {canon!r}'
        )


@pytest.mark.unit
def test_invalid_compounds_keep_raw_smiles() -> None:
    """invalid_compounds in ValidationResult keep the raw (non-canonical) SMILES."""
    df = pl.DataFrame({
        'ID': ['good', 'bad'],
        'SMILES': ['OCC', 'NOT_A_SMILES!!!'],
    })
    result = validate_compound_pool(df, n_jobs=1, progress=False)

    assert len(result.valid_compounds) == 1
    assert len(result.invalid_compounds) == 1
    assert result.invalid_compounds['SMILES'][0] == 'NOT_A_SMILES!!!'


# ---------------------------------------------------------------------------
# Test 2: Golden-key — canonicalize_for_cache == legacy recipe (REQ-14b)
# ---------------------------------------------------------------------------


def _legacy_canonicalize(smiles: str) -> str:
    """Inline reproduction of the deleted _canonicalize_smiles recipe."""
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else smiles


GOLDEN_SMILES: list[str] = [
    'CCO',
    'OCC',
    'c1ccccc1',
    'C1=CC=CC=C1',
    'CC(=O)O',
    'NC(=O)c1cnccn1',
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'not_a_valid_smiles!!!',
]


@pytest.mark.unit
def test_canonicalize_for_cache_matches_legacy_recipe() -> None:
    """canonicalize_for_cache is byte-identical to the deleted _canonicalize_smiles."""
    for s in GOLDEN_SMILES:
        got = canonicalize_for_cache(s)
        expected = _legacy_canonicalize(s)
        assert got == expected, (
            f'canonicalize_for_cache({s!r}) = {got!r} != legacy {expected!r}'
        )


@pytest.mark.unit
def test_cache_keys_identical_legacy_vs_new(tmp_path: Path) -> None:
    """_cache_keys_bytes16 on canonical SMILES is byte-equal to the legacy-canonical keys.

    REQ-14b: Item 5 triggers no cold rebuild — keys are byte-identical because
    canonicalize_for_cache reproduces the deleted _canonicalize_smiles recipe.
    """
    feat = create_featurizer('morgan', n_jobs=1)

    legacy_canonical = [_legacy_canonicalize(s) for s in GOLDEN_SMILES]
    new_canonical = [canonicalize_for_cache(s) for s in GOLDEN_SMILES]

    # Canonical strings must be identical first.
    assert legacy_canonical == new_canonical, (
        'canonicalize_for_cache diverges from legacy recipe on at least one SMILES'
    )

    legacy_keys = _cache_keys_bytes16(legacy_canonical, feat)
    new_keys = _cache_keys_bytes16(new_canonical, feat)

    np.testing.assert_array_equal(
        legacy_keys,
        new_keys,
        err_msg='Cache keys differ between legacy and new canonicalization path',
    )
    # Byte-level assertion as well.
    assert legacy_keys.tobytes() == new_keys.tobytes()


# ---------------------------------------------------------------------------
# Test 3: Raw extract_features produces 2 rows (REQ-14b accepted edge case)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_raw_extract_features_no_dedup(tmp_path: Path) -> None:
    """Raw (non-canonical) equivalent SMILES create 2 separate hash_index rows.

    After Item 5 there is no in-cache canonicalization.  Callers must canonicalize
    before extract_features; extract_features with verbatim non-equivalent strings
    produces one row each.
    """
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)

    extract_features(['OCC'], feat, cache_dir=tmp_path)
    extract_features(['CCO'], feat, cache_dir=tmp_path)

    cache_file = tmp_path / 'features_morgan_packed_uint8.h5'
    with h5py.File(cache_file, 'r') as f:
        n_rows = f['hash_index'].shape[0]

    # 'OCC' != 'CCO' as verbatim strings => two distinct keys => 2 rows.
    assert n_rows == 2, (
        f'Expected 2 hash_index rows for non-canonical raw input, got {n_rows}'
    )


# ---------------------------------------------------------------------------
# Test 4: Feature values unchanged by canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_feature_values_unchanged_by_canonicalization(tmp_path: Path) -> None:
    """Canonical and raw equivalent SMILES produce byte-identical feature vectors."""
    feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)

    canonical = canonicalize_for_cache('OCC')
    # Verify these two map to the same canonical form (i.e., CCO == canonical).
    assert canonicalize_for_cache('CCO') == canonical

    feats_canonical = extract_features([canonical], feat, cache_dir=tmp_path / 'a')
    feats_raw_equiv = extract_features(['CCO'], feat, cache_dir=tmp_path / 'b')

    np.testing.assert_array_equal(
        feats_canonical,
        feats_raw_equiv,
        err_msg='Feature vectors differ between canonical and raw equivalent SMILES',
    )


# ---------------------------------------------------------------------------
# Test 5: skip_validation reuses validated cache rows (zero new rows)
# ---------------------------------------------------------------------------


def _make_tiny_benchmark_csv(path: Path, n: int = 30) -> Path:
    """Write a tiny benchmark CSV with valid SMILES and numeric Activity."""
    smiles_pool = [
        'CCO', 'CCC', 'CCN', 'c1ccccc1', 'CC(=O)O', 'CC(C)O',
        'CCCC', 'CCCN', 'CCCl', 'CCBr',
        'c1ccncc1', 'c1ccoc1', 'c1ccsc1', 'CC#N', 'CCC=O',
        'CC(=O)N', 'CC(C)=O', 'CCS', 'CCSC', 'CCOC',
        'CCF', 'CCCI', 'C1CCCC1', 'C1CCCCC1', 'C1CCNCC1',
        'C1CCOCC1', 'CC(O)CC', 'CC(N)CC', 'CC(=O)CC', 'CCCO',
    ]
    rows = smiles_pool[:n]
    csv_path = path / 'benchmark.csv'
    lines = ['ID,SMILES,Activity']
    for i, s in enumerate(rows):
        lines.append(f'COMP_{i:03d},{s},{(i + 1) * 0.1:.2f}')
    csv_path.write_text('\n'.join(lines))
    return csv_path


@pytest.mark.integration
@pytest.mark.molecular
def test_skip_validation_reuses_cache_rows(tmp_path: Path) -> None:
    """skip_validation canonicalization pass keys identically to the normal path.

    Two sequential runs on the same raw pool / same cache_dir must share ALL
    feature rows — the second run adds zero new hash_index entries.
    """
    from learnm8 import run_active_learning

    csv_path = _make_tiny_benchmark_csv(tmp_path)
    cache_dir = tmp_path / 'shared_cache'

    # First run: normal validation (canonicalizes via validate_compound_pool).
    run_active_learning(
        compound_pool=str(csv_path),
        oracle=str(csv_path),
        target_col='Activity',
        learner='rf',
        featurizer='morgan',
        n_cycles=2,
        batch_fraction=0.1,
        cache_dir=cache_dir,
        output_dir=tmp_path / 'run1',
        random_state=42,
        n_jobs=1,
    )

    h5_path = cache_dir / 'features_morgan_packed_uint8.h5'
    assert h5_path.exists(), f'Cache file not found at {h5_path}'

    with h5py.File(h5_path, 'r') as f:
        rows_after_run1 = int(f['hash_index'].shape[0])

    assert rows_after_run1 > 0, 'Run 1 must have populated the cache'

    # Second run: skip_validation on the same raw CSV / same cache_dir.
    run_active_learning(
        compound_pool=str(csv_path),
        oracle=str(csv_path),
        target_col='Activity',
        learner='rf',
        featurizer='morgan',
        n_cycles=2,
        batch_fraction=0.1,
        cache_dir=cache_dir,
        output_dir=tmp_path / 'run2',
        random_state=42,
        skip_validation=True,
        n_jobs=1,
    )

    with h5py.File(h5_path, 'r') as f:
        rows_after_run2 = int(f['hash_index'].shape[0])

    assert rows_after_run2 == rows_after_run1, (
        f'skip_validation added {rows_after_run2 - rows_after_run1} new cache rows; '
        f'expected 0 (keys must be identical between normal and skip_validation paths)'
    )


# ---------------------------------------------------------------------------
# Test 6: Multi-featurizer sanity run
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multi_featurizer_canonical_smiles(tmp_path: Path) -> None:
    """Morgan and MACCS both succeed on canonical SMILES and return correct shapes."""
    raw_smiles = ['OCC', 'c1ccccc1', 'CC(=O)O', 'NC(=O)c1cnccn1']
    canonical = [canonicalize_for_cache(s) for s in raw_smiles]

    morgan_feat = create_featurizer('morgan', radius=2, fp_size=2048, n_jobs=1)
    maccs_feat = create_featurizer('maccs', n_jobs=1)

    morgan_feats = extract_features(canonical, morgan_feat, cache_dir=tmp_path / 'morgan')
    maccs_feats = extract_features(canonical, maccs_feat, cache_dir=tmp_path / 'maccs')

    assert morgan_feats.shape == (len(canonical), 2048)
    assert morgan_feats.dtype == np.float32

    assert maccs_feats.shape == (len(canonical), maccs_feat.get_dimension())
    assert maccs_feats.dtype == np.float32

    # Features must be binary (0.0 or 1.0) for both fingerprint types.
    assert set(np.unique(morgan_feats).tolist()) <= {0.0, 1.0}
    assert set(np.unique(maccs_feats).tolist()) <= {0.0, 1.0}

    # Verify via re-extraction (warm reads match cold).
    morgan_warm = extract_features(canonical, morgan_feat, cache_dir=tmp_path / 'morgan')
    maccs_warm = extract_features(canonical, maccs_feat, cache_dir=tmp_path / 'maccs')
    np.testing.assert_array_equal(morgan_feats, morgan_warm)
    np.testing.assert_array_equal(maccs_feats, maccs_warm)

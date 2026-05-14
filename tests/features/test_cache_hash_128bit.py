"""Unit tests for ``_cache_keys_bytes16`` (T025, FR-001, FR-002).

Verifies the 128-bit BLAKE2b cache-key API:
  - dtype is ``|S16`` (16 raw bytes per row)
  - shape is 1-D ``(N,)``
  - deterministic across calls
  - distinct SMILES produce distinct digests
  - featurizer config differences propagate into the digest
  - empty input returns an empty ``S16`` array
  - the suffix-hoisting micro-optimisation matches the naive concat recipe
"""

from __future__ import annotations

import hashlib

import numpy as np
import numpy.testing as npt
import pytest

from learnm8.features import SkfpFeaturizer, create_featurizer
from learnm8.features.cache import HASH_DTYPE, _cache_keys_bytes16


def _make_morgan(**kwargs) -> SkfpFeaturizer:
    return create_featurizer('morgan', n_jobs=1, **kwargs)


SAMPLE_SMILES: list[str] = [
    'CCO',
    'c1ccccc1',
    'CC(=O)O',
    'NC(=O)c1cnccn1',
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
]


@pytest.fixture
def morgan() -> SkfpFeaturizer:
    """A default-config Morgan featurizer (binary, 2048-bit, radius=2)."""
    return _make_morgan()


@pytest.mark.unit
def test_output_dtype_is_s16(morgan: SkfpFeaturizer) -> None:
    arr = _cache_keys_bytes16(SAMPLE_SMILES, morgan)
    assert arr.dtype == HASH_DTYPE
    assert arr.dtype == np.dtype('S16')
    # Each scalar element is a 16-byte bytes object.
    assert all(isinstance(item, bytes) and len(item) == 16 for item in arr.tolist())


@pytest.mark.unit
def test_output_shape_is_one_dimensional(morgan: SkfpFeaturizer) -> None:
    arr = _cache_keys_bytes16(SAMPLE_SMILES, morgan)
    assert arr.shape == (len(SAMPLE_SMILES),)
    assert arr.ndim == 1


@pytest.mark.unit
def test_determinism_byte_identical_across_calls(morgan: SkfpFeaturizer) -> None:
    first = _cache_keys_bytes16(SAMPLE_SMILES, morgan)
    second = _cache_keys_bytes16(SAMPLE_SMILES, morgan)
    npt.assert_array_equal(first, second)
    # Byte-level equality of the raw buffers as a stronger check.
    assert first.tobytes() == second.tobytes()


@pytest.mark.unit
def test_distinct_smiles_produce_distinct_hashes(morgan: SkfpFeaturizer) -> None:
    # Generate 1000 distinct SMILES-like strings; we don't need them to be
    # chemically valid since _cache_keys_bytes16 only digests the bytes.
    distinct_smiles: list[str] = [f'C{i}CCO' for i in range(1000)]
    arr = _cache_keys_bytes16(distinct_smiles, morgan)
    assert len(set(arr.tolist())) == 1000


@pytest.mark.unit
def test_same_smiles_different_featurizer_config_differs() -> None:
    binary_feat = _make_morgan()
    count_feat = _make_morgan(count=True)

    binary_keys = _cache_keys_bytes16(SAMPLE_SMILES, binary_feat)
    count_keys = _cache_keys_bytes16(SAMPLE_SMILES, count_feat)

    # Featurizer-name buckets may overlap (both 'morgan'), but the config_hash
    # MUST diverge, so every row must differ between the two arrays.
    for binary_digest, count_digest in zip(
        binary_keys.tolist(), count_keys.tolist(), strict=True
    ):
        assert binary_digest != count_digest


@pytest.mark.unit
def test_empty_input_returns_empty_s16_array(morgan: SkfpFeaturizer) -> None:
    arr = _cache_keys_bytes16([], morgan)
    assert arr.shape == (0,)
    assert arr.dtype == HASH_DTYPE


@pytest.mark.unit
def test_suffix_hoisting_matches_naive_concat(morgan: SkfpFeaturizer) -> None:
    """Verify the digest equals the naive recipe on canonicalised SMILES.

    SMILES are canonicalised before hashing, so the expected digest is computed
    over ``Chem.MolToSmiles(Chem.MolFromSmiles(s))`` for each sample.
    """
    from rdkit import Chem

    name = morgan.get_name()
    config_hash = morgan.get_config_hash()

    optimised = _cache_keys_bytes16(SAMPLE_SMILES, morgan)
    expected = np.array(
        [
            hashlib.blake2b(
                f'{Chem.MolToSmiles(Chem.MolFromSmiles(s))}_{name}_{config_hash}'.encode(),
                digest_size=16,
                usedforsecurity=False,
            ).digest()
            for s in SAMPLE_SMILES
        ],
        dtype=HASH_DTYPE,
    )
    npt.assert_array_equal(optimised, expected)


@pytest.mark.unit
def test_equivalent_smiles_share_cache_key(morgan: SkfpFeaturizer) -> None:
    """Equivalent SMILES representations canonicalise to the same digest."""
    # "CCO" / "OCC" (ethanol) and two benzene spellings.
    keys = _cache_keys_bytes16(['CCO', 'OCC', 'c1ccccc1', 'C1=CC=CC=C1'], morgan)
    assert keys[0] == keys[1]
    assert keys[2] == keys[3]
    assert keys[0] != keys[2]


@pytest.mark.unit
def test_invalid_smiles_uses_raw_string_fallback(morgan: SkfpFeaturizer) -> None:
    """SMILES RDKit cannot parse still get a stable, deterministic key."""
    bad = 'not_a_valid_smiles!!!'
    name = morgan.get_name()
    config_hash = morgan.get_config_hash()
    keys = _cache_keys_bytes16([bad], morgan)
    expected = hashlib.blake2b(
        f'{bad}_{name}_{config_hash}'.encode(),
        digest_size=16,
        usedforsecurity=False,
    ).digest()
    assert keys[0] == expected
    # Deterministic across calls.
    npt.assert_array_equal(keys, _cache_keys_bytes16([bad], morgan))

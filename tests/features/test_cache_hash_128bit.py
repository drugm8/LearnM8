"""Unit tests for ``_cache_keys_bytes16`` (T025, FR-001, FR-002).

Verifies the 128-bit BLAKE2b cache-key API:
  - dtype is ``|S16`` (16 raw bytes per row)
  - shape is 1-D ``(N,)``
  - deterministic across calls
  - distinct SMILES produce distinct digests
  - featurizer config differences propagate into the digest
  - empty input returns an empty ``S16`` array
  - the suffix-hoisting micro-optimisation matches the naive concat recipe

Feature 026 additions:
  - REQ-8: digests are byte-identical to the pre-026 recipe
    ``f'{s}_{name}_{config_hash}'.encode()``, pinned against hardcoded
    reference digests (see :data:`LEGACY_REFERENCE_DIGESTS`)
  - REQ-7: the serial and loky-parallel paths return identical arrays,
    including across slice boundaries
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import cast

import numpy as np
import numpy.testing as npt
import pytest

from learnm8.core.interfaces import Featurizer
from learnm8.features import SkfpFeaturizer, create_featurizer
from learnm8.features import cache as cache_module
from learnm8.features.cache import (
    HASH_DTYPE,
    _cache_keys_bytes16,
    _hash_smiles_slice,
)


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
    """Verify the digest equals the naive recipe on the verbatim SMILES input.

    After feature 025 Item 5, _cache_keys_bytes16 keys SMILES verbatim — there
    is no internal canonicalization step.  The expected digest is therefore
    computed over the raw input string (not the RDKit-canonical form).
    """
    name = morgan.get_name()
    config_hash = morgan.get_config_hash()

    optimised = _cache_keys_bytes16(SAMPLE_SMILES, morgan)
    expected = np.array(
        [
            hashlib.blake2b(
                f'{s}_{name}_{config_hash}'.encode(),
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
    """Equivalent SMILES share a cache key after canonicalization via canonicalize_for_cache.

    After feature 025 Item 5, _cache_keys_bytes16 does NOT canonicalize SMILES
    internally.  Equivalence is established by first passing SMILES through
    canonicalize_for_cache (which replicates what validate_compound_pool does),
    then computing keys on the resulting canonical strings.
    """
    from learnm8.core.validation import canonicalize_for_cache

    raw = ['CCO', 'OCC', 'c1ccccc1', 'C1=CC=CC=C1']
    canonical = [canonicalize_for_cache(s) for s in raw]

    # 'CCO' and 'OCC' (ethanol) must map to the same canonical form.
    assert canonical[0] == canonical[1], (
        f'Expected CCO and OCC to canonicalize identically, '
        f'got {canonical[0]!r} and {canonical[1]!r}'
    )
    # Benzene kekulé and aromatic forms must also map identically.
    assert canonical[2] == canonical[3], (
        f'Expected c1ccccc1 and C1=CC=CC=C1 to canonicalize identically, '
        f'got {canonical[2]!r} and {canonical[3]!r}'
    )

    keys = _cache_keys_bytes16(canonical, morgan)
    assert keys[0] == keys[1], 'Canonical ethanol forms must share a cache key'
    assert keys[2] == keys[3], 'Canonical benzene forms must share a cache key'
    assert keys[0] != keys[2], 'Ethanol and benzene must have distinct cache keys'


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


# ---------------------------------------------------------------------------
# Feature 026: REQ-8 legacy-recipe pin
# ---------------------------------------------------------------------------


class _FrozenFeaturizer:
    """Minimal stand-in exposing only what ``_cache_keys_bytes16`` consumes.

    ``_cache_keys_bytes16`` reads exactly two things off the featurizer:
    :meth:`get_name` and :meth:`get_config_hash`. Freezing both to literals
    makes :data:`LEGACY_REFERENCE_DIGESTS` stable forever — a real featurizer's
    ``get_config_hash()`` would shift on any scikit-fingerprints release that
    adds a ``get_params()`` key, which would break this test for a reason that
    has nothing to do with REQ-8's byte recipe.
    """

    def __init__(self, name: str = 'morgan', config_hash: str = '0123456789abcdef'):
        self._name = name
        self._config_hash = config_hash

    def get_name(self) -> str:
        return self._name

    def get_config_hash(self) -> str:
        return self._config_hash


def _frozen(**kwargs) -> Featurizer:
    """Build a :class:`_FrozenFeaturizer` typed as the interface it duck-types."""
    return cast(Featurizer, _FrozenFeaturizer(**kwargs))


# Real ZINC SMILES from tests/data/diverse_molecules.csv (rows 1-6), digested
# with the PRE-026 recipe f'{s}_morgan_0123456789abcdef'.encode() under
# blake2b(digest_size=16). Any change to the key recipe breaks these — that is
# the point: a broken digest silently orphans every cache in existence.
LEGACY_REFERENCE_DIGESTS: list[tuple[str, str]] = [
    (
        'C=Cn1cc(CN2CCC(N3CCS(=O)(=O)CC3)CC2)cn1',
        'c64a6a6ec494eb4778d31d932cf02789',
    ),
    (
        'COc1ccc([C@H](C)NC(=O)Nc2cc(C(N)=O)ccc2F)cc1F',
        '46951d6965a8ab4700302fe2fa363711',
    ),
    (
        'CCO[C@@H]1C[C@H](N(C)C[C@@H]2CCO[C@@H](CC)C2)C12CCC2',
        '432b07823376254dc23dca8cb665eea1',
    ),
    (
        r'Cn1c2ccccc2s/c1=N\N=C\c1ccccn1',
        'b7ff5d4ed0ba7a7d3d8151dc3d66bbf5',
    ),
    (
        'Cc1nc2ccc(N3CCc4[nH]c5ccc(Cl)cc5c4C3)nn2n1',
        '07a364dac1621b9b5cac3206c6ad34bd',
    ),
    (
        'C[S@@](=O)C1(CNC(=O)C(=O)NCCN2CCOCC2(C)C)CC1',
        '573ada29e50167ffeebe62fa4d4bc9fc',
    ),
]


@pytest.mark.unit
def test_reference_smiles_are_present_in_the_real_fixture() -> None:
    """The REQ-8 pin uses real molecules, not invented strings (§3.4 red line 1)."""
    fixture = Path(__file__).parent.parent / 'data' / 'diverse_molecules.csv'
    with fixture.open() as handle:
        fixture_smiles = {row['SMILES'] for row in csv.DictReader(handle)}
    for smiles, _ in LEGACY_REFERENCE_DIGESTS:
        assert smiles in fixture_smiles, f'{smiles!r} not in {fixture}'


@pytest.mark.unit
def test_digests_match_hardcoded_legacy_recipe() -> None:
    """REQ-8: every digest equals the pre-026 f-string recipe, byte for byte."""
    featurizer = _frozen()
    smiles = [s for s, _ in LEGACY_REFERENCE_DIGESTS]
    expected = np.array(
        [bytes.fromhex(h) for _, h in LEGACY_REFERENCE_DIGESTS], dtype=HASH_DTYPE
    )

    npt.assert_array_equal(_cache_keys_bytes16(smiles, featurizer), expected)


@pytest.mark.unit
def test_digests_match_legacy_recipe_under_parallel_kwarg() -> None:
    """REQ-8 holds on the parallel path too, not just the serial default."""
    featurizer = _frozen()
    smiles = [s for s, _ in LEGACY_REFERENCE_DIGESTS]
    expected = np.array(
        [bytes.fromhex(h) for _, h in LEGACY_REFERENCE_DIGESTS], dtype=HASH_DTYPE
    )

    npt.assert_array_equal(_cache_keys_bytes16(smiles, featurizer, n_jobs=4), expected)


@pytest.mark.unit
def test_config_hash_is_folded_into_the_suffix_not_the_smiles() -> None:
    """A different config_hash must move every digest (suffix really is hashed)."""
    smiles = [s for s, _ in LEGACY_REFERENCE_DIGESTS]
    baseline = _cache_keys_bytes16(smiles, _frozen())
    shifted = _cache_keys_bytes16(smiles, _frozen(config_hash='fedcba9876543210'))

    assert not np.any(baseline == shifted)


# ---------------------------------------------------------------------------
# Feature 026: REQ-7 serial / parallel path equivalence
# ---------------------------------------------------------------------------


def _fixture_smiles() -> list[str]:
    """Real SMILES from the diverse_molecules fixture (§3.4: no synthetic data)."""
    fixture = Path(__file__).parent.parent / 'data' / 'diverse_molecules.csv'
    with fixture.open() as handle:
        return [row['SMILES'] for row in csv.DictReader(handle)]


def _repeat_to(smiles: list[str], n: int) -> list[str]:
    """Tile the fixture up to ``n`` rows, keeping every row distinct.

    The index prefix makes each entry unique so a slice-ordering bug shows up
    as a value mismatch rather than being masked by duplicate digests.
    """
    return [f'{smiles[i % len(smiles)]}.[{i}He]' for i in range(n)]


@pytest.fixture
def low_parallel_threshold(monkeypatch: pytest.MonkeyPatch) -> int:
    """Drop PARALLEL_HASH_MIN_ROWS so the parallel path trips on small input.

    The production value is 200_000; hashing that many rows in a unit test
    would dominate the fast suite. The threshold is only a dispatch policy,
    so lowering it exercises exactly the same code.
    """
    threshold = 64
    monkeypatch.setattr(cache_module, 'PARALLEL_HASH_MIN_ROWS', threshold)
    return threshold


@pytest.mark.unit
def test_serial_and_parallel_paths_agree(low_parallel_threshold: int) -> None:
    """REQ-7: both dispatch paths return byte-identical arrays."""
    featurizer = _frozen()
    # 250 rows over 4 workers => ceil(250/4) == 63 => slices of 63/63/63/61,
    # so the final slice is short and every boundary is unaligned.
    smiles = _repeat_to(_fixture_smiles(), 250)
    assert len(smiles) >= low_parallel_threshold

    serial = _cache_keys_bytes16(smiles, featurizer, n_jobs=1)
    parallel = _cache_keys_bytes16(smiles, featurizer, n_jobs=4)

    npt.assert_array_equal(serial, parallel)


@pytest.mark.unit
def test_parallel_path_preserves_input_order(low_parallel_threshold: int) -> None:
    """REQ-7: concatenation is in input order, not completion order.

    Checked element-wise against a per-row reference rather than against the
    serial path, so a systematic reordering that happened to affect both
    cannot pass.
    """
    featurizer = _frozen()
    smiles = _repeat_to(_fixture_smiles(), 250)
    name = featurizer.get_name()
    config_hash = featurizer.get_config_hash()

    expected = np.array(
        [
            hashlib.blake2b(
                f'{s}_{name}_{config_hash}'.encode(),
                digest_size=16,
                usedforsecurity=False,
            ).digest()
            for s in smiles
        ],
        dtype=HASH_DTYPE,
    )

    npt.assert_array_equal(_cache_keys_bytes16(smiles, featurizer, n_jobs=4), expected)


@pytest.mark.unit
@pytest.mark.parametrize('n_rows', [65, 100, 127, 128, 129, 250])
def test_parallel_agrees_with_serial_across_slice_boundaries(
    low_parallel_threshold: int, n_rows: int
) -> None:
    """Row counts that divide evenly, unevenly, and by one either side."""
    featurizer = _frozen()
    smiles = _repeat_to(_fixture_smiles(), n_rows)

    npt.assert_array_equal(
        _cache_keys_bytes16(smiles, featurizer, n_jobs=1),
        _cache_keys_bytes16(smiles, featurizer, n_jobs=3),
    )


@pytest.mark.unit
def test_parallel_dispatch_actually_happens(
    low_parallel_threshold: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard against the equivalence tests passing because both paths are serial.

    ``_cache_keys_bytes16`` resolves ``joblib.Parallel`` at call time (the
    import lives inside the function, matching extraction.py), so patching the
    attribute on the joblib module observes the real dispatch.
    """
    import joblib

    calls: list[str] = []
    real_parallel = joblib.Parallel

    def _spy(*args, **kwargs):
        calls.append(kwargs.get('backend', ''))
        return real_parallel(*args, **kwargs)

    monkeypatch.setattr(joblib, 'Parallel', _spy)

    _cache_keys_bytes16(_repeat_to(_fixture_smiles(), 250), _frozen(), n_jobs=4)

    assert calls, 'expected joblib.Parallel dispatch above the threshold'
    assert calls[0] == 'loky', f'REQ-9 requires the loky backend, got {calls[0]!r}'


@pytest.mark.unit
def test_below_threshold_stays_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-7: small calls never pay loky dispatch overhead."""
    import joblib

    def _explode(*args, **kwargs):
        raise AssertionError('joblib.Parallel must not be used below the threshold')

    monkeypatch.setattr(joblib, 'Parallel', _explode)

    smiles = _fixture_smiles()
    assert len(smiles) < cache_module.PARALLEL_HASH_MIN_ROWS
    result = _cache_keys_bytes16(smiles, _frozen(), n_jobs=16)

    assert result.shape == (len(smiles),)


@pytest.mark.unit
def test_single_worker_stays_serial(
    low_parallel_threshold: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-7: effective_n_jobs(n_jobs) == 1 takes the serial loop even above N."""
    import joblib

    def _explode(*args, **kwargs):
        raise AssertionError('joblib.Parallel must not be used with one worker')

    monkeypatch.setattr(joblib, 'Parallel', _explode)

    result = _cache_keys_bytes16(
        _repeat_to(_fixture_smiles(), 250), _frozen(), n_jobs=1
    )

    assert result.shape == (250,)


@pytest.mark.unit
def test_parallel_threshold_is_at_the_measured_break_even() -> None:
    """Pin PARALLEL_HASH_MIN_ROWS to the T019 cold-pool break-even.

    Loky pool spin-up is a fixed ~8.7 s and the pool is cold on essentially
    every real call (``shutdown_featurization_pool`` runs once per cycle, and
    hashing precedes featurization in the ``cache_features`` wrapper). Below
    ~40M rows the dispatch costs more than the hashing it parallelises — at
    the originally-specified 200_000 it was measured 94x SLOWER than serial.
    Lowering this constant without re-measuring reintroduces that regression.
    """
    assert cache_module.PARALLEL_HASH_MIN_ROWS == 40_000_000


@pytest.mark.unit
def test_hash_smiles_slice_is_module_level_and_picklable() -> None:
    """REQ-9: the worker must be picklable for loky to ship it."""
    import pickle

    assert _hash_smiles_slice.__module__ == 'learnm8.features.cache'
    assert _hash_smiles_slice.__qualname__ == '_hash_smiles_slice'
    assert pickle.loads(pickle.dumps(_hash_smiles_slice)) is _hash_smiles_slice


@pytest.mark.unit
def test_wrapper_forwards_n_jobs_without_consuming_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-10: cache_features reads n_jobs with .get, so the wrapped fn still sees it.

    A ``.pop`` here would silently drop the caller's ``n_jobs`` on the floor
    and serialise the actual featurization — the exact opposite of 026's
    intent, and invisible to every correctness test.
    """
    seen_key_n_jobs: list[int] = []
    seen_inner_kwargs: list[dict] = []

    real_cache_keys = cache_module._cache_keys_bytes16

    def _spy_keys(smiles_list, featurizer, *, n_jobs=1):
        seen_key_n_jobs.append(n_jobs)
        return real_cache_keys(smiles_list, featurizer, n_jobs=n_jobs)

    monkeypatch.setattr(cache_module, '_cache_keys_bytes16', _spy_keys)

    @cache_module.cache_features(tmp_path)
    def _inner(smiles_list, featurizer, *args, **kwargs):
        seen_inner_kwargs.append(kwargs)
        return np.zeros((len(smiles_list), featurizer.get_dimension()), dtype=np.uint8)

    _inner(SAMPLE_SMILES, _make_morgan(), n_jobs=7, cache_dir=tmp_path)

    assert seen_key_n_jobs == [7], 'n_jobs must reach _cache_keys_bytes16'
    assert seen_inner_kwargs and seen_inner_kwargs[0].get('n_jobs') == 7, (
        'n_jobs must survive for the wrapped function (.get, not .pop)'
    )


@pytest.mark.unit
def test_hash_smiles_slice_matches_the_legacy_recipe() -> None:
    """The worker itself produces legacy-identical digests (REQ-8)."""
    suffix = b'_morgan_0123456789abcdef'
    smiles = [s for s, _ in LEGACY_REFERENCE_DIGESTS]
    expected = np.array(
        [bytes.fromhex(h) for _, h in LEGACY_REFERENCE_DIGESTS], dtype=HASH_DTYPE
    )

    result = _hash_smiles_slice(smiles, suffix)

    assert result.dtype == HASH_DTYPE
    npt.assert_array_equal(result, expected)

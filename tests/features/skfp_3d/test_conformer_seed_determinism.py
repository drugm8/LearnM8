"""Tests for SkfpFeaturizer.random_state plumbing (Feature 020 / T034-T039).

Covers FR-010 through FR-013 and SC-004 from
specs/020-cache-integrity/spec.md (User Story 4).

After the §1.6 factory collapse, the 9 individual 3D wrapper classes are gone;
all featurizers are constructed via :func:`learnm8.features.create_featurizer`.

Test coverage:
    T034 - byte-identical features with fixed seed (FR-010, FR-011, SC-004)
    T035 - different seeds produce different features (FR-011, SC-004)
    T036 - default and explicit seeds recorded in get_config (FR-010, FR-012)
    T037 - 2D featurizers do NOT include random_state (FR-012)
    T038 - GETAWAY positive cache-key disambiguation tests
    T039 - scikit-fingerprints<1.18.0 fallback warning (FR-013)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from learnm8.exceptions import LearnM8Warning
from learnm8.features import create_featurizer
from learnm8.features.base import DEFAULT_3D_RANDOM_STATE

CANONICAL_SMILES: list[str] = [
    'CCO',
    'c1ccccc1',
    'CC(=O)O',
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
]

IBUPROFEN_SMILES: str = 'CC(C)Cc1ccc(C(C)C(=O)O)cc1'


# ---------------------------------------------------------------------------
# Sanity: the constant is exactly 0xf00d (61453)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_3d_random_state_constant() -> None:
    """DEFAULT_3D_RANDOM_STATE must equal 0xf00d (61453, RDKit ETKDG convention)."""
    assert DEFAULT_3D_RANDOM_STATE == 0xF00D
    assert DEFAULT_3D_RANDOM_STATE == 61453


# ---------------------------------------------------------------------------
# T034: byte-identical features with fixed seed (FR-010, FR-011, SC-004)
# ---------------------------------------------------------------------------


_BYTE_IDENTICAL_NAMES: list[str] = [
    'whim',
    'usr',
    'usrcat',
    'e3fp',
    'morse',
    'rdf',
    'electroshape',
]


@pytest.mark.molecular
@pytest.mark.slow
@pytest.mark.parametrize('name', _BYTE_IDENTICAL_NAMES)
def test_byte_identical_with_fixed_seed(name: str) -> None:
    """Two instances with the same random_state produce byte-identical output.

    GETAWAY is excluded (rdkit#7264 non-determinism, see T038).
    autocorr is excluded (pre-existing use_3D=False bug; treated as
    2D-equivalent for this test).
    """
    f1 = create_featurizer(name, random_state=42, n_jobs=1)
    f2 = create_featurizer(name, random_state=42, n_jobs=1)

    a = f1.transform(CANONICAL_SMILES)
    b = f2.transform(CANONICAL_SMILES)

    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# T035: different seeds produce different features (FR-011, SC-004)
# ---------------------------------------------------------------------------


@pytest.mark.molecular
@pytest.mark.slow
def test_different_seed_different_features() -> None:
    """Different random_state values produce different conformer ensembles.

    Proves cache-key disambiguation: two cache rows for the same SMILES under
    different seeds, no silent overwrite.
    """
    f42 = create_featurizer('whim', random_state=42, n_jobs=1)
    f99 = create_featurizer('whim', random_state=99, n_jobs=1)

    smiles = [IBUPROFEN_SMILES]
    a = f42.transform(smiles)
    b = f99.transform(smiles)

    assert not np.array_equal(a, b), (
        'Expected different conformer ensembles for random_state=42 vs 99 '
        'on a flexible molecule (ibuprofen).'
    )


# ---------------------------------------------------------------------------
# T036: default and explicit seeds recorded in get_config (FR-010, FR-012)
# ---------------------------------------------------------------------------


_CONFIG_3D_NAMES: list[str] = ['whim', 'usr', 'e3fp']


@pytest.mark.unit
@pytest.mark.molecular
@pytest.mark.parametrize('name', _CONFIG_3D_NAMES)
def test_default_seed_is_recorded_in_config(name: str) -> None:
    """3D featurizers record DEFAULT_3D_RANDOM_STATE (0xf00d) when no seed given."""
    f = create_featurizer(name, n_jobs=1)
    config = f.get_config()
    assert 'random_state' in config
    assert config['random_state'] == 0xF00D
    assert config['random_state'] == 61453


@pytest.mark.unit
@pytest.mark.molecular
def test_explicit_seed_is_recorded_in_config() -> None:
    """3D featurizer records the user-supplied random_state value."""
    f = create_featurizer('whim', random_state=12345, n_jobs=1)
    config = f.get_config()
    assert config['random_state'] == 12345


# ---------------------------------------------------------------------------
# T037: 2D featurizers do NOT include random_state (FR-012)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.molecular
@pytest.mark.parametrize(
    'name,kwargs',
    [
        ('morgan', {}),
        ('morgan', {'count': True}),
        ('maccs', {}),
        ('rdkit', {}),
    ],
)
def test_2d_featurizer_config_does_not_include_random_state(
    name: str, kwargs: dict[str, Any]
) -> None:
    """2D featurizers preserve cache validity by NOT including random_state."""
    f = create_featurizer(name, n_jobs=1, **kwargs)
    config = f.get_config()
    assert 'random_state' not in config, (
        f'create_featurizer({name!r}, **{kwargs}) leaked random_state into '
        f'get_config(); this would invalidate existing 2D caches.'
    )


# ---------------------------------------------------------------------------
# T038: GETAWAY positive tests (cache-key disambiguation despite non-determinism)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.molecular
def test_getaway_random_state_in_config() -> None:
    """GETAWAY records random_state in get_config() (cache-key disambiguation)."""
    f = create_featurizer('getaway', random_state=42, n_jobs=1)
    config = f.get_config()
    assert 'random_state' in config
    assert config['random_state'] == 42


@pytest.mark.unit
@pytest.mark.molecular
def test_getaway_different_seeds_produce_different_cache_keys() -> None:
    """GETAWAY get_config_hash() differs by seed -> no silent cache overwrite.

    Note: this test does NOT assert byte-identical features. GETAWAY
    non-determinism is documented per rdkit#7264.
    """
    f42 = create_featurizer('getaway', random_state=42, n_jobs=1)
    f99 = create_featurizer('getaway', random_state=99, n_jobs=1)
    assert f42.get_config_hash() != f99.get_config_hash()


@pytest.mark.unit
@pytest.mark.skip(
    reason='GETAWAY non-deterministic per rdkit#7264 - cache-key-disambiguated only'
)
def test_getaway_byte_identical_skipped() -> None:
    """Codifies the GETAWAY byte-identity skip (see contract line 186-188)."""
    pytest.fail('should not run')


# ---------------------------------------------------------------------------
# T039: scikit-fingerprints<1.18.0 fallback (FR-013)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.molecular
def test_skfp_old_version_falls_back_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ConformerGenerator does not accept random_state, warn and fall back.

    Patches the import location used inside learnm8/features/base.py so that
    constructing ConformerGenerator(random_state=...) raises TypeError. The
    featurizer must:
      1. Emit a LearnM8Warning mentioning 'scikit-fingerprints<1.18.0'.
      2. Continue to construct successfully (no exception escapes).
      3. Set self.random_state = None (non-deterministic generation).
      4. Omit random_state from get_config() so cache keys differ from
         the deterministic path (fixes silent cache corruption).
    """
    import learnm8.features.base as base_module

    real_ctor = base_module.ConformerGenerator

    class _FakeConformerGenerator:
        """Stub that rejects random_state, simulating skfp<1.18.0."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if 'random_state' in kwargs:
                raise TypeError(
                    'ConformerGenerator() got an unexpected keyword argument '
                    "'random_state'"
                )
            self._real = real_ctor(*args, **kwargs)

        def transform(self, mols: Any) -> Any:
            return self._real.transform(mols)

    monkeypatch.setattr(base_module, 'ConformerGenerator', _FakeConformerGenerator)

    with pytest.warns(LearnM8Warning, match='scikit-fingerprints<1.18.0'):
        f = create_featurizer('whim', random_state=42, n_jobs=1)

    assert f.random_state is None
    assert f.conformer_gen is not None
    assert 'random_state' not in f.get_config()

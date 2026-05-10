"""Tests for SkfpFeaturizer.random_state plumbing (Feature 020 / T034-T039).

Covers FR-010 through FR-013 and SC-004 from
specs/020-cache-integrity/spec.md (User Story 4).

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
from learnm8.features.base import DEFAULT_3D_RANDOM_STATE


CANONICAL_SMILES: list[str] = [
    'CCO',
    'c1ccccc1',
    'CC(=O)O',
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
]

IBUPROFEN_SMILES: str = 'CC(C)Cc1ccc(C(C)C(=O)O)cc1'


def _import_3d(class_name: str) -> Any:
    """Import a 3D featurizer class or skip the test if unavailable."""
    module = pytest.importorskip(
        f'learnm8.features.skfp_3d.{class_name.lower().replace("featurizer", "")}'
    )
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Sanity: the constant is exactly 0xf00d (61453)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_3d_random_state_constant() -> None:
    """DEFAULT_3D_RANDOM_STATE must equal 0xf00d (61453, RDKit ETKDG convention)."""
    assert DEFAULT_3D_RANDOM_STATE == 0xf00d
    assert DEFAULT_3D_RANDOM_STATE == 61453


# ---------------------------------------------------------------------------
# T034: byte-identical features with fixed seed (FR-010, FR-011, SC-004)
# ---------------------------------------------------------------------------


_BYTE_IDENTICAL_CLASSES: list[str] = [
    'WHIMFeaturizer',
    'USRFeaturizer',
    'USRCATFeaturizer',
    'E3FPFeaturizer',
    'MORSEFeaturizer',
    'RDFFeaturizer',
    'ElectroShapeFeaturizer',
]


@pytest.mark.molecular
@pytest.mark.slow
@pytest.mark.parametrize('featurizer_class_name', _BYTE_IDENTICAL_CLASSES)
def test_byte_identical_with_fixed_seed(featurizer_class_name: str) -> None:
    """Two instances with the same random_state produce byte-identical output.

    GETAWAY is excluded (rdkit#7264 non-determinism, see T038).
    AutocorrFeaturizer is excluded (pre-existing use_3D=False bug; treated as
    2D-equivalent for this test).
    """
    featurizer_cls = _import_3d(featurizer_class_name)

    f1 = featurizer_cls(random_state=42)
    f2 = featurizer_cls(random_state=42)

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
    WHIMFeaturizer = _import_3d('WHIMFeaturizer')

    f42 = WHIMFeaturizer(random_state=42)
    f99 = WHIMFeaturizer(random_state=99)

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


_CONFIG_3D_CLASSES: list[str] = [
    'WHIMFeaturizer',
    'USRFeaturizer',
    'E3FPFeaturizer',
]


@pytest.mark.unit
@pytest.mark.molecular
@pytest.mark.parametrize('featurizer_class_name', _CONFIG_3D_CLASSES)
def test_default_seed_is_recorded_in_config(featurizer_class_name: str) -> None:
    """3D featurizers record DEFAULT_3D_RANDOM_STATE (0xf00d) when no seed given."""
    featurizer_cls = _import_3d(featurizer_class_name)
    f = featurizer_cls()
    config = f.get_config()
    assert 'random_state' in config
    assert config['random_state'] == 0xf00d
    assert config['random_state'] == 61453


@pytest.mark.unit
@pytest.mark.molecular
def test_explicit_seed_is_recorded_in_config() -> None:
    """3D featurizer records the user-supplied random_state value."""
    WHIMFeaturizer = _import_3d('WHIMFeaturizer')
    f = WHIMFeaturizer(random_state=12345)
    config = f.get_config()
    assert config['random_state'] == 12345


# ---------------------------------------------------------------------------
# T037: 2D featurizers do NOT include random_state (FR-012)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.molecular
@pytest.mark.parametrize(
    'featurizer_class_name,kwargs',
    [
        ('MorganFeaturizer', {}),
        ('MorganFeaturizer', {'count': True}),
        ('MACCSFeaturizer', {}),
        ('RDKitFeaturizer', {}),
    ],
)
def test_2d_featurizer_config_does_not_include_random_state(
    featurizer_class_name: str, kwargs: dict[str, Any]
) -> None:
    """2D featurizers preserve cache validity by NOT including random_state."""
    module_name = featurizer_class_name.lower().replace('featurizer', '')
    if module_name == 'rdkit':
        module_path = 'learnm8.features.skfp_2d.rdkit'
    elif module_name == 'maccs':
        module_path = 'learnm8.features.skfp_2d.maccs'
    elif module_name == 'morgan':
        module_path = 'learnm8.features.skfp_2d.morgan'
    else:
        module_path = f'learnm8.features.skfp_2d.{module_name}'

    module = pytest.importorskip(module_path)
    featurizer_cls = getattr(module, featurizer_class_name)
    f = featurizer_cls(**kwargs)
    config = f.get_config()
    assert 'random_state' not in config, (
        f'{featurizer_class_name}(**{kwargs}) leaked random_state into '
        f'get_config(); this would invalidate existing 2D caches.'
    )


# ---------------------------------------------------------------------------
# T038: GETAWAY positive tests (cache-key disambiguation despite non-determinism)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.molecular
def test_getaway_random_state_in_config() -> None:
    """GETAWAY records random_state in get_config() (cache-key disambiguation)."""
    GETAWAYFeaturizer = _import_3d('GETAWAYFeaturizer')
    f = GETAWAYFeaturizer(random_state=42)
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
    GETAWAYFeaturizer = _import_3d('GETAWAYFeaturizer')
    f42 = GETAWAYFeaturizer(random_state=42)
    f99 = GETAWAYFeaturizer(random_state=99)
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
      3. Still record self.random_state on the instance.
    """
    import learnm8.features.base as base_module

    real_ctor = base_module.ConformerGenerator

    class _FakeConformerGenerator:
        """Stub that rejects random_state, simulating skfp<1.18.0."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if 'random_state' in kwargs:
                raise TypeError(
                    "ConformerGenerator() got an unexpected keyword argument "
                    "'random_state'"
                )
            # Fallback path: defer to the real constructor without random_state.
            self._real = real_ctor(*args, **kwargs)

        def transform(self, mols: Any) -> Any:
            return self._real.transform(mols)

    monkeypatch.setattr(base_module, 'ConformerGenerator', _FakeConformerGenerator)

    WHIMFeaturizer = _import_3d('WHIMFeaturizer')

    with pytest.warns(LearnM8Warning, match='scikit-fingerprints<1.18.0'):
        f = WHIMFeaturizer(random_state=42)

    # Featurizer constructed successfully and still tracks the requested seed.
    assert f.random_state == 42
    assert f.conformer_gen is not None

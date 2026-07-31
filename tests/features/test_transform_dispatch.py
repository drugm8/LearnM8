"""Parity and dispatch tests for ``SkfpFeaturizer.transform`` (feature 026).

Feature 026 stops materialising RDKit ``Mol`` objects in the calling process
for every featurizer that can accept SMILES directly, so the fingerprint parses
worker-side instead. The value contract is absolute: no featurizer's output may
change for any input.

These tests pin that contract by running the pre-026 path
(``fingerprint.transform(mol_from_smiles.transform(smiles))``) alongside the
new dispatch and comparing:

  - REQ-1: every non-3D, non-SMILES-consuming featurizer is byte-identical and
    never touches ``mol_from_smiles``
  - REQ-3: ``lingo`` / ``mhfp`` / ``secfp`` keep the ``Mol`` round-trip, so
    non-canonical input still canonicalises
  - REQ-4: the large-input warning fires exactly once, and only above threshold

The 3D path (REQ-2) is covered by ``tests/features/skfp_3d/``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from learnm8.exceptions import FeatureExtractionError
from learnm8.features import (
    FEATURIZER_REGISTRY,
    FEATURIZERS_2D,
    FEATURIZERS_3D,
    SkfpFeaturizer,
    create_featurizer,
)
from learnm8.features import base as base_module
from learnm8.features.base import (
    _SMILES_CONSUMING_FINGERPRINTS,
    SMILES_CONSUMING_WARN_ROWS,
    _reset_smiles_consuming_warning,
)

# Registry names whose skfp class reads the SMILES text itself (REQ-3).
SMILES_CONSUMING_NAMES: tuple[str, ...] = ('lingo', 'mhfp', 'secfp')


def _takes_fast_path(name: str) -> bool:
    """Decide from RUNTIME behaviour, not registry membership.

    ``transform`` branches on ``requires_3d()`` (i.e. the skfp instance's
    ``requires_conformers`` attribute), which does not always agree with the
    FEATURIZERS_3D set: ``autocorr`` is listed as 3D but reports
    ``requires_conformers=False``, so it actually takes the fast path. Deriving
    the sweep from the registry sets would silently skip it.
    """
    if name in SMILES_CONSUMING_NAMES:
        return False
    return not create_featurizer(name, n_jobs=1).requires_3d()


# Every registry featurizer that reaches the REQ-1 branch. Sorted for stable ids.
FAST_PATH_NAMES: tuple[str, ...] = tuple(
    sorted(name for name in FEATURIZER_REGISTRY if _takes_fast_path(name))
)

# RDKit 2026.03.2 aborts with a Pre-condition Violation ("bad bond stereo",
# Canon.cpp:362, dblBond->getStereo() > Bond::STEREOANY) while canonicalising
# this fixture molecule. It is NOT a feature-026 regression: map4, mhfp and
# secfp raise identically on the pre-026 Mol path and the post-026 SMILES path,
# because all three route through MHFPEncoder/MolToSmiles canonicalisation.
# Verified by transforming the molecule alone through both paths on this and on
# a clean checkout. Excluded from those three featurizers' parity sweeps only —
# every other featurizer handles it, and dropping it globally would weaken the
# comparison for the other 27.
RDKIT_STEREO_BUG_SMILES: str = r'Cn1c2ccccc2s/c1=N\N=C\c1ccccn1'
RDKIT_STEREO_BUG_FEATURIZERS: frozenset[str] = frozenset({'map4', 'mhfp', 'secfp'})


def _load_fixture() -> list[str]:
    """Real ZINC molecules — §3.4 red line 1 forbids synthetic molecular data."""
    path = Path(__file__).parent.parent / 'data' / 'diverse_molecules.csv'
    with path.open() as handle:
        return [row['SMILES'] for row in csv.DictReader(handle)]


@pytest.fixture(scope='module')
def fixture_smiles() -> list[str]:
    return _load_fixture()


def _smiles_for(name: str, smiles: list[str]) -> list[str]:
    """Fixture rows a given featurizer can actually process on this RDKit."""
    if name in RDKIT_STEREO_BUG_FEATURIZERS:
        return [s for s in smiles if s != RDKIT_STEREO_BUG_SMILES]
    return smiles


def _legacy_transform(featurizer: SkfpFeaturizer, smiles: list[str]) -> np.ndarray:
    """Reproduce the pre-026 transform body: always Mol-ify, then fingerprint.

    Mirrors the dtype narrowing the real ``transform`` applies so the comparison
    is against the value users actually received, not the raw skfp output.
    """
    mols = featurizer.mol_from_smiles.transform(smiles)
    raw = np.asarray(featurizer.fingerprint.transform(mols))
    return raw.astype(np.dtype(featurizer.get_compute_dtype()), copy=False)


def _assert_matches(new: np.ndarray, legacy: np.ndarray, name: str) -> None:
    """Binary/integer must be exactly equal; float within np.allclose."""
    assert new.shape == legacy.shape, f'{name}: shape {new.shape} != {legacy.shape}'
    assert new.dtype == legacy.dtype, f'{name}: dtype {new.dtype} != {legacy.dtype}'

    if np.issubdtype(new.dtype, np.floating):
        # Descriptor featurizers legitimately emit NaN; treat NaN as matching
        # NaN in the same cell, which np.allclose does under equal_nan.
        assert np.allclose(new, legacy, equal_nan=True), (
            f'{name}: float features diverged from the pre-026 path'
        )
    else:
        npt.assert_array_equal(
            new, legacy, err_msg=f'{name}: integer features diverged'
        )


# ---------------------------------------------------------------------------
# REQ-1: fast-path parity
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize('name', FAST_PATH_NAMES)
def test_fast_path_matches_pre_026_output(name: str, fixture_smiles: list[str]) -> None:
    """REQ-1: passing SMILES through changes no feature value."""
    featurizer = create_featurizer(name, n_jobs=1)
    smiles = _smiles_for(name, fixture_smiles)

    _assert_matches(
        featurizer.transform(smiles), _legacy_transform(featurizer, smiles), name
    )


@pytest.mark.unit
@pytest.mark.parametrize('name', FAST_PATH_NAMES)
def test_fast_path_never_builds_mols_in_parent(
    name: str, fixture_smiles: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-1: the whole point — no parent-heap Mol list.

    Parity alone cannot prove this: passing Mols would give identical values.
    Detonating ``mol_from_smiles.transform`` is what pins the memory win.
    """
    featurizer = create_featurizer(name, n_jobs=1)

    def _explode(*args, **kwargs):
        raise AssertionError(
            f'{name} must not materialise Mol objects in the calling process'
        )

    monkeypatch.setattr(featurizer.mol_from_smiles, 'transform', _explode)

    smiles = _smiles_for(name, fixture_smiles)
    result = featurizer.transform(smiles)
    assert result.shape[0] == len(smiles)


@pytest.mark.unit
def test_every_registry_featurizer_lands_in_exactly_one_branch() -> None:
    """Guard against a featurizer silently dropping out of the parity sweep.

    Partitions the whole registry into the three ``transform`` branches and
    asserts the union is complete and the parts are disjoint, so adding a
    featurizer without extending this file fails loudly.
    """
    conformer_path = {
        name
        for name in FEATURIZER_REGISTRY
        if create_featurizer(name, n_jobs=1).requires_3d()
    }
    fast = set(FAST_PATH_NAMES)
    consuming = set(SMILES_CONSUMING_NAMES)

    assert fast | consuming | conformer_path == set(FEATURIZER_REGISTRY)
    assert not fast & consuming
    assert not fast & conformer_path
    assert not consuming & conformer_path
    assert len(FAST_PATH_NAMES) >= 25, 'parity sweep unexpectedly small'


@pytest.mark.unit
def test_autocorr_is_swept_despite_being_registered_3d() -> None:
    """Regression pin for the registry/runtime mismatch the sweep uncovered.

    ``autocorr`` is in FEATURIZERS_3D but its skfp instance reports
    ``requires_conformers=False``, so it reaches the REQ-1 fast path. A sweep
    built from FEATURIZERS_2D would not have covered it.
    """
    assert 'autocorr' in FEATURIZERS_3D
    assert 'autocorr' not in FEATURIZERS_2D
    assert not create_featurizer('autocorr', n_jobs=1).requires_3d()
    assert 'autocorr' in FAST_PATH_NAMES


@pytest.mark.unit
@pytest.mark.parametrize('name', sorted(RDKIT_STEREO_BUG_FEATURIZERS))
def test_rdkit_stereo_bug_is_path_independent(name: str) -> None:
    """The excluded molecule fails on BOTH paths — proving it is not a 026 bug.

    If a future RDKit fixes ``Canon.cpp`` this test starts failing, which is the
    signal to delete :data:`RDKIT_STEREO_BUG_SMILES` and restore full coverage.
    If instead a new featurizer starts failing, the exclusion list is stale.
    """
    featurizer = create_featurizer(name, n_jobs=1)
    offender = [RDKIT_STEREO_BUG_SMILES]

    with pytest.raises(Exception):  # noqa: B017 - RDKit raises bare RuntimeError
        featurizer.fingerprint.transform(featurizer.mol_from_smiles.transform(offender))

    with pytest.raises(Exception):  # noqa: B017
        featurizer.fingerprint.transform(offender)


@pytest.mark.unit
def test_rdkit_stereo_bug_molecule_is_fine_everywhere_else(
    fixture_smiles: list[str],
) -> None:
    """Only the three known featurizers choke on it; the exclusion is minimal."""
    assert RDKIT_STEREO_BUG_SMILES in fixture_smiles

    featurizer = create_featurizer('morgan', n_jobs=1)
    result = featurizer.transform([RDKIT_STEREO_BUG_SMILES])

    assert result.shape[0] == 1


@pytest.mark.unit
def test_rdkit_2d_descriptors_takes_the_fast_path() -> None:
    """It calls skfp's ensure_smiles but discards the result — see the audit note.

    descriptastorus' ``calculateMol(m, smiles)`` ignores its ``smiles``
    argument, so the canonicalisation cannot reach the output and the
    fingerprint belongs on the fast path despite matching a naive
    ``grep ensure_smiles``.
    """
    featurizer = create_featurizer('rdkit_2d_descriptors', n_jobs=1)
    assert (
        featurizer.fingerprint.__class__.__name__ not in _SMILES_CONSUMING_FINGERPRINTS
    )


# ---------------------------------------------------------------------------
# REQ-3: SMILES-consuming fingerprints keep the Mol round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize('name', SMILES_CONSUMING_NAMES)
def test_smiles_consuming_are_denylisted(name: str) -> None:
    featurizer = create_featurizer(name, n_jobs=1)
    assert featurizer.fingerprint.__class__.__name__ in _SMILES_CONSUMING_FINGERPRINTS


@pytest.mark.unit
@pytest.mark.parametrize('name', SMILES_CONSUMING_NAMES)
def test_smiles_consuming_matches_pre_026_output(
    name: str, fixture_smiles: list[str]
) -> None:
    """REQ-3: these still go through Mol, so they are trivially unchanged."""
    featurizer = create_featurizer(name, n_jobs=1)
    smiles = _smiles_for(name, fixture_smiles)

    _assert_matches(
        featurizer.transform(smiles), _legacy_transform(featurizer, smiles), name
    )


@pytest.mark.unit
@pytest.mark.parametrize('name', SMILES_CONSUMING_NAMES)
def test_smiles_consuming_still_canonicalises(name: str) -> None:
    """REQ-3 acceptance: 'OCC' and 'CCO' are the same molecule, same vector.

    This is the test that would fail if these three were moved to the fast
    path — skfp's ensure_smiles canonicalises Mol input via MolToSmiles, and
    without the round-trip the raw strings would hash/slice differently.
    """
    featurizer = create_featurizer(name, n_jobs=1)

    features = featurizer.transform(['OCC', 'CCO'])

    npt.assert_array_equal(
        features[0],
        features[1],
        err_msg=f'{name}: non-canonical SMILES produced a different vector',
    )


@pytest.mark.unit
@pytest.mark.parametrize('name', SMILES_CONSUMING_NAMES)
def test_smiles_consuming_uses_mol_round_trip(name: str) -> None:
    """REQ-3: prove the round-trip is actually taken, not incidentally equal."""
    featurizer = create_featurizer(name, n_jobs=1)
    calls: list[int] = []
    real_transform = featurizer.mol_from_smiles.transform

    def _spy(smiles_list):
        calls.append(len(smiles_list))
        return real_transform(smiles_list)

    featurizer.mol_from_smiles.transform = _spy  # type: ignore[method-assign]
    featurizer.transform(['OCC', 'CCO'])

    assert calls == [2], f'{name} must convert SMILES to Mol in the parent'


# ---------------------------------------------------------------------------
# REQ-4: one-shot large-input warning
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_warning_guard():
    """The guard is process-global; isolate every test from the others."""
    _reset_smiles_consuming_warning()
    yield
    _reset_smiles_consuming_warning()


@pytest.fixture(autouse=True)
def restore_log_propagation():
    """Let caplog see learnm8 records regardless of test ordering.

    ``learnm8.utils.logging.setup_logging`` sets ``propagate = False`` on the
    ``learnm8`` logger. Once any earlier test in the session calls it, records
    from ``learnm8.*`` never reach the root logger where pytest's caplog
    handler is installed, and warning assertions here fail depending purely on
    execution order. Forcing propagation on for the duration of each test makes
    these assertions order-independent.
    """
    learnm8_logger = logging.getLogger('learnm8')
    previous = learnm8_logger.propagate
    learnm8_logger.propagate = True
    yield
    learnm8_logger.propagate = previous


@pytest.fixture
def low_warn_threshold(monkeypatch: pytest.MonkeyPatch) -> int:
    """Drop SMILES_CONSUMING_WARN_ROWS so the test does not need 1M molecules."""
    threshold = 4
    monkeypatch.setattr(base_module, 'SMILES_CONSUMING_WARN_ROWS', threshold)
    return threshold


@pytest.mark.unit
def test_warning_fires_once_above_threshold(
    low_warn_threshold: int, caplog: pytest.LogCaptureFixture
) -> None:
    """REQ-4: fires on the first large call and never again."""
    featurizer = create_featurizer('lingo', n_jobs=1)
    smiles = ['CCO'] * (low_warn_threshold + 1)

    with caplog.at_level(logging.WARNING, logger=base_module.__name__):
        featurizer.transform(smiles)
        featurizer.transform(smiles)
        featurizer.transform(smiles)

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and 'consumes SMILES strings directly' in r.message
    ]
    assert len(warnings) == 1, f'expected exactly one warning, got {len(warnings)}'
    assert 'lingo' in warnings[0].message


@pytest.mark.unit
def test_warning_never_fires_below_threshold(
    low_warn_threshold: int, caplog: pytest.LogCaptureFixture
) -> None:
    """REQ-4: small calls stay silent."""
    featurizer = create_featurizer('lingo', n_jobs=1)

    with caplog.at_level(logging.WARNING, logger=base_module.__name__):
        featurizer.transform(['CCO'] * (low_warn_threshold - 1))

    assert not [
        r for r in caplog.records if 'consumes SMILES strings directly' in r.message
    ]


@pytest.mark.unit
def test_warning_fires_exactly_at_threshold(
    low_warn_threshold: int, caplog: pytest.LogCaptureFixture
) -> None:
    """REQ-4 says 'at least SMILES_CONSUMING_WARN_ROWS' — the boundary is inclusive."""
    featurizer = create_featurizer('lingo', n_jobs=1)

    with caplog.at_level(logging.WARNING, logger=base_module.__name__):
        featurizer.transform(['CCO'] * low_warn_threshold)

    assert [
        r for r in caplog.records if 'consumes SMILES strings directly' in r.message
    ]


@pytest.mark.unit
def test_warning_does_not_fire_for_fast_path_featurizers(
    low_warn_threshold: int, caplog: pytest.LogCaptureFixture
) -> None:
    """A featurizer with no parent-heap cost must not claim one."""
    featurizer = create_featurizer('morgan', n_jobs=1)

    with caplog.at_level(logging.WARNING, logger=base_module.__name__):
        featurizer.transform(['CCO'] * (low_warn_threshold + 10))

    assert not [
        r for r in caplog.records if 'consumes SMILES strings directly' in r.message
    ]


@pytest.mark.unit
def test_reset_hook_rearms_the_warning(
    low_warn_threshold: int, caplog: pytest.LogCaptureFixture
) -> None:
    """REQ-4: the guard is resettable by the private test hook."""
    featurizer = create_featurizer('lingo', n_jobs=1)
    smiles = ['CCO'] * (low_warn_threshold + 1)

    with caplog.at_level(logging.WARNING, logger=base_module.__name__):
        featurizer.transform(smiles)
        _reset_smiles_consuming_warning()
        featurizer.transform(smiles)

    warnings = [
        r for r in caplog.records if 'consumes SMILES strings directly' in r.message
    ]
    assert len(warnings) == 2


@pytest.mark.unit
def test_production_threshold_is_one_million() -> None:
    """Pin the documented constant so a stray edit is visible."""
    assert SMILES_CONSUMING_WARN_ROWS == 1_000_000


# ---------------------------------------------------------------------------
# REQ-2: the 3D guard is reachable through the new branch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_3d_without_conformers_still_raises() -> None:
    """REQ-2: the disabled-conformer error survives the dispatch rewrite."""
    featurizer = create_featurizer('whim', n_jobs=1, auto_generate_conformers=False)
    assert featurizer.conformer_gen is None

    with pytest.raises(FeatureExtractionError, match='requires 3D conformers'):
        featurizer.transform(['CCO'])


# ---------------------------------------------------------------------------
# Guard rails on SkfpFeaturizer paths the 026 dispatch sits between
# ---------------------------------------------------------------------------
# These cover pre-026 behaviour that had no test, and that the constitution's
# >=85% module gate for features/base.py requires. They are included here
# because the dispatch rewrite runs immediately after the length guard and
# immediately before the dimension lookup, so a regression in either would
# surface as a 026 failure.


@pytest.mark.unit
@pytest.mark.parametrize('name', ['morgan', 'maccs', 'lingo'])
def test_unparseable_smiles_still_raises_feature_extraction_error(name: str) -> None:
    """Both dispatch paths surface invalid SMILES as FeatureExtractionError.

    Found while running the T018 measurement over the Enamine REAL library,
    which contains chemically invalid entries such as the bad-valence SMILES
    below. The all-valid ``diverse_molecules`` fixture never exercised this.

    The two paths reach the failure differently — the pre-026 path builds a
    ``None`` Mol and skfp rejects the NoneType, while the fast path fails
    inside skfp's own parse — but both raise ``TypeError`` internally and
    ``transform`` wraps either into ``FeatureExtractionError``. The public
    contract is therefore unchanged; only the message text differs, and the
    new one is more specific (it names the offending SMILES and its index).
    """
    featurizer = create_featurizer(name, n_jobs=1)
    invalid = ['CCO', 'CC[O-]C=CC(=O)NCC(=S)N', 'c1ccccc1']

    with pytest.raises(FeatureExtractionError, match='Feature extraction failed'):
        featurizer.transform(invalid)


@pytest.mark.unit
def test_overlong_smiles_is_rejected_before_dispatch() -> None:
    """The MAX_SMILES_LENGTH guard runs ahead of all three branches."""
    featurizer = create_featurizer('morgan', n_jobs=1)
    too_long = 'C' * (base_module.MAX_SMILES_LENGTH + 1)

    with pytest.raises(FeatureExtractionError, match='exceeds maximum allowed'):
        featurizer.transform([too_long])


@pytest.mark.unit
def test_empty_input_short_circuits_before_dispatch() -> None:
    """An empty list returns an empty matrix without reaching a fingerprint."""
    featurizer = create_featurizer('morgan', n_jobs=1)

    result = featurizer.transform([])

    assert result.shape == (0, featurizer.get_dimension())


class _StubFingerprint:
    """Minimal fingerprint exposing one chosen dimension attribute."""

    requires_conformers = False

    def __init__(self, attr: str | None, value: int, params: dict | None = None):
        if attr is not None:
            setattr(self, attr, value)
        self._params = params or {}

    def get_params(self) -> dict:
        return self._params


def _stub_featurizer(fingerprint) -> SkfpFeaturizer:
    return SkfpFeaturizer(
        fingerprint,
        auto_generate_conformers=False,
        n_jobs=1,
        storage_dtype='float32',
        feature_type='continuous',
        fingerprint_name='stub',
        fingerprint_params={},
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ('attr', 'value'),
    [
        ('n_features_out', 128),
        ('n_features_out_', 256),
        ('fp_size', 512),
        ('n_features', 64),
    ],
)
def test_get_dimension_attribute_fallback_chain(attr: str, value: int) -> None:
    """Each rung of the get_dimension lookup resolves to its attribute."""
    featurizer = _stub_featurizer(_StubFingerprint(attr, value))

    assert featurizer.get_dimension() == value


@pytest.mark.unit
@pytest.mark.parametrize('key', ['fp_size', 'n_features'])
def test_get_dimension_falls_back_to_get_params(key: str) -> None:
    """With no attribute set, the dimension comes from get_params()."""
    featurizer = _stub_featurizer(_StubFingerprint(None, 0, params={key: 321}))

    assert featurizer.get_dimension() == 321


@pytest.mark.unit
def test_get_dimension_raises_when_undiscoverable() -> None:
    """A fingerprint exposing nothing usable fails loudly, not silently."""
    featurizer = _stub_featurizer(_StubFingerprint(None, 0, params={}))

    with pytest.raises(AttributeError, match='Cannot determine feature dimension'):
        featurizer.get_dimension()


@pytest.mark.unit
def test_get_description_falls_back_to_name() -> None:
    """Without an explicit description, the featurizer name is used."""
    featurizer = _stub_featurizer(_StubFingerprint('fp_size', 16))

    assert featurizer.get_description() == featurizer.get_name() == 'stub'

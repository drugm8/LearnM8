"""Test-time signature audit of LEARNER_REGISTRY (feature 023, D2).

Catches the case where a registered learner accepts ``compute_uncertainty``
via ``**kwargs`` (silently ignoring it) — the audit asserts the keyword-only
parameter is *explicit* with a default of ``True``. See also
``tests/learners/test_compute_uncertainty_behavior.py`` for the orthogonal
behavioral test (Dissent B resolution).
"""

from __future__ import annotations

import inspect

import pytest

from learnm8.api import LEARNER_REGISTRY, SMILES_NATIVE_LEARNERS
from learnm8.core.interfaces import Learner


@pytest.mark.unit
@pytest.mark.parametrize('name', sorted(LEARNER_REGISTRY.keys()))
def test_every_registered_learner_accepts_compute_uncertainty(name):
    """Every learner class registered in LEARNER_REGISTRY must declare an
    explicit ``compute_uncertainty`` keyword-only parameter with default
    ``True`` on its ``predict`` method (FR-001)."""
    cls = LEARNER_REGISTRY[name]
    sig = inspect.signature(cls.predict)
    params = sig.parameters

    assert 'compute_uncertainty' in params, (
        f'{name}.predict missing compute_uncertainty parameter; expected '
        f'keyword-only argument with default=True. Migration guide: see '
        f'CHANGELOG entry for v0.11.0.'
    )

    p = params['compute_uncertainty']
    assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
        f'{name}.predict has compute_uncertainty as {p.kind.name}; '
        f'expected KEYWORD_ONLY (after a bare * in the signature).'
    )
    assert p.default is True, (
        f'{name}.predict has compute_uncertainty default={p.default!r}; '
        f'expected True so existing call sites keep working unchanged.'
    )


@pytest.mark.unit
def test_smiles_native_learners_matches_requires_smiles():
    """``SMILES_NATIVE_LEARNERS`` must list exactly the registered learners
    whose ``requires_smiles()`` is True.

    The CLI featurizer guard reads this set instead of instantiating a learner,
    so drift would either reject a valid configuration (chemprop with no
    featurizer) or wave through a feature-based learner that needs one.

    Only classes overriding the base method are instantiated — ``Learner``'s
    default returns False, so unchanged classes need no construction.
    """
    overriders = {
        name: cls
        for name, cls in LEARNER_REGISTRY.items()
        if cls.requires_smiles is not Learner.requires_smiles
    }
    smiles_native = {
        name for name, cls in overriders.items() if cls().requires_smiles()
    }

    assert smiles_native == set(SMILES_NATIVE_LEARNERS), (
        f'SMILES_NATIVE_LEARNERS is stale: registry reports {sorted(smiles_native)}, '
        f'constant holds {sorted(SMILES_NATIVE_LEARNERS)}. Update learnm8/api.py.'
    )

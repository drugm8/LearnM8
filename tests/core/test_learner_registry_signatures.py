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

from learnm8.api import LEARNER_REGISTRY


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

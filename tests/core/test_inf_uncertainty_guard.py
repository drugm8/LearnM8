"""Feature 023 FR-012: ``assert_no_inf_uncertainty`` must short-circuit on None.

When the cycle resolves ``compute_uncertainty=False`` and the learner returns
None for the uncertainty array, the guard at ``cycle.py:263`` is gated on
``if uncertainties is not None``. This test pins that contract and also pins
the symmetric behaviour on the helper itself: calling
``assert_no_inf_uncertainty`` with None must NOT raise.
"""

from __future__ import annotations

import numpy as np
import pytest

from learnm8.utils.numerical import assert_no_inf_uncertainty

pytestmark = pytest.mark.unit


def test_assert_no_inf_uncertainty_short_circuits_on_none():
    """Calling the guard with None must be a no-op (feature 023 FR-012)."""
    # No exception raised.
    assert_no_inf_uncertainty(None, ['a', 'b', 'c'])  # type: ignore[arg-type]


def test_assert_no_inf_uncertainty_still_raises_on_inf():
    """Regression guard: the canonical Inf check still fires on real arrays."""
    bad = np.array([1.0, np.inf, 3.0])
    with pytest.raises(Exception, match='Inf uncertainties'):
        assert_no_inf_uncertainty(bad, ['x', 'y', 'z'])


def test_assert_no_inf_uncertainty_passes_on_finite():
    """Sanity: finite arrays pass without raising."""
    good = np.array([1.0, 2.0, 3.0])
    assert_no_inf_uncertainty(good, ['x', 'y', 'z'])

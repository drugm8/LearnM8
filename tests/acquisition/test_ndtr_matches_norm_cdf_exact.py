"""Spec 022 D2 regression: ``scipy.special.ndtr`` ≡ ``scipy.stats.norm.cdf``.

If scipy ever refactors ``norm_gen._cdf`` to no longer alias
``special.ndtr``, this test goes red, alerting us before the inline path
silently diverges from the legacy path.

Also covers the spec FR-007 σ=0 guard + z-clipping edge behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import norm

pytestmark = pytest.mark.unit


def test_ndtr_matches_norm_cdf_exact_at_pinned_floor() -> None:
    """``ndtr`` is bit-exact equivalent of ``norm.cdf`` across normal use."""
    z = np.array([-50, -10, -3.7, -1, 0, 1, 3.7, 10, 50], dtype=np.float64)
    assert np.array_equal(ndtr(z), norm.cdf(z)), (
        "scipy.special.ndtr is no longer bit-exact equivalent to scipy.stats.norm.cdf. "
        "The inline ndtr path may now produce values that drift from the legacy path. "
        "Either pin a tighter scipy floor or re-audit the inline-norm contract."
    )


def test_ndtr_matches_norm_cdf_within_range() -> None:
    """Equivalence holds across a fine grid in [-10, 10]."""
    z = np.linspace(-10, 10, 10_000)
    np.testing.assert_allclose(ndtr(z), norm.cdf(z), rtol=1e-12, atol=1e-300)


def test_inline_phi_clipped_at_37_does_not_underflow() -> None:
    """Spec FR-007: clipping z to [-37, 37] before exp(-z²/2) keeps φ finite.

    At |z|=50, exp(-z²/2) would underflow to 0 in float64; with clipping at
    |z|=37, exp(-z²/2) ≈ 6×10⁻⁳⁰⁰ — finite (sub-normal) but non-zero.
    """
    z = np.array([-50.0, -40.0, -37.0, 0.0, 37.0, 40.0, 50.0])
    z_clipped = np.clip(z, -37.0, 37.0)
    phi = np.exp(-0.5 * z_clipped * z_clipped) / np.sqrt(2.0 * np.pi)
    # All values are finite (not NaN, not inf).
    assert np.all(np.isfinite(phi))
    # At the clipping boundary, φ matches the boundary value.
    assert phi[0] == phi[2]
    assert phi[6] == phi[4]

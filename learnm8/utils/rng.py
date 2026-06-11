"""Counter-based RNG for chunking-invariant stochastic acquisition.

Streaming acquisition (feature: streaming predict→select fusion) scores the
candidate pool one chunk at a time. A *stateful* ``np.random.Generator`` would
make the draw assigned to candidate *i* depend on how the pool was chunked and
on host memory (which sets the prediction batch size). That breaks
reproducibility across machines.

The fix is a counter-based keyed hash: the value at global stream position *i*
is a pure function ``f(key, i)`` (SplitMix64 mixing of ``key + i``), so it is
identical no matter where chunk boundaries fall and exactly one value is
produced per position. (numpy's ``Philox.advance`` cannot be used for this:
it does not reconcile with ``Generator.random()``'s float64 buffering, so
seeking to an arbitrary double position is not possible via the public API.)

Contract / determinism notes:
- ``chunk_uniform(seed, s, n)`` returns the same values as the corresponding
  slice of ``chunk_uniform(seed, 0, s + n)`` — position invariance is the whole
  point; pinned by ``tests/utils/test_rng.py``.
- ``chunk_normal`` uses inverse-CDF (``ndtri``) so it consumes EXACTLY one
  uniform per output. Ziggurat/Box-Muller would consume a variable/!=1 number
  of draws per value and destroy position alignment.
- Calling a streaming strategy's scorer twice in one cycle with the same
  ``seed_components`` reproduces identical draws — unlike a stateful Generator
  that would advance. This is intentional and documented.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri

# Default seed used when the caller has no explicit ``random_state`` (mirrors
# the legacy per-strategy constructor defaults of 42 in RandomAcquisition /
# TopKAcquisition / ThompsonSamplingAcquisition) so seed components are never
# ``None`` and the hash key is always well-defined.
DEFAULT_RNG_SEED = 42

# Smallest representable double > 0 used to clamp uniforms away from 0.0 before
# ndtri (ndtri(0.0) = -inf, which would auto-select a compound under ascending
# acquisition). The mixer can produce 0; the half-open upper bound is < 1.
_UNIFORM_FLOOR = np.finfo(np.float64).tiny

# SplitMix64 constants (Steele, Lea & Flood 2014). All arithmetic is uint64,
# wrapping modulo 2**64.
_SM64_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SM64_M1 = np.uint64(0xBF58476D1CE4E5B9)
_SM64_M2 = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)
# 2**-53: scales a 53-bit integer into [0, 1) with full double precision.
_TWO_POW_NEG53 = np.float64(1.0 / (1 << 53))
_SHIFT11 = np.uint64(11)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised SplitMix64 finaliser over a uint64 array (wrapping arithmetic)."""
    with np.errstate(over='ignore'):
        z = x
        z = (z ^ (z >> _S30)) * _SM64_M1
        z = (z ^ (z >> _S27)) * _SM64_M2
        z = z ^ (z >> _S31)
    return np.asarray(z, dtype=np.uint64)


def _key_base(seed_components: tuple[int, ...]) -> np.uint64:
    """Fold ``seed_components`` into a single well-mixed uint64 key offset.

    ``None`` components are replaced by :data:`DEFAULT_RNG_SEED` so the key is
    always concrete. Each component is mixed in sequentially so distinct tuples
    (e.g. different cycles) yield independent streams.
    """
    acc = np.uint64(0)
    with np.errstate(over='ignore'):
        for c in seed_components:
            val = DEFAULT_RNG_SEED if c is None else int(c)
            mixed = _splitmix64(
                np.asarray(acc + (np.uint64(val & 0xFFFFFFFFFFFFFFFF) ^ _SM64_GAMMA))
            )
            acc = np.uint64(mixed)
    return np.uint64(acc)


def chunk_uniform(
    seed_components: tuple[int, ...], start: int, n: int
) -> np.ndarray:
    """Uniform ``[0, 1)`` draws for stream positions ``[start, start + n)``.

    Args:
        seed_components: Tuple identifying the stream (e.g. ``(random_state,
            cycle_idx)``). ``None`` entries fall back to :data:`DEFAULT_RNG_SEED`.
        start: Global stream offset of the first value in this chunk.
        n: Number of values to draw.

    Returns:
        ``float64`` array of length ``n``. Position invariant: the result equals
        ``chunk_uniform(seed_components, 0, start + n)[start:]``.
    """
    if n < 0:
        raise ValueError(f'n must be non-negative, got {n}')
    if start < 0:
        raise ValueError(f'start must be non-negative, got {start}')
    if n == 0:
        return np.empty(0, dtype=np.float64)
    key = _key_base(seed_components)
    indices = np.arange(start, start + n, dtype=np.uint64)
    with np.errstate(over='ignore'):
        mixed = _splitmix64(key + indices)
    # Top 53 bits → [0, 1). Standard high-quality uint64→double construction.
    result: np.ndarray = (mixed >> _SHIFT11).astype(np.float64) * _TWO_POW_NEG53
    return result


def chunk_normal(
    seed_components: tuple[int, ...], start: int, n: int
) -> np.ndarray:
    """Standard-normal draws for stream positions ``[start, start + n)``.

    Inverse-CDF transform of :func:`chunk_uniform` — exactly one uniform per
    output, so positions stay aligned across any chunking. Uniforms are clamped
    to ``[tiny, 1)`` so ``ndtri`` never returns ``-inf``.
    """
    u = chunk_uniform(seed_components, start, n)
    if n == 0:
        return u
    np.maximum(u, _UNIFORM_FLOOR, out=u)
    return ndtri(u)

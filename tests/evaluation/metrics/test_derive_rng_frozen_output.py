"""Spec 022 D2 regression test: ``_derive_rng`` frozen-output stability.

Numpy's ``default_rng([seed, cycle])`` seed-array semantics are stable since
numpy 1.25 per NEP-19. This test asserts the exact integers
``_derive_rng(seed, cycle).integers(0, 1_000_000, size=10)`` produces match
hardcoded reference arrays. If numpy ever changes ``default_rng``
implementation (NEP-19 forbids this without a major version bump, but defense
in depth) this test trips before reservoir contents silently shift.

Reference values captured 2026-05-10 with numpy 1.26.4 in the
``learnm8`` conda env. Per plan.md fallback option, ``numpy>=1.25`` (not
strictly ``>=2.0``) satisfies the NEP-19 determinism contract — the
captured values are stable across the 1.25–2.x range.
"""

from __future__ import annotations

import pytest

from learnm8.evaluation.metrics.similarity import _derive_rng

NEP19_FROZEN_SEED42_CYCLE0 = (
    89250, 773956, 654571, 438878, 433015,
    858597, 85945, 697368, 201469, 94177,
)
NEP19_FROZEN_SEED42_CYCLE1 = (
    637987, 793502, 449405, 245958, 920042,
    404426, 822724, 632755, 902618, 816194,
)
NEP19_FROZEN_SEED42_CYCLE5 = (
    261009, 174686, 597003, 978610, 706853,
    57004, 100141, 171094, 471883, 733027,
)
NEP19_FROZEN_SEEDNONE_CYCLE0 = (
    46116, 914268, 604490, 744101, 96510,
    824957, 181651, 202536, 708542, 643404,
)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("cycle", "expected"),
    [
        (0, NEP19_FROZEN_SEED42_CYCLE0),
        (1, NEP19_FROZEN_SEED42_CYCLE1),
        (5, NEP19_FROZEN_SEED42_CYCLE5),
    ],
)
def test_derive_rng_seed42_produces_frozen_sequence(
    cycle: int, expected: tuple[int, ...]
) -> None:
    """Sanity-check NEP-19 stability for the spec 022 reservoir RNG.

    If this fails: numpy ``default_rng([seed, cycle])`` output drifted, which
    means reservoir contents are no longer reproducible across numpy versions.
    Investigate before relying on cross-version reservoir determinism.
    """
    rng = _derive_rng(global_seed=42, cycle=cycle)
    actual = tuple(int(x) for x in rng.integers(0, 1_000_000, size=10))
    assert actual == expected, (
        f"_derive_rng(42, {cycle}) shifted output. Numpy version may have "
        f"violated NEP-19 stability promise. got={actual}"
    )


@pytest.mark.slow
def test_derive_rng_seed_none_produces_frozen_sequence() -> None:
    """Sanity-check the no-seed fallback path (uses constant 0xA17C7E offset).

    The seed=None path is used when callers do not pass ``random_state``;
    determinism within a run is the contract (cross-run reproducibility is
    not promised, but the offset is fixed so within-process behaviour is).
    """
    rng = _derive_rng(global_seed=None, cycle=0)
    actual = tuple(int(x) for x in rng.integers(0, 1_000_000, size=10))
    assert actual == NEP19_FROZEN_SEEDNONE_CYCLE0, (
        f"_derive_rng(None, 0) shifted output. The 0xA17C7E offset path "
        f"may have changed. got={actual}"
    )


@pytest.mark.slow
def test_derive_rng_two_calls_same_seed_match() -> None:
    """Two separate _derive_rng(42, 0) calls produce bit-identical sequences."""
    rng_a = _derive_rng(global_seed=42, cycle=0)
    rng_b = _derive_rng(global_seed=42, cycle=0)
    a = tuple(int(x) for x in rng_a.integers(0, 1_000_000, size=20))
    b = tuple(int(x) for x in rng_b.integers(0, 1_000_000, size=20))
    assert a == b, "Two _derive_rng(42, 0) calls produced different sequences"

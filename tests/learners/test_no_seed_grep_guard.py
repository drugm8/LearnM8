"""Spec 021 / US1 / acceptance checklist — grep guard.

Walks ``learnm8/`` and asserts zero occurrences of ``np.random.seed(`` outside
explicitly allowed locations. This guard prevents regressions in any
contributor PR that would re-introduce process-global numpy RNG mutation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PATTERN = re.compile(r"np\.random\.seed\(")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LEARNM8_DIR = _PROJECT_ROOT / "learnm8"


def test_no_global_np_random_seed_calls_in_learnm8():
    offenders: list[tuple[Path, int, str]] = []
    for py_file in _LEARNM8_DIR.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if _PATTERN.search(line):
                offenders.append((py_file.relative_to(_PROJECT_ROOT), line_no, line.strip()))

    assert not offenders, (
        "Found np.random.seed(...) calls in learnm8/. Use np.random.default_rng(seed) "
        "and a local Generator instead.\nOffending lines:\n"
        + "\n".join(f"  {p}:{ln}: {src}" for p, ln, src in offenders)
    )

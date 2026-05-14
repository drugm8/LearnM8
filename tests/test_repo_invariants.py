"""Repo-invariant CI guard for stale renamed identifiers (feature 019, Should-Fix #11).

Replaces the original shell ``grep`` quickstart with a portable, pytest-native
scan over ``learnm8/``, ``tests/``, and ``docs/``. Excludes specs/contracts/
CHANGELOG that intentionally describe the rename.
"""

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS: tuple[str, ...] = ("learnm8", "tests", "docs")
SCAN_GLOBS: tuple[str, ...] = ("**/*.py", "**/*.md")

EXEMPTIONS: frozenset[str] = frozenset({
    "CHANGELOG.md",
    "tests/test_repo_invariants.py",
    # Rename-test file intentionally references the deprecated identifiers.
    "tests/evaluation/test_bit_position_uniformity_entropy_rename.py",
    # The renamed-symbol __getattr__ hint dict in similarity.py contains the old
    # names by design (so users get a helpful error, not a silent KeyError).
    "learnm8/evaluation/metrics/similarity.py",
    "specs/019-math-correctness/spec.md",
    "specs/019-math-correctness/plan.md",
    "specs/019-math-correctness/data-model.md",
    "specs/019-math-correctness/quickstart.md",
    "specs/019-math-correctness/review-report.md",
    "specs/019-math-correctness/tasks.md",
    "specs/019-math-correctness/contracts/bit_marginal_entropy_rename.md",
    "specs/019-math-correctness/contracts/score_improvement_ratio.md",
    "specs/019-math-correctness/contracts/repo_invariants.md",
    "specs/019-math-correctness/contracts/sigma_clamp_helper.md",
    "specs/019-math-correctness/contracts/nan_prediction_handling.md",
    "specs/019-math-correctness/contracts/ei_pi_sigma_zero.md",
    "specs/019-math-correctness/contracts/entropy_acquisition.md",
    "specs/019-math-correctness/contracts/prediction_entropy.md",
    "specs/019-math-correctness/contracts/stable_pruning_sort.md",
    "specs/019-math-correctness/contracts/tanimoto_kernel_diag.md",
    "specs/019-math-correctness/research/research.md",
    "specs/019-math-correctness/research/landscape.md",
    "specs/019-math-correctness/research/codebase.md",
    "specs/019-math-correctness/research/pitfalls.md",
    "specs/019-math-correctness/research/deep-docs.md",
    "specs/019-math-correctness/research/structure-overview.md",
})


def _scan_for_substring(needle: str) -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []
    for d in SCAN_DIRS:
        base = REPO_ROOT / d
        if not base.exists():
            continue
        for pattern in SCAN_GLOBS:
            for path in base.rglob(pattern):
                rel = path.relative_to(REPO_ROOT).as_posix()
                if rel in EXEMPTIONS:
                    continue
                try:
                    with path.open(encoding="utf-8") as f:
                        for i, line in enumerate(f, start=1):
                            if needle in line:
                                hits.append((path, i, line.rstrip()))
                except (OSError, UnicodeDecodeError):
                    continue
    return hits


@pytest.mark.parametrize(
    "needle",
    [
        "shannon_entropy_diversity",
        "_shannon_entropy_from_bit_sum",
        "calculate_average_score_ratio",
        "calculate_batch_average_score_ratio",
        "cumulative_avg_score_ratio",
        "batch_avg_score_ratio",
    ],
)
def test_no_stale_renamed_references(needle: str) -> None:
    hits = _scan_for_substring(needle)
    assert not hits, (
        f"Found {len(hits)} stale references to {needle!r} after feature 019 rename. "
        f"Update each to the new name (see specs/019-math-correctness/contracts/). "
        f"First 5 hits:\n"
        + "\n".join(
            f"  {p.relative_to(REPO_ROOT)}:{ln}: {text}"
            for p, ln, text in hits[:5]
        )
    )

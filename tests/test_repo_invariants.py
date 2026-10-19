"""Repo-invariant CI guard for stale renamed identifiers.

A portable, pytest-native scan over ``learnm8/``, ``tests/``, and ``docs/``.
Excludes the CHANGELOG and the files that intentionally describe the rename.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS: tuple[str, ...] = ('learnm8', 'tests', 'docs')
SCAN_GLOBS: tuple[str, ...] = ('**/*.py', '**/*.md')

EXEMPTIONS: frozenset[str] = frozenset(
    {
        'CHANGELOG.md',
        'tests/test_repo_invariants.py',
        # Rename-test file intentionally references the deprecated identifiers.
        'tests/evaluation/test_bit_position_uniformity_entropy_rename.py',
        # The renamed-symbol __getattr__ hint dict in similarity.py contains the old
        # names by design (so users get a helpful error, not a silent KeyError).
        'learnm8/evaluation/metrics/similarity.py',
    }
)


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
                    with path.open(encoding='utf-8') as f:
                        for i, line in enumerate(f, start=1):
                            if needle in line:
                                hits.append((path, i, line.rstrip()))
                except (OSError, UnicodeDecodeError):
                    continue
    return hits


@pytest.mark.parametrize(
    'needle',
    [
        'shannon_entropy_diversity',
        '_shannon_entropy_from_bit_sum',
        'calculate_average_score_ratio',
        'calculate_batch_average_score_ratio',
        'cumulative_avg_score_ratio',
        'batch_avg_score_ratio',
    ],
)
def test_no_stale_renamed_references(needle: str) -> None:
    hits = _scan_for_substring(needle)
    assert not hits, (
        f'Found {len(hits)} stale references to {needle!r}, which was renamed. '
        f'Update each to the new name. '
        f'First 5 hits:\n'
        + '\n'.join(
            f'  {p.relative_to(REPO_ROOT)}:{ln}: {text}' for p, ln, text in hits[:5]
        )
    )

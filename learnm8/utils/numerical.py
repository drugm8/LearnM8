"""Numerical-safety primitives shared by acquisition functions and the cycle loop.

Introduced in feature 019 (Mathematical Correctness — Silent Science Bugs) as the
single source of truth for σ-clamping and NaN/Inf assertion. Three call sites
need ``clamp_sigma`` (EI, PI, EntropyAcquisition) and one needs ``assert_no_nan``
(``learnm8/core/cycle.py``); centralising here prevents the three-place
duplication anti-pattern.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

import numpy as np

from learnm8.exceptions import LearnerError

SIGMA_FLOOR: Final[float] = 1e-9

# `IdSource` accepts any indexable Sequence (list, np.ndarray, tuple) OR a lazy
# Iterable. The error path materialises to a list only when there's a hit, so
# the clean path pays no Python-list cost on million-compound runs.
IdSource = Sequence[str] | Iterable[str]


def clamp_sigma(sigma: np.ndarray) -> np.ndarray:
    """Promote σ to float64 (zero-copy if already float64) and clamp at :data:`SIGMA_FLOOR`.

    Returns a new array; never modifies the input. The float64 promotion is
    intentional: ``SIGMA_FLOOR`` (1e-9) is below float32 eps (1.19e-7), so a raw
    float32 input would silently be a no-op.
    """
    return np.maximum(np.asarray(sigma, dtype=np.float64), SIGMA_FLOOR)


def _format_id_list(ids: IdSource, indices: np.ndarray) -> str:
    if not isinstance(ids, Sequence):
        ids = list(ids)
    bad_ids = [ids[i] for i in indices[:10]]
    return "[" + ", ".join(repr(i) for i in bad_ids) + "]"


def assert_no_nan(arr: np.ndarray, ids: IdSource, name: str) -> None:
    """Raise :class:`LearnerError` if ``arr`` contains any NaN.

    Message format: ``"NaN <name> detected: <N> row[s], first IDs: [<id1>, ...]"``
    using ``repr()`` on each ID (robust to embedded quotes), pluralising
    "row"/"rows" correctly. No-op if ``arr`` is NaN-free.
    """
    indices = np.flatnonzero(np.isnan(arr))
    n = int(indices.size)
    if n == 0:
        return
    ids_str = _format_id_list(ids, indices)
    row_word = "row" if n == 1 else "rows"
    raise LearnerError(f"NaN {name} detected: {n} {row_word}, first IDs: {ids_str}")


def assert_no_inf_uncertainty(uncertainties: np.ndarray, ids: IdSource) -> None:
    """Raise :class:`LearnerError` if ``uncertainties`` contains any ±inf entry.

    Spec edge case: legitimate ``inf`` σ (a learner explicitly signalling
    unbounded uncertainty) must NOT be silently clamped to a finite value.
    """
    indices = np.flatnonzero(np.isinf(uncertainties))
    n = int(indices.size)
    if n == 0:
        return
    ids_str = _format_id_list(ids, indices)
    row_word = "row" if n == 1 else "rows"
    raise LearnerError(
        f"Inf uncertainties not supported: {n} {row_word}, first IDs: {ids_str}"
    )

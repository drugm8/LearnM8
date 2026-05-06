"""Diversity metrics for active-learning evaluation.

This module computes diversity metrics — the inverse of pairwise similarity —
across the compounds selected during an active-learning campaign. The legacy
O(n²) similarity functions (intra_batch_diversity / inter_cycle_similarity /
batch_novelty_score) have been replaced with three rigorously-defined,
efficiently-computed primitives, each emitted twice per cycle (once for the
newly-selected batch, once for the cumulative selected set):

    * mean_tanimoto_similarity_sampled  - mean Tanimoto from adaptive
      pair sampling (pilot-then-scale, target 95% CI half-width 0.005)
    * scaffold_diversity_index          - unique Bemis-Murcko scaffolds /
      valid compounds
    * shannon_entropy_diversity         - base-2 entropy of fingerprint
      bit-activation frequencies (online accumulator)

All metrics operate ONLY on the selected compounds (bounded by n_cycles ×
batch_size), never on the full pool — this is the key to scaling the metric
path to 100M-compound libraries.

The filename ``similarity.py`` is retained for git-history continuity even
though the module now computes diversity (the inverse of similarity).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import polars as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from scipy.stats import entropy as scipy_entropy

if TYPE_CHECKING:
    from learnm8.features.base import SkfpFeaturizer

logger = logging.getLogger(__name__)


# ============================================================================
# Public API constants
# ============================================================================

DIVERSITY_KEYS: tuple[str, ...] = (
    "mean_tanimoto_similarity_sampled_batch",
    "mean_tanimoto_similarity_sampled_cumulative",
    "scaffold_diversity_index_batch",
    "scaffold_diversity_index_cumulative",
    "shannon_entropy_diversity_batch",
    "shannon_entropy_diversity_cumulative",
)

# ============================================================================
# Module-level constants (FR-025)
# ============================================================================

N_PILOT_PAIRS: int = 500          # pilot draw for σ̂ estimation
TARGET_EPSILON: float = 0.005     # 95% CI half-width target
N_PAIRS_MIN: int = 1_000          # adaptive lower clamp
N_PAIRS_MAX: int = 100_000        # adaptive upper clamp (perf safety)
Z_95: float = 1.96                # 95% normal critical value
FALLBACK_FP_RADIUS: int = 2
FALLBACK_FP_SIZE: int = 2048


# ============================================================================
# RunCache dataclass (post-review revision)
# ============================================================================


@dataclass
class RunCache:
    """Per-run diversity-metric caches.

    Constructed once in ``run_active_learning`` and passed explicitly through
    cycle execution to ``compute_diversity_metrics``. Replaces the previous
    module-level cache + ``weakref.finalize`` design after multi-perspective
    review identified two failure modes: (a) Polars ``with_columns()`` returns
    new DataFrames each cycle so ``id(df)`` keying breaks mid-run, and
    (b) ``weakref.finalize`` fires at GC time which is non-deterministic when
    the results dict retains references.

    Lifetime: caller-owned. ``run_active_learning`` calls ``clear()`` in a
    ``finally`` block to release references promptly. Concurrent runs each
    own their own RunCache → process-safe by isolation; no internal locks.
    """

    fp_cache: dict[str, ExplicitBitVect] = field(default_factory=dict)
    scaffold_cache: dict[str, str] = field(default_factory=dict)
    cumulative_fp_buffer: list[ExplicitBitVect] = field(default_factory=list)
    cumulative_smiles_seen: set[str] = field(default_factory=set)
    bit_sum_buffer: np.ndarray | None = None

    def clear(self) -> None:
        """Manual eviction. Called from run_active_learning's finally block."""
        self.fp_cache.clear()
        self.scaffold_cache.clear()
        self.cumulative_fp_buffer.clear()
        self.cumulative_smiles_seen.clear()
        self.bit_sum_buffer = None


# ============================================================================
# Private helpers
# ============================================================================


def _derive_rng(global_seed: int | None, cycle: int) -> np.random.Generator:
    """Derive a per-cycle deterministic RNG from (global_seed, cycle).

    Per FR-008: re-running with the same ``random_state`` produces identical
    pair samples for the same cycle. When ``global_seed`` is None (FR-009),
    we fall back to a constant offset so within-run sampling is still stable
    across cycles even though cross-run reproducibility is not guaranteed.
    """
    if global_seed is None:
        return np.random.default_rng([0xA17C7E, cycle])
    return np.random.default_rng([int(global_seed), int(cycle)])


def _sample_distinct_pairs(
    n: int,
    n_pairs: int,
    rng: np.random.Generator,
    excluded: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Draw ``n_pairs`` distinct (i, j) pairs with i < j, avoiding ``excluded``.

    Uses rejection sampling on independent uniform index draws. Cheap when
    n_pairs << N(N-1)/2 (the regime we operate in by design).
    """
    if n < 2:
        return []
    excluded = excluded or set()
    seen: set[tuple[int, int]] = set(excluded)
    pairs: list[tuple[int, int]] = []
    target = len(excluded) + n_pairs
    # Conservative max attempts; for our adaptive bounds this terminates fast.
    max_attempts = max(target * 8, 1_000_000)
    attempts = 0
    while len(pairs) < n_pairs and attempts < max_attempts:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            attempts += 1
            continue
        if i > j:
            i, j = j, i
        key = (i, j)
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        pairs.append(key)
        attempts += 1
    return pairs


def _bulk_tanimoto_for_pairs(
    fps: list[ExplicitBitVect],
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    """Compute Tanimoto for an explicit list of (i, j) pairs.

    Uses ``rdkit.DataStructs.BulkTanimotoSimilarity`` per query fingerprint
    (the queries here are deduplicated by the left index so we issue the
    minimum number of bulk calls).
    """
    if not pairs:
        return np.empty(0, dtype=np.float64)
    by_left: dict[int, list[tuple[int, int]]] = {}
    for idx, (i, j) in enumerate(pairs):
        by_left.setdefault(i, []).append((idx, j))
    out = np.empty(len(pairs), dtype=np.float64)
    for i, entries in by_left.items():
        right_fps = [fps[j] for _, j in entries]
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], right_fps)
        for (idx, _), sim in zip(entries, sims):
            out[idx] = sim
    return out


def _mean_tanimoto_adaptive_sampled(
    fps: list[ExplicitBitVect],
    rng: np.random.Generator,
) -> float | None:
    """Adaptive mean Tanimoto via pilot-then-scale sampling (FR-001).

    1. Pilot: draw ``N_PILOT_PAIRS`` distinct pairs and compute σ̂.
    2. Size: ``n_pairs = ceil(Z_95² · σ̂² / TARGET_EPSILON²)``,
       clamped to ``[N_PAIRS_MIN, N_PAIRS_MAX]``.
    3. Exhaustive shortcut: if the total pair space ``N(N-1)/2`` ≤ n_pairs,
       compute the exact mean over all pairs.
    4. Top-up: draw the remainder distinct from the pilot's pairs.
    5. Return the mean over all ``n_pairs`` (pilot is not discarded).

    The same RNG instance threads both the pilot and top-up draws so two
    seeded runs produce bit-identical samples.

    Returns ``None`` when fewer than 2 fingerprints are supplied.
    """
    n = len(fps)
    if n < 2:
        return None

    total_pairs = (n * (n - 1)) // 2

    # Tiny set: exhaustive over all pairs.
    if total_pairs <= max(N_PAIRS_MIN, N_PILOT_PAIRS):
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        sims = _bulk_tanimoto_for_pairs(fps, all_pairs)
        return float(sims.mean()) if sims.size > 0 else None

    # Pilot draw.
    pilot = _sample_distinct_pairs(n, N_PILOT_PAIRS, rng)
    if len(pilot) < 2:
        return None
    pilot_sims = _bulk_tanimoto_for_pairs(fps, pilot)
    sigma_hat = float(pilot_sims.std(ddof=1)) if pilot_sims.size > 1 else 0.0

    # Adaptive sizing, clamped.
    if sigma_hat <= 0.0:
        n_pairs = N_PAIRS_MIN
    else:
        raw = int(np.ceil((Z_95 ** 2) * (sigma_hat ** 2) / (TARGET_EPSILON ** 2)))
        n_pairs = max(N_PAIRS_MIN, min(raw, N_PAIRS_MAX))

    if n_pairs <= len(pilot):
        return float(pilot_sims.mean())

    # Exhaustive shortcut at the scaled bound.
    if total_pairs <= n_pairs:
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        sims = _bulk_tanimoto_for_pairs(fps, all_pairs)
        return float(sims.mean()) if sims.size > 0 else None

    # Top-up draw, excluding the pilot's pairs.
    excluded = set(pilot)
    extra_n = n_pairs - len(pilot)
    extra = _sample_distinct_pairs(n, extra_n, rng, excluded=excluded)
    extra_sims = _bulk_tanimoto_for_pairs(fps, extra)
    combined = np.concatenate([pilot_sims, extra_sims])
    if combined.size == 0:
        return None
    return float(combined.mean())


def _scaffold_for_smiles(smi: str) -> str | None:
    """Compute the canonical Bemis-Murcko scaffold SMILES for one SMILES.

    Uses ``LargestFragmentChooser`` to handle salts / multi-component SMILES,
    then ``MurckoScaffold.MurckoScaffoldSmiles`` for the scaffold string.
    Acyclic compounds canonically yield the empty string ``""``. Returns
    ``None`` on RDKit parse / scaffold failure (caller drops these from
    both numerator and denominator per FR-002).
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        chooser = rdMolStandardize.LargestFragmentChooser()
        largest = chooser.choose(mol)
        if largest is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=largest, includeChirality=False)
    except Exception:  # pragma: no cover - defensive
        return None


def _scaffold_diversity_index(
    smiles_list: list[str],
    scaffold_cache: dict[str, str],
) -> float | None:
    """Bemis-Murcko unique-scaffold ratio (FR-002).

    Returns ``len(unique_scaffolds) / valid_compound_count``. Compounds
    whose scaffold cannot be computed are excluded from BOTH numerator
    and denominator. ``None`` if no valid compound remains.

    Cache contract: every successfully-computed scaffold is memoised in
    ``scaffold_cache`` keyed by SMILES so subsequent cycles re-encountering
    the same compound skip RDKit work.
    """
    if not smiles_list:
        return None
    unique: set[str] = set()
    valid_count = 0
    for smi in smiles_list:
        if smi in scaffold_cache:
            scaffold = scaffold_cache[smi]
        else:
            scaffold = _scaffold_for_smiles(smi)
            if scaffold is None:
                continue
            scaffold_cache[smi] = scaffold
        unique.add(scaffold)
        valid_count += 1
    if valid_count == 0:
        return None
    return float(len(unique) / valid_count)


def _shannon_entropy_from_bit_sum(
    bit_sum: np.ndarray | None,
    n_compounds: int,
) -> float | None:
    """Shannon entropy of the per-bit activation-frequency distribution (FR-003).

    Operates on an online accumulator: ``bit_sum[k]`` is the count of
    compounds with bit k = 1 across the cumulative set; ``n_compounds`` is
    the total compound count. ``bit_frequencies = bit_sum / n_compounds``;
    ``scipy.stats.entropy`` then auto-normalises by the sum so the result
    is the entropy of the normalised bit-position distribution (upper bound
    ``log₂(n_bits)``).

    Returns ``None`` if no compounds are supplied or all bit-frequencies are
    zero.
    """
    if n_compounds <= 0 or bit_sum is None:
        return None
    bit_frequencies = bit_sum.astype(np.float64) / float(n_compounds)
    if not np.any(bit_frequencies > 0):
        return None
    try:
        value = float(scipy_entropy(bit_frequencies, base=2))
    except (ValueError, ZeroDivisionError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _morgan_fp_for_smiles(
    smi: str,
    morgan_gen: rdFingerprintGenerator.FingerprintGenerator64,
) -> ExplicitBitVect | None:
    """Build a Morgan(2, 2048) ExplicitBitVect for a single SMILES."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return morgan_gen.GetFingerprint(mol)
    except Exception:  # pragma: no cover - defensive
        return None


def _fp_to_packed(fp: ExplicitBitVect, n_bits: int) -> np.ndarray:
    """Convert an ExplicitBitVect to a 1D uint8 array of {0,1} values.

    Sized at ``n_bits`` (the fingerprint length). Used for online Shannon
    accumulator updates.
    """
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _get_or_build_fps(
    smiles_list: list[str],
    featurizer: "SkfpFeaturizer | None",
    cache_dir: Path | None,
    fp_cache: dict[str, ExplicitBitVect],
) -> tuple[list[ExplicitBitVect], np.ndarray, str, list[int]]:
    """Resolve fingerprints for a SMILES list, choosing the best source.

    Returns:
        rdkit_fps: list of ExplicitBitVect (one per VALID SMILES, length
            equals ``len(valid_idx)``).
        packed_matrix: ``(len(valid_idx), n_bits)`` uint8 array with bit
            values; the caller uses this to update the online Shannon
            accumulator.
        fingerprint_used_label: provenance string written to the
            ``fingerprint_used`` column of cycle_metrics.csv. Either
            ``"morgan_2_2048"`` (Morgan fall-back path; we always use the
            local Morgan generator for the pair-sample work even when the
            run featurizer is binary, since the featurizer's HDF5-cached
            ndarray is a less convenient input than ExplicitBitVect for
            BulkTanimotoSimilarity. The label distinguishes whether the
            run's featurizer was already binary (reuse-eligible label)
            or non-binary (fallback label).
        valid_idx: indices into ``smiles_list`` whose fingerprint was
            successfully built (used by the caller to filter parallel
            arrays for downstream computation).

    Caching: ``fp_cache`` is mutated to memoise each canonical SMILES'
    fingerprint so repeated cycles never re-hash a previously-seen compound.
    """
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=FALLBACK_FP_RADIUS, fpSize=FALLBACK_FP_SIZE
    )
    is_binary_featurizer = bool(
        featurizer is not None and getattr(featurizer, "feature_type", None) == "binary"
    )
    label_base = "morgan_2_2048"
    if featurizer is not None and is_binary_featurizer:
        try:
            fp_name = featurizer.get_name()
        except Exception:  # pragma: no cover - defensive
            fp_name = "morgan"
        fingerprint_used_label = f"{fp_name}_2_2048" if fp_name == "morgan" else f"{fp_name}"
    elif featurizer is not None and not is_binary_featurizer:
        fingerprint_used_label = f"{label_base}_fallback"
    else:
        fingerprint_used_label = label_base

    rdkit_fps: list[ExplicitBitVect] = []
    valid_idx: list[int] = []
    n_bits = FALLBACK_FP_SIZE
    packed_rows: list[np.ndarray] = []
    skipped = 0
    for k, smi in enumerate(smiles_list):
        fp = fp_cache.get(smi)
        if fp is None:
            fp = _morgan_fp_for_smiles(smi, morgan_gen)
            if fp is None:
                skipped += 1
                continue
            fp_cache[smi] = fp
        rdkit_fps.append(fp)
        valid_idx.append(k)
        packed_rows.append(_fp_to_packed(fp, n_bits))
    if skipped:
        logger.warning(
            "diversity: skipped %d invalid SMILES while building fingerprints",
            skipped,
        )
    if packed_rows:
        packed_matrix = np.vstack(packed_rows).astype(np.uint8)
    else:
        packed_matrix = np.zeros((0, n_bits), dtype=np.uint8)
    return rdkit_fps, packed_matrix, fingerprint_used_label, valid_idx


# ============================================================================
# Public function
# ============================================================================


def _normalise_disable(
    disable: bool | Iterable[str],
) -> tuple[bool, frozenset[str]]:
    """Resolve the ``disable`` arg into (all_disabled, named_disabled_set).

    - True             → all six disabled
    - False            → none disabled
    - Iterable[str]    → only the named keys disabled (must be in DIVERSITY_KEYS)
    """
    if disable is True:
        return True, frozenset(DIVERSITY_KEYS)
    if disable is False:
        return False, frozenset()
    named = frozenset(disable)
    unknown = named - frozenset(DIVERSITY_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown DIVERSITY_KEYS in disable: {sorted(unknown)}. "
            f"Valid keys: {DIVERSITY_KEYS}"
        )
    return (named == frozenset(DIVERSITY_KEYS)), named


def compute_diversity_metrics(
    *,
    newly_selected_df: pl.DataFrame,
    cumulative_selected_df: pl.DataFrame,
    cycle: int,
    run_cache: RunCache,
    random_state: int | None = None,
    featurizer: "SkfpFeaturizer | None" = None,
    cache_dir: Path | None = None,
    disable: bool | Iterable[str] = False,
) -> dict[str, float | str | None]:
    """Compute the six per-cycle diversity metrics.

    See ``contracts/diversity_metrics.md`` for the full behavioural contract.
    The returned dict always contains exactly the six DIVERSITY_KEYS plus a
    ``fingerprint_used`` provenance string. Values are Python ``float`` (never
    numpy scalars) or ``None`` on disable / failure.
    """
    if cycle < 0:
        raise ValueError(f"cycle must be >= 0, got {cycle}")
    if "SMILES" not in newly_selected_df.columns:
        raise ValueError("newly_selected_df must contain a 'SMILES' column")
    if "SMILES" not in cumulative_selected_df.columns:
        raise ValueError("cumulative_selected_df must contain a 'SMILES' column")

    all_disabled, disabled_set = _normalise_disable(disable)

    metrics: dict[str, float | str | None] = {key: None for key in DIVERSITY_KEYS}
    metrics["fingerprint_used"] = None

    if all_disabled:
        return metrics

    rng = _derive_rng(random_state, cycle)
    batch_smiles: list[str] = newly_selected_df.get_column("SMILES").to_list()
    cumulative_smiles: list[str] = cumulative_selected_df.get_column("SMILES").to_list()

    # ----- Batch fingerprints (always rebuilt; small) -----
    batch_start = time.perf_counter()
    batch_fps, batch_packed, fp_label_batch, _batch_valid_idx = _get_or_build_fps(
        batch_smiles, featurizer, cache_dir, run_cache.fp_cache
    )
    batch_ms = (time.perf_counter() - batch_start) * 1000.0

    # ----- Cumulative fingerprints (append-only buffer) -----
    cum_start = time.perf_counter()
    new_cumulative_smiles = [
        s for s in cumulative_smiles if s not in run_cache.cumulative_smiles_seen
    ]
    new_fps, new_packed, fp_label_cum, _cum_valid_idx = _get_or_build_fps(
        new_cumulative_smiles, featurizer, cache_dir, run_cache.fp_cache
    )
    run_cache.cumulative_fp_buffer.extend(new_fps)
    for s in new_cumulative_smiles:
        run_cache.cumulative_smiles_seen.add(s)

    n_bits = FALLBACK_FP_SIZE
    if run_cache.bit_sum_buffer is None:
        run_cache.bit_sum_buffer = np.zeros(n_bits, dtype=np.int64)
    if new_packed.shape[0] > 0:
        run_cache.bit_sum_buffer += new_packed.sum(axis=0).astype(np.int64)
    cumulative_fps = run_cache.cumulative_fp_buffer
    cum_ms = (time.perf_counter() - cum_start) * 1000.0

    fingerprint_used_label = fp_label_batch or fp_label_cum
    metrics["fingerprint_used"] = fingerprint_used_label

    # ----- Mean Tanimoto (adaptive) -----
    try:
        if "mean_tanimoto_similarity_sampled_batch" not in disabled_set:
            metrics["mean_tanimoto_similarity_sampled_batch"] = (
                _mean_tanimoto_adaptive_sampled(batch_fps, rng)
            )
        if "mean_tanimoto_similarity_sampled_cumulative" not in disabled_set:
            metrics["mean_tanimoto_similarity_sampled_cumulative"] = (
                _mean_tanimoto_adaptive_sampled(cumulative_fps, rng)
            )
    except Exception as exc:
        logger.warning(
            "diversity cycle=%d Tanimoto failed (%s); setting to None",
            cycle, type(exc).__name__,
        )
        metrics["mean_tanimoto_similarity_sampled_batch"] = None
        metrics["mean_tanimoto_similarity_sampled_cumulative"] = None

    # ----- Scaffold diversity -----
    scaffold_start = time.perf_counter()
    try:
        if "scaffold_diversity_index_batch" not in disabled_set:
            metrics["scaffold_diversity_index_batch"] = _scaffold_diversity_index(
                batch_smiles, run_cache.scaffold_cache
            )
        if "scaffold_diversity_index_cumulative" not in disabled_set:
            metrics["scaffold_diversity_index_cumulative"] = _scaffold_diversity_index(
                cumulative_smiles, run_cache.scaffold_cache
            )
    except Exception as exc:
        logger.warning(
            "diversity cycle=%d scaffold failed (%s); setting to None",
            cycle, type(exc).__name__,
        )
        metrics["scaffold_diversity_index_batch"] = None
        metrics["scaffold_diversity_index_cumulative"] = None
    scaffold_ms = (time.perf_counter() - scaffold_start) * 1000.0

    # ----- Shannon entropy (online accumulator + per-batch direct sum) -----
    shannon_start = time.perf_counter()
    try:
        if "shannon_entropy_diversity_batch" not in disabled_set:
            if batch_packed.shape[0] > 0:
                batch_bit_sum = batch_packed.sum(axis=0).astype(np.int64)
                metrics["shannon_entropy_diversity_batch"] = _shannon_entropy_from_bit_sum(
                    batch_bit_sum, batch_packed.shape[0]
                )
            else:
                metrics["shannon_entropy_diversity_batch"] = None
        if "shannon_entropy_diversity_cumulative" not in disabled_set:
            metrics["shannon_entropy_diversity_cumulative"] = _shannon_entropy_from_bit_sum(
                run_cache.bit_sum_buffer, len(run_cache.cumulative_fp_buffer)
            )
    except Exception as exc:
        logger.warning(
            "diversity cycle=%d Shannon failed (%s); setting to None",
            cycle, type(exc).__name__,
        )
        metrics["shannon_entropy_diversity_batch"] = None
        metrics["shannon_entropy_diversity_cumulative"] = None
    shannon_ms = (time.perf_counter() - shannon_start) * 1000.0

    logger.debug(
        "diversity cycle=%d batch_ms=%.1f cum_ms=%.1f scaffold_ms=%.1f shannon_ms=%.1f",
        cycle, batch_ms, cum_ms, scaffold_ms, shannon_ms,
    )

    # Coerce numpy scalars to Python floats per FR-013.
    for key in DIVERSITY_KEYS:
        v = metrics[key]
        if v is not None and not isinstance(v, float):
            try:
                metrics[key] = float(v)
            except (TypeError, ValueError):
                metrics[key] = None

    return metrics


__all__ = [
    "DIVERSITY_KEYS",
    "RunCache",
    "compute_diversity_metrics",
    "N_PILOT_PAIRS",
    "TARGET_EPSILON",
    "N_PAIRS_MIN",
    "N_PAIRS_MAX",
    "Z_95",
]

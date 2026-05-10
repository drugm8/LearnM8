"""Enrichment metrics for virtual screening evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from learnm8.utils.logging import get_logger, log_warning

ScoreDirection = Literal["higher", "lower"]


def calculate_enrichment_factor(scores: np.ndarray, labels: np.ndarray,
                               percentile: float, score_direction: str = 'higher') -> float:
    """
    Calculate Enrichment Factor for virtual screening.

    EF = (n_actives_selected / n_selected) / (n_actives_total / n_total)

    Args:
        scores: Screening scores
        labels: Binary labels (0/1)
        percentile: Percentile threshold for selection (e.g., 1.0 for top 1%)
        score_direction: 'higher' or 'lower' for score interpretation

    Returns:
        Enrichment factor value
    """
    if not 0 < percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    # Sort by scores according to direction
    if score_direction == 'lower':
        sorted_indices = np.argsort(scores)  # Ascending: lowest scores first
    else:  # score_direction == 'higher'
        sorted_indices = np.argsort(scores)[::-1]  # Descending: highest scores first
    sorted_labels = labels[sorted_indices]

    # Calculate selection size
    n_total = len(labels)
    n_selected = max(1, int(n_total * percentile / 100))

    # Count actives
    n_actives_total = np.sum(labels)
    n_actives_selected = np.sum(sorted_labels[:n_selected])

    # Avoid division by zero
    if n_actives_total == 0 or n_selected == 0:
        return 0.0

    # Calculate enrichment factor
    ef = (n_total * n_actives_selected) / (n_actives_total * n_selected)

    return round(ef, 3)


def calculate_top_k_overlap(predictions_df: pl.DataFrame, ground_truth_df: pl.DataFrame,
                           k: int, target_col: str, score_direction: str = 'higher') -> float:
    """
    Calculate percentage overlap between top-k predicted and true compounds.

    Args:
        predictions_df: DataFrame with 'ID' and 'prediction' columns
        ground_truth_df: DataFrame with 'ID' and target_col
        k: Number of top compounds to compare
        target_col: Column name in ground truth
        score_direction: 'higher' or 'lower' for score interpretation

    Returns:
        Percentage overlap (0-100)
    """
    # Merge predictions with ground truth
    merged = predictions_df.join(
        ground_truth_df.select(['ID', target_col]),
        on='ID',
        how='inner'
    )

    # Handle empty data case
    if len(merged) == 0:
        return 0.0

    if len(merged) < k:
        logger = get_logger()
        log_warning(logger, f"Only {len(merged)} compounds available, using all for top-k calculation")
        k = len(merged)

    # Ensure prediction column is numeric
    if merged.get_column('prediction').dtype != pl.Float64 and merged.get_column('prediction').dtype != pl.Float32:
        merged = merged.with_columns(
            pl.col('prediction').cast(pl.Float64, strict=False)
        )
        merged = merged.filter(pl.col('prediction').is_not_null())
        if len(merged) == 0:
            return 0.0

    # Sort by scores
    descending = (score_direction == 'higher')

    # Get top k by predictions
    top_k_predicted = set(
        merged.sort('prediction', descending=descending)
        .head(k)
        .get_column('ID')
        .to_list()
    )

    # Get top k by ground truth
    top_k_true = set(
        merged.sort(target_col, descending=descending)
        .head(k)
        .get_column('ID')
        .to_list()
    )

    # Calculate overlap
    overlap_count = len(top_k_predicted & top_k_true)
    overlap_percentage = (overlap_count / k) * 100

    return round(overlap_percentage, 2)


def calculate_multiple_top_k_overlaps(predictions_df: pl.DataFrame, ground_truth_df: pl.DataFrame,
                                    target_col: str, score_direction: str = 'higher') -> dict:
    """
    Calculate multiple top-K overlaps for different K values.

    Args:
        predictions_df: DataFrame with 'ID' and 'prediction' columns
        ground_truth_df: DataFrame with 'ID' and target_col
        target_col: Column name in ground truth
        score_direction: 'higher' or 'lower' for score interpretation

    Returns:
        Dictionary with keys: top_100_overlap, top_1000_overlap, top_0_1_percent_overlap, top_1_percent_overlap, top_10_percent_overlap
    """
    # Merge predictions with ground truth
    merged = predictions_df.join(
        ground_truth_df.select(['ID', target_col]),
        on='ID',
        how='inner'
    )
    n_total = len(merged)

    if n_total == 0:
        return {
            'top_100_overlap': 0.0,
            'top_1000_overlap': 0.0,
            'top_0_1_percent_overlap': 0.0,
            'top_1_percent_overlap': 0.0,
            'top_10_percent_overlap': 0.0
        }

    # Ensure prediction column is numeric
    if merged.get_column('prediction').dtype != pl.Float64 and merged.get_column('prediction').dtype != pl.Float32:
        merged = merged.with_columns(
            pl.col('prediction').cast(pl.Float64, strict=False)
        )
        merged = merged.filter(pl.col('prediction').is_not_null())
        n_total = len(merged)
        if n_total == 0:
            return {
                'top_100_overlap': 0.0,
                'top_1000_overlap': 0.0,
                'top_0_1_percent_overlap': 0.0,
                'top_1_percent_overlap': 0.0,
                'top_10_percent_overlap': 0.0
            }

    # Define K values: fixed numbers and percentages
    k_values = {
        'top_100_overlap': min(100, n_total),
        'top_1000_overlap': min(1000, n_total),
        'top_0_1_percent_overlap': max(1, int(n_total * 0.001)),  # 0.1%
        'top_1_percent_overlap': max(1, int(n_total * 0.01)),     # 1%
        'top_10_percent_overlap': max(1, int(n_total * 0.10))     # 10%
    }

    results = {}
    descending = (score_direction == 'higher')

    for key, k in k_values.items():
        if k > n_total:
            k = n_total

        # Get top k by predictions
        top_k_predicted = set(
            merged.sort('prediction', descending=descending)
            .head(k)
            .get_column('ID')
            .to_list()
        )

        # Get top k by ground truth
        top_k_true = set(
            merged.sort(target_col, descending=descending)
            .head(k)
            .get_column('ID')
            .to_list()
        )

        # Calculate overlap
        overlap_count = len(top_k_predicted & top_k_true)
        overlap_percentage = (overlap_count / k) * 100
        results[key] = round(overlap_percentage, 2)

    return results


def calculate_multiple_enrichment_factors(scores: np.ndarray, labels: np.ndarray,
                                        score_direction: str = 'higher') -> dict:
    """
    Calculate multiple enrichment factors at fixed percentiles: 5%, 1%, 0.5%, 0.1%.

    Args:
        scores: Screening scores
        labels: Binary labels (0/1)
        score_direction: 'higher' or 'lower' for score interpretation

    Returns:
        Dictionary with keys: ef_5, ef_1, ef_0_5, ef_0_1
    """
    percentiles = [5.0, 1.0, 0.5, 0.1]

    results = {}
    for p in percentiles:
        try:
            ef_value = calculate_enrichment_factor(scores, labels, p, score_direction)
            # Create key name (e.g., 5.0 -> ef_5, 0.5 -> ef_0_5)
            key = f"ef_{str(p).replace('.', '_')}"
            results[key] = ef_value
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger = get_logger()
            key = f"ef_{str(p).replace('.', '_')}"
            log_warning(logger, f"Could not calculate {key}: {e}. Setting to 0.0.")
            results[key] = 0.0

    return results


def calculate_ground_truth_enrichment_factors(ground_truth_df: pl.DataFrame,
                                            target_col: str,
                                            score_direction: str = 'higher') -> dict:
    """
    Calculate ground truth enrichment factors using true target scores vs Activity labels.
    These remain constant across all cycles since they're based on ground truth data.

    Args:
        ground_truth_df: DataFrame with ground truth values including Activity column
        target_col: Column name for target scores to use as screening scores
        score_direction: 'higher' or 'lower' for score interpretation

    Returns:
        Dictionary with keys: ground_truth_ef_5_0, ground_truth_ef_1_0, ground_truth_ef_0_5, ground_truth_ef_0_1
    """
    # Check if Activity column exists for binary labels
    if 'Activity' not in ground_truth_df.columns:
        # Return empty dict with None values if no Activity column
        return {
            'ground_truth_ef_5_0': None,
            'ground_truth_ef_1_0': None,
            'ground_truth_ef_0_5': None,
            'ground_truth_ef_0_1': None
        }

    # Remove any NaN values for calculation
    valid_data = ground_truth_df.filter(
        pl.col(target_col).is_not_null() & pl.col('Activity').is_not_null()
    )

    if len(valid_data) == 0:
        return {
            'ground_truth_ef_5_0': None,
            'ground_truth_ef_1_0': None,
            'ground_truth_ef_0_5': None,
            'ground_truth_ef_0_1': None
        }

    # Use target scores as screening scores and Activity as binary labels
    scores = valid_data.get_column(target_col).to_numpy()
    labels = valid_data.get_column('Activity').to_numpy()

    # Calculate enrichment factors at fixed percentiles
    percentiles = [5.0, 1.0, 0.5, 0.1]

    results = {}
    for p in percentiles:
        try:
            ef_value = calculate_enrichment_factor(scores, labels, p, score_direction)
            # Create key name with ground_truth prefix - use consistent naming with other EF functions
            key = f"ground_truth_ef_{str(p).replace('.', '_')}"
            results[key] = ef_value
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger = get_logger()
            key = f"ground_truth_ef_{str(p).replace('.', '_')}"
            log_warning(logger, f"Could not calculate {key}: {e}. Setting to None.")
            results[key] = None

    return results


def calculate_multiple_top_k_discovery_rates(
    selected_ids: set,
    ground_truth_df: pl.DataFrame,
    target_column: str,
    score_direction: str = 'higher'
) -> dict:
    """
    Calculate discovery rates at multiple K values (both fixed and percentage-based).

    Discovery rate = fraction of true top-K compounds that have been selected.
    NO predictions needed - pure discovery metric.

    Args:
        selected_ids: Set of all compound IDs selected so far (cumulative)
        ground_truth_df: DataFrame with ground truth target values (must have 'ID' column)
        target_column: Column name for target property
        score_direction: 'higher' or 'lower' for score interpretation

    Returns:
        Dictionary with 6 discovery rates:
        - top_10_discovery: Discovery rate for top-10 compounds (%)
        - top_100_discovery: Discovery rate for top-100 compounds (%)
        - top_1000_discovery: Discovery rate for top-1000 compounds (%)
        - top_0_1_pct_discovery: Discovery rate for top-0.1% compounds (%)
        - top_1_pct_discovery: Discovery rate for top-1% compounds (%)
        - top_10_pct_discovery: Discovery rate for top-10% compounds (%)
    """
    n_total = len(ground_truth_df)
    descending = (score_direction == 'higher')

    # Define K values: fixed numbers and percentages
    k_values = {
        'top_10_discovery': min(10, n_total),
        'top_100_discovery': min(100, n_total),
        'top_1000_discovery': min(1000, n_total),
        'top_0_1_pct_discovery': max(1, int(n_total * 0.001)),  # 0.1%
        'top_1_pct_discovery': max(1, int(n_total * 0.01)),     # 1%
        'top_10_pct_discovery': max(1, int(n_total * 0.10))     # 10%
    }

    max_k = max(k_values.values())
    sorted_ids = (
        ground_truth_df.sort(target_column, descending=descending)
        .head(max_k)
        .get_column('ID')
        .to_list()
    )

    results = {}
    for key, k in k_values.items():
        true_top_k = set(sorted_ids[:k])
        discovered = selected_ids & true_top_k
        discovery_rate = (len(discovered) / k) * 100
        results[key] = round(discovery_rate, 2)

    return results


def calculate_cumulative_enrichment_factor(
    selected_ids: set,
    ground_truth_df: pl.DataFrame,
    activity_column: str = 'Activity'
) -> float:
    """
    Calculate enrichment factor of all selections vs random.

    Works with BINARY labels (active/inactive). Returns None if Activity column absent.

    Args:
        selected_ids: Set of all selected compound IDs
        ground_truth_df: DataFrame with ground truth Activity labels
        activity_column: Column name for binary activity (0/1)

    Returns:
        Enrichment factor, or None if Activity column not present
    """
    # Check if Activity column present
    if activity_column not in ground_truth_df.columns:
        return None

    # Selected compounds
    selected_df = ground_truth_df.filter(pl.col('ID').is_in(selected_ids))
    n_selected = len(selected_df)
    if n_selected == 0:
        return None

    n_actives_found = selected_df.filter(pl.col(activity_column) == 1).height

    # Population
    n_total = len(ground_truth_df)
    n_actives_total = ground_truth_df.filter(pl.col(activity_column) == 1).height

    # Calculate EF
    if n_actives_total == 0:
        return None

    hit_rate_selected = n_actives_found / n_selected
    hit_rate_population = n_actives_total / n_total

    cumulative_ef = hit_rate_selected / hit_rate_population

    return round(cumulative_ef, 3)


def calculate_batch_hit_rate(
    newly_selected_df: pl.DataFrame,
    activity_column: str = 'Activity'
) -> float:
    """
    Calculate hit rate for current batch.

    Returns None if Activity column not present.

    Args:
        newly_selected_df: DataFrame of newly selected compounds (with Activity measured)
        activity_column: Column name for binary activity

    Returns:
        Hit rate as fraction (0-1), or None if Activity column absent
    """
    if activity_column not in newly_selected_df.columns:
        return None

    n_batch = len(newly_selected_df)
    if n_batch == 0:
        return None

    n_actives_batch = newly_selected_df.filter(pl.col(activity_column) == 1).height

    batch_hit_rate = n_actives_batch / n_batch

    return round(batch_hit_rate, 4)


def calculate_batch_enrichment_factor(
    newly_selected_df: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
    activity_column: str = 'Activity'
) -> float:
    """
    Calculate enrichment factor for this cycle's batch.

    Returns None if Activity column not present.

    Args:
        newly_selected_df: DataFrame of newly selected compounds
        ground_truth_df: Full ground truth DataFrame
        activity_column: Column name for binary activity

    Returns:
        Batch enrichment factor, or None if Activity column absent
    """
    # Check if Activity column present
    if activity_column not in newly_selected_df.columns or \
       activity_column not in ground_truth_df.columns:
        return None

    # Batch statistics
    n_batch = len(newly_selected_df)
    if n_batch == 0:
        return None

    n_actives_batch = newly_selected_df.filter(pl.col(activity_column) == 1).height

    # Population statistics
    n_total = len(ground_truth_df)
    n_actives_total = ground_truth_df.filter(pl.col(activity_column) == 1).height

    # Calculate EF
    if n_actives_total == 0:
        return None

    hit_rate_batch = n_actives_batch / n_batch
    hit_rate_population = n_actives_total / n_total

    batch_ef = hit_rate_batch / hit_rate_population

    return round(batch_ef, 3)


def _validate_score_direction(score_direction: str) -> ScoreDirection:
    if score_direction not in ("higher", "lower"):
        raise ValueError(
            f"score_direction must be 'higher' or 'lower', got {score_direction!r}."
        )
    return score_direction  # type: ignore[return-value]


def _improvement_ratio(
    selected_mean: float,
    pop_mean: float,
    score_direction: ScoreDirection,
) -> float:
    """Sign-aware improvement ratio formula (FR-012, FR-013).

    Neutral baseline is ``0.0`` ("no improvement") in both finite and degenerate
    pop_mean cases; ``± inf`` only when the improvement is genuinely unbounded
    (selected != 0 but pop == 0).
    """
    delta = selected_mean - pop_mean if score_direction == "higher" else pop_mean - selected_mean

    if pop_mean == 0.0:
        if selected_mean == 0.0:
            return 0.0
        return float(np.copysign(np.inf, delta))
    return delta / abs(pop_mean)


def calculate_score_improvement_ratio(
    selected_ids: set | Sequence[str],
    ground_truth_df: pl.DataFrame,
    target_column: str,
    *,
    score_direction: ScoreDirection,
) -> float:
    """Sign-aware improvement of selected mean over population mean.

    Introduced in feature 019 (no backward-compat alias for the legacy
    abs()-trick implementation). Formula::

      higher:  (selected_mean - pop_mean) / |pop_mean|
      lower:   (pop_mean - selected_mean) / |pop_mean|

    Sign convention:
      ``> 0`` means selection improves on population in the desired direction.
      ``> 1`` means > 100% relative improvement (selected mean ≥ 2 × pop_mean
      for ``higher``, or ≥ 2× more-negative for ``lower``). Neutral baseline
      is ``0`` (NOT ``1``; breaks from classical Enrichment Factor convention).

    Edge cases:
      - ``pop_mean == 0`` and ``selected_mean == 0`` → returns ``0.0`` (parity).
      - ``pop_mean == 0`` and ``selected_mean != 0`` → returns signed ``inf``.
      - Empty ``selected_ids`` → raises ``ValueError`` (fail-fast per FR-013).

    Args:
        selected_ids: Set or sequence of compound IDs selected so far.
        ground_truth_df: DataFrame with ``ID`` and ``target_column``.
        target_column: Column name for target property.
        score_direction: ``'higher'`` or ``'lower'`` (kw-only, no default).

    Returns:
        Improvement ratio (finite or signed ``inf``, never NaN).
    """
    direction = _validate_score_direction(score_direction)
    if len(selected_ids) == 0:
        raise ValueError(
            "score_improvement_ratio called with empty selected_ids; "
            "short-circuit at the call site instead."
        )

    selected_df = ground_truth_df.filter(pl.col("ID").is_in(selected_ids))
    if len(selected_df) == 0:
        raise ValueError(
            "score_improvement_ratio called with selected_ids that do not match "
            "any rows in ground_truth_df; check ID consistency across pool and selection."
        )
    selected_mean = float(selected_df.get_column(target_column).mean())  # type: ignore[arg-type]
    pop_mean = float(ground_truth_df.get_column(target_column).mean())  # type: ignore[arg-type]
    return _improvement_ratio(selected_mean, pop_mean, direction)


def calculate_batch_score_improvement_ratio(
    newly_selected_df: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
    target_column: str,
    *,
    score_direction: ScoreDirection,
) -> float:
    """Per-batch sign-aware improvement ratio (introduced in feature 019).

    Same contract as :func:`calculate_score_improvement_ratio`, but operates on
    an already-filtered ``newly_selected_df`` instead of an ID set.
    """
    direction = _validate_score_direction(score_direction)
    if len(newly_selected_df) == 0:
        raise ValueError(
            "batch_score_improvement_ratio called with empty newly_selected_df; "
            "short-circuit at the call site instead."
        )
    selected_mean = float(newly_selected_df.get_column(target_column).mean())  # type: ignore[arg-type]
    pop_mean = float(ground_truth_df.get_column(target_column).mean())  # type: ignore[arg-type]
    return _improvement_ratio(selected_mean, pop_mean, direction)


def calculate_multiple_unlabeled_top_k_overlaps(
    unlabeled_predictions_df: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
    target_column: str,
    score_direction: str = 'higher',
    merged_target: pl.DataFrame | None = None
) -> dict:
    """
    Calculate top-K ranking overlaps on UNLABELED compounds only.

    CRITICAL: unlabeled_predictions_df MUST exclude all labeled/training compounds
    to avoid training data contamination.

    Ranking overlap = model's predicted top-K ∩ true top-K (within unlabeled pool)

    Args:
        unlabeled_predictions_df: DataFrame with 'ID' and 'prediction' for UNLABELED compounds only
        ground_truth_df: DataFrame with ground truth values
        target_column: Column name for target property
        score_direction: 'higher' or 'lower' for score interpretation
        merged_target: Pre-joined DataFrame (unlabeled_predictions_df ⋈ ground_truth_df on target_column).
            When provided, skips the internal join for performance.

    Returns:
        Dictionary with:
        - unlabeled_top_100_overlap: Overlap percentage for top-100 (%)
        - unlabeled_top_1000_overlap: Overlap percentage for top-1000 (%)
    """
    k_values = {
        'unlabeled_top_100_overlap': 100,
        'unlabeled_top_1000_overlap': 1000
    }

    if merged_target is None:
        merged = unlabeled_predictions_df.join(
            ground_truth_df.select(['ID', target_column]),
            on='ID',
            how='inner'
        )
    else:
        merged = merged_target

    descending = (score_direction == 'higher')
    n_merged = len(merged)

    if n_merged == 0:
        return {key: 0.0 for key in k_values}

    max_k = max(k_values.values())
    pred_ranked_ids = (
        merged.sort('prediction', descending=descending)
        .head(max_k)
        .get_column('ID')
        .to_list()
    )
    true_ranked_ids = (
        merged.sort(target_column, descending=descending)
        .head(max_k)
        .get_column('ID')
        .to_list()
    )

    results = {}
    for key, k in k_values.items():
        k_actual = min(k, n_merged)
        model_top_k = set(pred_ranked_ids[:k_actual])
        true_top_k = set(true_ranked_ids[:k_actual])
        overlap = len(model_top_k & true_top_k)
        overlap_pct = (overlap / k_actual) * 100
        results[key] = round(overlap_pct, 2)

    return results


def calculate_multiple_unlabeled_enrichment_factors(
    unlabeled_predictions_df: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
    activity_column: str,
    score_direction: str = 'higher',
    merged_activity: pl.DataFrame | None = None
) -> dict:
    """
    Calculate prospective enrichment factors on UNLABELED compounds only.

    "If we selected top X% of unlabeled by model, what EF would we get?"

    Returns None values if Activity column absent.

    Args:
        unlabeled_predictions_df: Predictions on UNLABELED compounds only
        ground_truth_df: Ground truth with Activity labels
        activity_column: Column name for binary labels
        score_direction: 'higher' or 'lower' for score interpretation
        merged_activity: Pre-joined DataFrame (unlabeled_predictions_df ⋈ ground_truth_df on activity_column).
            When provided, skips the internal join for performance. Must be None when activity_column absent.

    Returns:
        Dictionary with:
        - unlabeled_ef_1_0: Prospective EF at 1% (or None)
        - unlabeled_ef_5_0: Prospective EF at 5% (or None)
    """
    if merged_activity is not None:
        merged = merged_activity
    else:
        # Check if Activity column present
        if activity_column not in ground_truth_df.columns:
            return {
                'unlabeled_ef_1_0': None,
                'unlabeled_ef_5_0': None
            }
        merged = unlabeled_predictions_df.join(
            ground_truth_df.select(['ID', activity_column]),
            on='ID',
            how='inner'
        )

    if len(merged) == 0:
        return {
            'unlabeled_ef_1_0': None,
            'unlabeled_ef_5_0': None
        }

    percentiles = [1.0, 5.0]
    results = {}
    descending = (score_direction == 'higher')

    for p in percentiles:
        # Sort by model predictions
        sorted_df = merged.sort('prediction', descending=descending)

        # Select top percentile
        n_total = len(sorted_df)
        n_select = max(1, int(n_total * p / 100))
        top_percentile = sorted_df.head(n_select)

        # Calculate EF
        n_actives_selected = top_percentile.filter(pl.col(activity_column) == 1).height
        n_actives_total = merged.filter(pl.col(activity_column) == 1).height

        if n_select == 0 or n_actives_total == 0:
            ef = 0.0
        else:
            hit_rate_selected = n_actives_selected / n_select
            hit_rate_population = n_actives_total / n_total
            ef = hit_rate_selected / hit_rate_population

        # Create key name (1.0 -> ef_1_0, 5.0 -> ef_5_0)
        key = f"unlabeled_ef_{str(p).replace('.', '_')}"
        results[key] = round(ef, 3)

    return results


def calculate_unlabeled_ranking_correlation(
    unlabeled_predictions_df: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
    target_column: str,
    merged_target: pl.DataFrame | None = None
) -> float:
    """
    Calculate Spearman correlation on UNLABELED compounds only.

    Args:
        unlabeled_predictions_df: Predictions on unlabeled only
        ground_truth_df: Ground truth values
        target_column: Target property column
        merged_target: Pre-joined DataFrame (unlabeled_predictions_df ⋈ ground_truth_df on target_column).
            When provided, skips the internal join for performance.

    Returns:
        Spearman correlation coefficient
    """
    from scipy.stats import spearmanr

    if merged_target is None:
        merged = unlabeled_predictions_df.join(
            ground_truth_df.select(['ID', target_column]),
            on='ID',
            how='inner'
        )
    else:
        merged = merged_target

    if len(merged) < 2:
        return 0.0

    predictions = merged.get_column('prediction').to_numpy()
    ground_truth_vals = merged.get_column(target_column).to_numpy()

    correlation, _ = spearmanr(predictions, ground_truth_vals)

    return round(correlation, 4) if not np.isnan(correlation) else 0.0

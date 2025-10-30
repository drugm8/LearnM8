"""Enrichment metrics for virtual screening evaluation."""

import numpy as np
import pandas as pd
from learnm8.utils.logging import get_logger, log_warning


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


def calculate_top_k_overlap(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
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
    merged = pd.merge(predictions_df, ground_truth_df[['ID', target_col]], on='ID')

    # Handle empty data case
    if len(merged) == 0:
        return 0.0

    if len(merged) < k:
        logger = get_logger()
        log_warning(logger, f"Only {len(merged)} compounds available, using all for top-k calculation")
        k = len(merged)

    # Ensure prediction column is numeric
    if merged['prediction'].dtype == 'object':
        merged['prediction'] = pd.to_numeric(merged['prediction'], errors='coerce')
        merged = merged.dropna(subset=['prediction'])
        if len(merged) == 0:
            return 0.0

    # Sort by scores
    ascending = (score_direction == 'lower')

    # Get top k by predictions
    top_k_predicted = set(merged.nlargest(k, 'prediction', keep='first')['ID'].values) \
                      if not ascending else \
                      set(merged.nsmallest(k, 'prediction', keep='first')['ID'].values)

    # Get top k by ground truth
    top_k_true = set(merged.nlargest(k, target_col, keep='first')['ID'].values) \
                 if not ascending else \
                 set(merged.nsmallest(k, target_col, keep='first')['ID'].values)

    # Calculate overlap
    overlap_count = len(top_k_predicted & top_k_true)
    overlap_percentage = (overlap_count / k) * 100

    return round(overlap_percentage, 2)


def calculate_multiple_top_k_overlaps(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
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
    merged = pd.merge(predictions_df, ground_truth_df[['ID', target_col]], on='ID')
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
    if merged['prediction'].dtype == 'object':
        merged['prediction'] = pd.to_numeric(merged['prediction'], errors='coerce')
        merged = merged.dropna(subset=['prediction'])
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
    ascending = (score_direction == 'lower')

    for key, k in k_values.items():
        if k > n_total:
            k = n_total

        # Get top k by predictions
        top_k_predicted = set(merged.nlargest(k, 'prediction', keep='first')['ID'].values) \
                          if not ascending else \
                          set(merged.nsmallest(k, 'prediction', keep='first')['ID'].values)

        # Get top k by ground truth
        top_k_true = set(merged.nlargest(k, target_col, keep='first')['ID'].values) \
                     if not ascending else \
                     set(merged.nsmallest(k, target_col, keep='first')['ID'].values)

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
        except Exception as e:
            # Handle edge cases gracefully
            key = f"ef_{str(p).replace('.', '_')}"
            results[key] = 0.0

    return results


def calculate_ground_truth_enrichment_factors(ground_truth_df: pd.DataFrame,
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
    valid_data = ground_truth_df.dropna(subset=[target_col, 'Activity'])

    if len(valid_data) == 0:
        return {
            'ground_truth_ef_5_0': None,
            'ground_truth_ef_1_0': None,
            'ground_truth_ef_0_5': None,
            'ground_truth_ef_0_1': None
        }

    # Use target scores as screening scores and Activity as binary labels
    scores = valid_data[target_col].values
    labels = valid_data['Activity'].values

    # Calculate enrichment factors at fixed percentiles
    percentiles = [5.0, 1.0, 0.5, 0.1]

    results = {}
    for p in percentiles:
        try:
            ef_value = calculate_enrichment_factor(scores, labels, p, score_direction)
            # Create key name with ground_truth prefix - use consistent naming with other EF functions
            key = f"ground_truth_ef_{str(p).replace('.', '_')}"
            results[key] = ef_value
        except Exception as e:
            # Handle edge cases gracefully
            key = f"ground_truth_ef_{str(p).replace('.', '_')}"
            results[key] = None

    return results


def calculate_multiple_top_k_discovery_rates(
    selected_ids: set,
    ground_truth_df: pd.DataFrame,
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
    ascending = (score_direction == 'lower')

    # Define K values: fixed numbers and percentages
    k_values = {
        'top_10_discovery': min(10, n_total),
        'top_100_discovery': min(100, n_total),
        'top_1000_discovery': min(1000, n_total),
        'top_0_1_pct_discovery': max(1, int(n_total * 0.001)),  # 0.1%
        'top_1_pct_discovery': max(1, int(n_total * 0.01)),     # 1%
        'top_10_pct_discovery': max(1, int(n_total * 0.10))     # 10%
    }

    results = {}
    for key, k in k_values.items():
        # Get true top-K compounds
        if ascending:
            true_top_k = set(ground_truth_df.nsmallest(k, target_column)['ID'].values)
        else:
            true_top_k = set(ground_truth_df.nlargest(k, target_column)['ID'].values)

        # Calculate discovery: how many of true top-K have we selected?
        discovered = selected_ids & true_top_k
        discovery_rate = (len(discovered) / k) * 100
        results[key] = round(discovery_rate, 2)

    return results


def calculate_cumulative_enrichment_factor(
    selected_ids: set,
    ground_truth_df: pd.DataFrame,
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
    selected_df = ground_truth_df[ground_truth_df['ID'].isin(selected_ids)]
    n_selected = len(selected_df)
    if n_selected == 0:
        return None

    n_actives_found = (selected_df[activity_column] == 1).sum()

    # Population
    n_total = len(ground_truth_df)
    n_actives_total = (ground_truth_df[activity_column] == 1).sum()

    # Calculate EF
    if n_actives_total == 0:
        return None

    hit_rate_selected = n_actives_found / n_selected
    hit_rate_population = n_actives_total / n_total

    cumulative_ef = hit_rate_selected / hit_rate_population

    return round(cumulative_ef, 3)


def calculate_batch_hit_rate(
    newly_selected_df: pd.DataFrame,
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

    n_actives_batch = (newly_selected_df[activity_column] == 1).sum()

    batch_hit_rate = n_actives_batch / n_batch

    return round(batch_hit_rate, 4)


def calculate_batch_enrichment_factor(
    newly_selected_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
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

    n_actives_batch = (newly_selected_df[activity_column] == 1).sum()

    # Population statistics
    n_total = len(ground_truth_df)
    n_actives_total = (ground_truth_df[activity_column] == 1).sum()

    # Calculate EF
    if n_actives_total == 0:
        return None

    hit_rate_batch = n_actives_batch / n_batch
    hit_rate_population = n_actives_total / n_total

    batch_ef = hit_rate_batch / hit_rate_population

    return round(batch_ef, 3)


def calculate_average_score_ratio(
    selected_ids: set,
    ground_truth_df: pd.DataFrame,
    target_column: str,
    score_direction: str = 'higher'
) -> float:
    """
    Calculate ratio of average scores between all selections and population.

    Alternative to enrichment factor when only continuous scores available
    (no binary Activity labels).

    The ratio compares the magnitude of selected compounds' average score
    to the population average. For score_direction='lower', uses absolute
    values to correctly handle negative scores (e.g., docking energies where
    more negative values indicate better binding).

    Args:
        selected_ids: Set of all compound IDs selected so far (cumulative)
        ground_truth_df: DataFrame with ground truth target values
        target_column: Column name for target property
        score_direction: 'higher' or 'lower' for score interpretation

    Returns:
        Score ratio (>1.0 means selections better than average)
    """
    # Selected compounds
    selected_df = ground_truth_df[ground_truth_df['ID'].isin(selected_ids)]
    if len(selected_df) == 0:
        return 1.0

    avg_score_selected = selected_df[target_column].mean()

    # Population
    avg_score_population = ground_truth_df[target_column].mean()

    # Calculate ratio based on direction
    if score_direction == 'higher':
        # Higher scores are better
        score_ratio = avg_score_selected / avg_score_population
    else:
        # Lower scores are better (e.g., docking scores, energies)
        # Use absolute values for magnitude comparison to handle negative scores
        score_ratio = abs(avg_score_selected) / abs(avg_score_population)

    return round(score_ratio, 3)


def calculate_batch_average_score_ratio(
    newly_selected_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    target_column: str,
    score_direction: str = 'higher'
) -> float:
    """
    Calculate score ratio for current batch only.

    The ratio compares the magnitude of batch average score to population
    average. For score_direction='lower', uses absolute values to correctly
    handle negative scores (e.g., docking energies where more negative values
    indicate better binding).

    Args:
        newly_selected_df: DataFrame of newly selected compounds this cycle
        ground_truth_df: Full ground truth DataFrame
        target_column: Column name for target property
        score_direction: 'higher' or 'lower' for score interpretation

    Returns:
        Batch score ratio (>1.0 means batch better than average)
    """
    if len(newly_selected_df) == 0:
        return 1.0

    avg_score_batch = newly_selected_df[target_column].mean()
    avg_score_population = ground_truth_df[target_column].mean()

    # Calculate ratio based on direction
    if score_direction == 'higher':
        score_ratio = avg_score_batch / avg_score_population
    else:
        # Lower scores are better (e.g., docking scores, energies)
        # Use absolute values for magnitude comparison to handle negative scores
        score_ratio = abs(avg_score_batch) / abs(avg_score_population)

    return round(score_ratio, 3)


def calculate_multiple_unlabeled_top_k_overlaps(
    unlabeled_predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    target_column: str,
    score_direction: str = 'higher'
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

    Returns:
        Dictionary with:
        - unlabeled_top_100_overlap: Overlap percentage for top-100 (%)
        - unlabeled_top_1000_overlap: Overlap percentage for top-1000 (%)
    """
    k_values = {
        'unlabeled_top_100_overlap': 100,
        'unlabeled_top_1000_overlap': 1000
    }

    # Merge predictions with ground truth (unlabeled only)
    merged = pd.merge(
        unlabeled_predictions_df,
        ground_truth_df[['ID', target_column]],
        on='ID'
    )

    results = {}
    ascending = (score_direction == 'lower')

    for key, k in k_values.items():
        # Adjust k if unlabeled pool smaller
        k_actual = min(k, len(merged))
        if k_actual == 0:
            results[key] = 0.0
            continue

        # Get top-K by MODEL predictions (on unlabeled)
        if ascending:
            model_top_k = set(merged.nsmallest(k_actual, 'prediction')['ID'].values)
        else:
            model_top_k = set(merged.nlargest(k_actual, 'prediction')['ID'].values)

        # Get top-K by TRUTH (within unlabeled)
        if ascending:
            true_top_k = set(merged.nsmallest(k_actual, target_column)['ID'].values)
        else:
            true_top_k = set(merged.nlargest(k_actual, target_column)['ID'].values)

        # Calculate overlap
        overlap = len(model_top_k & true_top_k)
        overlap_pct = (overlap / k_actual) * 100
        results[key] = round(overlap_pct, 2)

    return results


def calculate_multiple_unlabeled_enrichment_factors(
    unlabeled_predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    activity_column: str,
    score_direction: str = 'higher'
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

    Returns:
        Dictionary with:
        - unlabeled_ef_1_0: Prospective EF at 1% (or None)
        - unlabeled_ef_5_0: Prospective EF at 5% (or None)
    """
    # Check if Activity column present
    if activity_column not in ground_truth_df.columns:
        return {
            'unlabeled_ef_1_0': None,
            'unlabeled_ef_5_0': None
        }

    # Merge unlabeled predictions with Activity labels
    merged = pd.merge(
        unlabeled_predictions_df,
        ground_truth_df[['ID', activity_column]],
        on='ID'
    )

    if len(merged) == 0:
        return {
            'unlabeled_ef_1_0': None,
            'unlabeled_ef_5_0': None
        }

    percentiles = [1.0, 5.0]
    results = {}
    ascending = (score_direction == 'lower')

    for p in percentiles:
        # Sort by model predictions
        sorted_df = merged.sort_values('prediction', ascending=ascending)

        # Select top percentile
        n_total = len(sorted_df)
        n_select = max(1, int(n_total * p / 100))
        top_percentile = sorted_df.head(n_select)

        # Calculate EF
        n_actives_selected = (top_percentile[activity_column] == 1).sum()
        n_actives_total = (merged[activity_column] == 1).sum()

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
    unlabeled_predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    target_column: str
) -> float:
    """
    Calculate Spearman correlation on UNLABELED compounds only.

    Args:
        unlabeled_predictions_df: Predictions on unlabeled only
        ground_truth_df: Ground truth values
        target_column: Target property column

    Returns:
        Spearman correlation coefficient
    """
    from scipy.stats import spearmanr

    merged = pd.merge(
        unlabeled_predictions_df,
        ground_truth_df[['ID', target_column]],
        on='ID'
    )

    if len(merged) < 2:
        return 0.0

    correlation, _ = spearmanr(merged['prediction'], merged[target_column])

    return round(correlation, 4) if not np.isnan(correlation) else 0.0
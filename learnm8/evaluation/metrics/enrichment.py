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
    ascending = (score_direction == 'lower')
    sorted_indices = np.argsort(scores) if ascending else np.argsort(scores)[::-1]
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
                           k: int, target_column: str, score_direction: str = 'higher') -> float:
    """
    Calculate percentage overlap between top-k predicted and true compounds.

    Args:
        predictions_df: DataFrame with 'ID' and 'prediction' columns
        ground_truth_df: DataFrame with 'ID' and target_column
        k: Number of top compounds to compare
        target_column: Column name in ground truth
        score_direction: 'higher' or 'lower' for score interpretation

    Returns:
        Percentage overlap (0-100)
    """
    # Merge predictions with ground truth
    merged = pd.merge(predictions_df, ground_truth_df[['ID', target_column]], on='ID')

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
    top_k_true = set(merged.nlargest(k, target_column, keep='first')['ID'].values) \
                 if not ascending else \
                 set(merged.nsmallest(k, target_column, keep='first')['ID'].values)

    # Calculate overlap
    overlap_count = len(top_k_predicted & top_k_true)
    overlap_percentage = (overlap_count / k) * 100

    return round(overlap_percentage, 2)


def calculate_multiple_top_k_overlaps(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
                                    target_column: str, score_direction: str = 'higher') -> dict:
    """
    Calculate multiple top-K overlaps for different K values.

    Args:
        predictions_df: DataFrame with 'ID' and 'prediction' columns
        ground_truth_df: DataFrame with 'ID' and target_column
        target_column: Column name in ground truth
        score_direction: 'higher' or 'lower' for score interpretation

    Returns:
        Dictionary with keys: top_100_overlap, top_1000_overlap, top_0_1_percent_overlap, top_1_percent_overlap, top_10_percent_overlap
    """
    # Merge predictions with ground truth
    merged = pd.merge(predictions_df, ground_truth_df[['ID', target_column]], on='ID')
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
        top_k_true = set(merged.nlargest(k, target_column, keep='first')['ID'].values) \
                     if not ascending else \
                     set(merged.nsmallest(k, target_column, keep='first')['ID'].values)

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
                                            target_column: str,
                                            score_direction: str = 'higher') -> dict:
    """
    Calculate ground truth enrichment factors using true target scores vs Activity labels.
    These remain constant across all cycles since they're based on ground truth data.

    Args:
        ground_truth_df: DataFrame with ground truth values including Activity column
        target_column: Column name for target scores to use as screening scores
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
    valid_data = ground_truth_df.dropna(subset=[target_column, 'Activity'])

    if len(valid_data) == 0:
        return {
            'ground_truth_ef_5_0': None,
            'ground_truth_ef_1_0': None,
            'ground_truth_ef_0_5': None,
            'ground_truth_ef_0_1': None
        }

    # Use target scores as screening scores and Activity as binary labels
    scores = valid_data[target_column].values
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
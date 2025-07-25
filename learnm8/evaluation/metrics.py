"""Evaluation metrics for active learning performance."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_average_score(scores: np.ndarray) -> float:
    """
    Calculate average score of compounds.
    
    Args:
        scores: Array of scores
        
    Returns:
        Average score
    """
    return np.mean(scores)


def calculate_enrichment_factor(scores: np.ndarray, labels: np.ndarray, 
                               percentile: float) -> float:
    """
    Calculate Enrichment Factor for virtual screening.
    
    EF = (n_actives_selected / n_selected) / (n_actives_total / n_total)
    
    Args:
        scores: Screening scores
        labels: Binary labels (0/1)
        percentile: Percentile threshold for selection (e.g., 1.0 for top 1%)
        
    Returns:
        Enrichment factor value
    """
    if not 0 < percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    
    # Sort by scores (descending)
    sorted_indices = np.argsort(scores)[::-1]
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
    
    if len(merged) < k:
        print(f"Warning: Only {len(merged)} compounds available, using all for top-k calculation")
        k = len(merged)
    
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
"""ML model performance metrics for active learning evaluation."""

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import spearmanr


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE value as percentage
    """
    # Handle potential division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = mean_absolute_percentage_error(y_true, y_pred)
        # Convert to percentage and handle inf/nan values
        mape_percentage = mape * 100
        if np.isnan(mape_percentage) or np.isinf(mape_percentage):
            return 0.0
        return mape_percentage


def calculate_spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Spearman rank correlation coefficient.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Spearman correlation coefficient
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0

    try:
        correlation, _ = spearmanr(y_true, y_pred)
        # Handle nan values
        if np.isnan(correlation):
            return 0.0
        return correlation
    except Exception:
        return 0.0


def calculate_average_score(scores: np.ndarray) -> float:
    """
    Calculate average score of compounds.

    Args:
        scores: Array of scores

    Returns:
        Average score, or None if empty/invalid array
    """
    if len(scores) == 0:
        return None

    # Filter out NaN values
    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) == 0:
        return None

    return np.mean(valid_scores)
"""ML model performance metrics for active learning evaluation."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_percentage_error


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


def calculate_spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """
    Calculate Spearman rank correlation coefficient.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Spearman correlation coefficient, or None when undefined
        (N<2, σ_pred=0 constant predictions, or scipy returns NaN).
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return None

    try:
        correlation, _ = spearmanr(y_true, y_pred)
        if np.isnan(correlation):
            return None
        return float(correlation)
    except (ValueError, TypeError):
        return None


def calculate_average_score(scores: np.ndarray) -> float | None:
    """Calculate average score of compounds.

    Predictions and target scores are guaranteed to be NaN-free at this point
    (cycle.py FR-005 upstream guard, feature 019). The previous silent
    ``np.isnan`` filter is removed so any NaN escaping the upstream guard
    propagates instead of being absorbed.

    Args:
        scores: Array of scores.

    Returns:
        Average score, or ``None`` if the array is empty.
    """
    if len(scores) == 0:
        return None
    return float(np.mean(scores))

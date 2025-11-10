import numpy as np
import polars as pl
from typing import Optional, Tuple


def downsample_for_viz(data: np.ndarray, max_points: int = 5000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample data for visualization to improve performance.

    Args:
        data: Array to downsample
        max_points: Maximum number of points to keep
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (downsampled_data, indices_kept)
    """
    if len(data) <= max_points:
        return data, np.arange(len(data))

    np.random.seed(random_state)
    indices = np.random.choice(len(data), max_points, replace=False)
    indices = np.sort(indices)
    return data[indices], indices


def get_status_colors(n_compounds: int, labeled_ids: set, selected_ids: set) -> np.ndarray:
    """Assign colors to compounds based on their status.

    Args:
        n_compounds: Total number of compounds
        labeled_ids: Set of labeled compound IDs
        selected_ids: Set of selected compound IDs

    Returns:
        Array of color codes (0=unlabeled, 1=labeled, 2=selected)
    """
    colors = np.full(n_compounds, 0, dtype=int)

    for i in range(n_compounds):
        if i in selected_ids:
            colors[i] = 2
        elif i in labeled_ids:
            colors[i] = 1

    return colors


def detect_benchmark_mode(metrics_df: pl.DataFrame) -> bool:
    """Detect if results are from benchmark mode based on metrics columns.

    Args:
        metrics_df: Polars DataFrame with cycle metrics

    Returns:
        True if benchmark mode detected, False otherwise
    """
    benchmark_cols = ['top_10_percent_overlap', 'ef_1_0', 'ground_truth_ef_1_0']
    return any(col in metrics_df.columns for col in benchmark_cols)


def format_metric_value(value: float, metric_name: str) -> str:
    """Format metric value for display based on metric type.

    Args:
        value: Metric value to format
        metric_name: Name of the metric

    Returns:
        Formatted string representation
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'

    if 'percent' in metric_name.lower() or 'overlap' in metric_name.lower():
        return f'{value:.1f}%'
    elif 'ef' in metric_name.lower() or 'enrichment' in metric_name.lower():
        return f'{value:.2f}x'
    elif 'r2' in metric_name.lower() or 'correlation' in metric_name.lower():
        return f'{value:.3f}'
    else:
        return f'{value:.3f}'

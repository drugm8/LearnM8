import numpy as np
import pandas as pd
from typing import Optional, Tuple


def downsample_for_viz(data: np.ndarray, max_points: int = 5000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    if len(data) <= max_points:
        return data, np.arange(len(data))

    np.random.seed(random_state)
    indices = np.random.choice(len(data), max_points, replace=False)
    indices = np.sort(indices)
    return data[indices], indices


def get_status_colors(n_compounds: int, labeled_ids: set, selected_ids: set) -> np.ndarray:
    colors = np.full(n_compounds, 0, dtype=int)

    for i in range(n_compounds):
        if i in selected_ids:
            colors[i] = 2
        elif i in labeled_ids:
            colors[i] = 1

    return colors


def detect_benchmark_mode(metrics_df: pd.DataFrame) -> bool:
    benchmark_cols = ['top_10_percent_overlap', 'ef_1_0', 'ground_truth_ef_1_0']
    return any(col in metrics_df.columns for col in benchmark_cols)


def format_metric_value(value: float, metric_name: str) -> str:
    if pd.isna(value):
        return 'N/A'

    if 'percent' in metric_name.lower() or 'overlap' in metric_name.lower():
        return f'{value:.1f}%'
    elif 'ef' in metric_name.lower() or 'enrichment' in metric_name.lower():
        return f'{value:.2f}x'
    elif 'r2' in metric_name.lower() or 'correlation' in metric_name.lower():
        return f'{value:.3f}'
    else:
        return f'{value:.3f}'

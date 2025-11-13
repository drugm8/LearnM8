"""Helper functions for multi-seed experiment aggregation and analysis.

Provides utilities for:
- Running experiments with multiple random seeds
- Aggregating results across seeds (mean, std dev)
- Smart detection of existing seed results
- Saving aggregated statistics
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import polars as pl


def aggregate_seed_results(
    seed_results: Dict[int, Dict]
) -> Dict[str, Any]:
    """Aggregate results across multiple random seeds.

    Args:
        seed_results: Dictionary mapping seed values to result dicts.
            Each result dict should contain:
            - 'cycle_metrics': List of metric dicts, one per cycle
            - 'elapsed_time': Total experiment runtime

    Returns:
        Dictionary containing:
        - 'cycle_metrics_mean': List of dicts with mean values per cycle
        - 'cycle_metrics_std': List of dicts with std dev per cycle
        - 'elapsed_time_mean': Mean runtime across seeds
        - 'elapsed_time_std': Std dev of runtime across seeds

    Example:
        >>> seed_results = {
        ...     42: {'cycle_metrics': [...], 'elapsed_time': 120.5},
        ...     123: {'cycle_metrics': [...], 'elapsed_time': 118.2},
        ...     456: {'cycle_metrics': [...], 'elapsed_time': 121.3}
        ... }
        >>> aggregated = aggregate_seed_results(seed_results)
    """
    if not seed_results:
        return {
            'cycle_metrics_mean': [],
            'cycle_metrics_std': [],
            'elapsed_time_mean': 0.0,
            'elapsed_time_std': 0.0
        }

    all_cycle_metrics = [
        seed_results[seed]['cycle_metrics']
        for seed in sorted(seed_results.keys())
    ]

    n_cycles = len(all_cycle_metrics[0])
    aggregated_mean = []
    aggregated_std = []

    for cycle_idx in range(n_cycles):
        cycle_data = [
            metrics[cycle_idx]
            for metrics in all_cycle_metrics
        ]

        mean_dict = {
            'cycle': cycle_data[0]['cycle'],
            'strategy': cycle_data[0]['strategy']
        }
        std_dict = {
            'cycle': cycle_data[0]['cycle'],
            'strategy': cycle_data[0]['strategy']
        }

        for key in cycle_data[0].keys():
            if key in ['cycle', 'strategy']:
                continue

            values = [d.get(key) for d in cycle_data if key in d and d.get(key) is not None]

            if values and all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
                mean_dict[key] = float(np.mean(values))
                std_dict[key] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            else:
                mean_dict[key] = cycle_data[0].get(key)
                std_dict[key] = 0.0

        aggregated_mean.append(mean_dict)
        aggregated_std.append(std_dict)

    elapsed_times = [
        seed_results[seed]['elapsed_time']
        for seed in sorted(seed_results.keys())
        if seed_results[seed].get('elapsed_time') is not None
    ]

    if elapsed_times:
        elapsed_time_mean = float(np.mean(elapsed_times))
        elapsed_time_std = float(np.std(elapsed_times, ddof=1)) if len(elapsed_times) > 1 else 0.0
    else:
        elapsed_time_mean = 0.0
        elapsed_time_std = 0.0

    return {
        'cycle_metrics_mean': aggregated_mean,
        'cycle_metrics_std': aggregated_std,
        'elapsed_time_mean': elapsed_time_mean,
        'elapsed_time_std': elapsed_time_std
    }


def check_existing_seed_results(
    display_name: str,
    condition: str,
    output_base: Path,
    seeds: List[int] = None
) -> List[int]:
    """Check which random seeds already have completed results.

    Args:
        display_name: Learner display name (e.g., 'chemprop_no_ft')
        condition: Condition name (acquisition strategy or featurizer name)
        output_base: Base output directory path
        seeds: List of seed values to check (default: [42, 123, 456])

    Returns:
        List of seed values that are missing or incomplete.

    Example:
        >>> missing = check_existing_seed_results('rf', 'greedy', Path('validation/reports'))
        >>> print(missing)  # [123, 456] if only seed 42 exists
    """
    if seeds is None:
        seeds = [42, 123, 456]

    missing_seeds = []
    for seed in seeds:
        output_dir = output_base / 'data' / f'{display_name}_{condition}' / f'seed_{seed}'

        required_files = [
            'compounds_final.csv',
            'cycle_metrics.csv',
            'selection_history.csv'
        ]

        if not all((output_dir / f).exists() for f in required_files):
            missing_seeds.append(seed)

    return missing_seeds


def save_aggregated_results(
    seed_results: Dict[int, Dict],
    aggregated: Dict,
    output_dir: Path
) -> None:
    """Save aggregated multi-seed results to CSV files.

    Creates two files:
    1. aggregated_cycle_metrics.csv - All cycles from all seeds with 'seed' column
    2. summary_statistics.csv - Mean and std dev per metric per cycle

    Args:
        seed_results: Dictionary mapping seed → result dict
        aggregated: Output from aggregate_seed_results()
        output_dir: Directory to save aggregated CSVs

    Example:
        >>> save_aggregated_results(seed_results, aggregated, Path('output'))
        # Creates output/aggregated_cycle_metrics.csv and output/summary_statistics.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in sorted(seed_results.keys()):
        result = seed_results[seed]
        for cycle_dict in result['cycle_metrics']:
            row = cycle_dict.copy()
            row['seed'] = seed
            all_rows.append(row)

    if all_rows:
        df = pl.DataFrame(all_rows)

        # Drop columns with nested data (lists/structs) that CSV can't handle
        nested_cols = []
        for col in df.columns:
            dtype = df[col].dtype
            if dtype in [pl.List, pl.Struct] or dtype.base_type() in [pl.List, pl.Struct]:
                nested_cols.append(col)

        if nested_cols:
            df = df.drop(nested_cols)

        df.write_csv(output_dir / 'aggregated_cycle_metrics.csv')

    mean_df = pl.DataFrame(aggregated['cycle_metrics_mean'])
    std_df = pl.DataFrame(aggregated['cycle_metrics_std'])

    summary_rows = []
    for cycle_idx in range(len(mean_df)):
        cycle_num = mean_df[cycle_idx, 'cycle']
        for col in mean_df.columns:
            if col in ['cycle', 'strategy']:
                continue

            mean_val = mean_df[cycle_idx, col]
            std_val = std_df[cycle_idx, col]

            # Convert None to NaN for consistent float typing
            if mean_val is not None or std_val is not None:
                summary_rows.append({
                    'cycle': cycle_num,
                    'metric': col,
                    'mean': float(mean_val) if mean_val is not None else float('nan'),
                    'std': float(std_val) if std_val is not None else float('nan')
                })

    if summary_rows:
        summary_df = pl.DataFrame(summary_rows)
        summary_df.write_csv(output_dir / 'summary_statistics.csv')


def load_existing_seed_results(
    display_name: str,
    condition: str,
    output_base: Path,
    seeds: List[int] = None
) -> Dict[int, Dict]:
    """Load existing seed results from disk.

    Args:
        display_name: Learner display name
        condition: Condition name (acquisition or featurizer)
        output_base: Base output directory
        seeds: Seeds to load (default: [42, 123, 456])

    Returns:
        Dictionary mapping seed → result dict with:
        - 'compounds_df': Final compounds DataFrame
        - 'cycle_metrics': List of cycle metric dicts
        - 'elapsed_time': None (not stored in existing CSVs)

    Example:
        >>> results = load_existing_seed_results('rf', 'greedy', Path('validation/reports'))
        >>> results[42]['cycle_metrics']  # Access cycle metrics for seed 42
    """
    if seeds is None:
        seeds = [42, 123, 456]

    seed_results = {}

    for seed in seeds:
        output_dir = output_base / 'data' / f'{display_name}_{condition}' / f'seed_{seed}'

        if not output_dir.exists():
            continue

        compounds_path = output_dir / 'compounds_final.csv'
        cycle_metrics_path = output_dir / 'cycle_metrics.csv'

        if not (compounds_path.exists() and cycle_metrics_path.exists()):
            continue

        try:
            compounds_df = pl.read_csv(compounds_path, comment_prefix='#')
            cycle_metrics_df = pl.read_csv(cycle_metrics_path, comment_prefix='#')
            cycle_metrics = cycle_metrics_df.to_dicts()

            seed_results[seed] = {
                'compounds_df': compounds_df,
                'cycle_metrics': cycle_metrics,
                'elapsed_time': None
            }
        except Exception as e:
            print(f"Warning: Could not load results for seed {seed}: {e}")
            continue

    return seed_results

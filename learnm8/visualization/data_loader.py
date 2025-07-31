"""Simple data loading utilities for A/B testing visualizations."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def load_cycle_data(csv_file: Path) -> pd.DataFrame:
    """Load cycle-by-cycle data from parameter sweep results.
    
    Args:
        csv_file: Path to parameter_sweep_cycle_by_cycle.csv
        
    Returns:
        DataFrame with cycle-by-cycle metrics
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not csv_file.exists():
        raise FileNotFoundError(f"Cycle data file not found: {csv_file}")
    
    # Read CSV and handle empty strings
    df = pd.read_csv(csv_file, na_values=['', ' ', 'nan', 'NaN'], keep_default_na=True)
    
    # Clean up column names - handle double suffix issues
    column_mapping = {}
    for col in df.columns:
        # Fix double suffix issues like 'uncertainty_mean_mean' -> 'uncertainty_mean'
        if col.endswith('_mean_mean'):
            new_col = col.replace('_mean_mean', '_mean')
            column_mapping[col] = new_col
        elif col.endswith('_std_std'):
            new_col = col.replace('_std_std', '_std')
            column_mapping[col] = new_col
        # Handle other potential issues like 'prediction_std_std' -> 'prediction_std'
        elif col.endswith('_std_mean'):
            # This appears to be mislabeled - should be just '_std'
            new_col = col.replace('_std_mean', '_std')
            column_mapping[col] = new_col
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"Cleaned up {len(column_mapping)} column names with suffix issues")
    
    # Convert numeric columns properly
    numeric_columns = []
    string_columns = ['experiment_id', 'experiment_name', 'learner_type', 'initial_strategy', 
                     'selection_strategy', 'custom_cycle_spec', 'pruning_strategy_mean', 'pruning_strategy_std',
                     'strategy_mean', 'strategy_std']
    
    for col in df.columns:
        if col not in string_columns:
            # Try to convert to numeric, keeping NaN for non-convertible values
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_columns.append(col)
            except:
                pass
    
    print(f"Converted {len(numeric_columns)} columns to numeric type")
    
    # Check for required columns
    required_cols = ['experiment_id', 'cycle', 'learner_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate percent explored if not present
    if 'percent_explored' not in df.columns:
        if 'cumulative_labeled_mean' in df.columns and 'original_pool_size_mean' in df.columns:
            df['percent_explored'] = (df['cumulative_labeled_mean'] / df['original_pool_size_mean']) * 100
        elif 'cumulative_labeled_mean' in df.columns and 'remaining_pool_mean' in df.columns:
            # Alternative calculation: cumulative / (cumulative + remaining) * 100
            total_pool = df['cumulative_labeled_mean'] + df['remaining_pool_mean']
            df['percent_explored'] = (df['cumulative_labeled_mean'] / total_pool) * 100
        else:
            # Fallback: use cycle number as proxy (will need manual scaling)
            df['percent_explored'] = df['cycle'] * 1.0
    
    return df


def filter_data(df: pd.DataFrame, 
                learner: Optional[str] = None,
                acquisition: Optional[str] = None, 
                batch_size: Optional[float] = None,
                initial_strategy: Optional[str] = None) -> pd.DataFrame:
    """Filter data by experimental parameters.
    
    Args:
        df: Input DataFrame
        learner: Filter by learner type
        acquisition: Filter by acquisition strategy  
        batch_size: Filter by batch size fraction
        initial_strategy: Filter by initial strategy
        
    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()
    
    if learner is not None:
        filtered = filtered[filtered['learner_type'] == learner]
        
    if acquisition is not None:
        # Try multiple column names for acquisition strategy
        # Priority: custom_cycle_spec > selection_strategy > acquisition_strategy
        if 'custom_cycle_spec' in filtered.columns:
            # For custom_cycle_spec, match if it contains the acquisition strategy
            mask = filtered['custom_cycle_spec'].str.contains(acquisition, na=False, case=False)
            if mask.any():
                filtered = filtered[mask]
            elif 'selection_strategy' in filtered.columns:
                filtered = filtered[filtered['selection_strategy'] == acquisition]
            elif 'acquisition_strategy' in filtered.columns:
                filtered = filtered[filtered['acquisition_strategy'] == acquisition]
        elif 'selection_strategy' in filtered.columns:
            filtered = filtered[filtered['selection_strategy'] == acquisition]
        elif 'acquisition_strategy' in filtered.columns:
            filtered = filtered[filtered['acquisition_strategy'] == acquisition]
            
    if batch_size is not None:
        if 'batch_size_fraction' in filtered.columns:
            filtered = filtered[filtered['batch_size_fraction'] == batch_size]
        elif 'batch_fraction_mean' in filtered.columns:
            filtered = filtered[filtered['batch_fraction_mean'] == batch_size]
            
    if initial_strategy is not None:
        if 'initial_strategy' in filtered.columns:
            filtered = filtered[filtered['initial_strategy'] == initial_strategy]
    
    return filtered


def get_available_parameters(df: pd.DataFrame) -> Dict[str, List]:
    """Get lists of available parameter values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with parameter names and their unique values
    """
    params = {}
    
    # Core parameters
    if 'learner_type' in df.columns:
        params['learners'] = sorted(df['learner_type'].unique())
        
    # Acquisition strategies - check multiple possible column names
    # Priority: custom_cycle_spec > selection_strategy > acquisition_strategy
    acquisitions = []
    if 'custom_cycle_spec' in df.columns:
        custom_specs = [s for s in df['custom_cycle_spec'].unique() if pd.notna(s) and s != '']
        if custom_specs:
            acquisitions = sorted(custom_specs)
    
    if not acquisitions:
        if 'selection_strategy' in df.columns:
            acquisitions = sorted([s for s in df['selection_strategy'].unique() if pd.notna(s)])
        elif 'acquisition_strategy' in df.columns:
            acquisitions = sorted([s for s in df['acquisition_strategy'].unique() if pd.notna(s)])
    
    params['acquisitions'] = acquisitions
        
    if 'batch_size_fraction' in df.columns:
        params['batch_sizes'] = sorted(df['batch_size_fraction'].unique())
    elif 'batch_fraction_mean' in df.columns:
        params['batch_sizes'] = sorted(df['batch_fraction_mean'].unique())
    else:
        params['batch_sizes'] = []
        
    if 'initial_strategy' in df.columns:
        params['initial_strategies'] = sorted(df['initial_strategy'].unique())
    else:
        params['initial_strategies'] = []
    
    return params


def aggregate_by_cycle(df: pd.DataFrame, group_by: str, metric: str) -> pd.DataFrame:
    """Aggregate metric values by cycle for plotting.
    
    Args:
        df: Input DataFrame
        group_by: Column to group by (e.g., 'learner_type')
        metric: Metric column to aggregate (should end with '_mean')
        
    Returns:
        DataFrame with aggregated statistics
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in data")
    
    # Find corresponding std column
    std_metric = metric.replace('_mean', '_std')
    
    if std_metric in df.columns:
        # Use existing std column from CSV
        grouped = df.groupby([group_by, 'cycle', 'percent_explored']).agg({
            metric: 'mean',
            std_metric: 'mean',  # Average the std values across experiments
            'experiment_id': 'count'
        }).reset_index()
        
        grouped = grouped.rename(columns={
            metric: 'mean',
            std_metric: 'std', 
            'experiment_id': 'count'
        })
    else:
        # Fallback: calculate std across experiments
        grouped = df.groupby([group_by, 'cycle', 'percent_explored'])[metric].agg([
            'mean', 'std', 'count'
        ]).reset_index()
    
    # Handle missing std values
    grouped['std'] = grouped['std'].fillna(0)
    
    # Calculate standard error and confidence intervals
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'].clip(lower=1))
    grouped['ci_lower'] = grouped['mean'] - 1.96 * grouped['sem']
    grouped['ci_upper'] = grouped['mean'] + 1.96 * grouped['sem']
    
    return grouped


def get_final_performance(df: pd.DataFrame, group_by: str, metrics: List[str]) -> pd.DataFrame:
    """Get final cycle performance for each group.
    
    Args:
        df: Input DataFrame
        group_by: Column to group by
        metrics: List of metric columns
        
    Returns:
        DataFrame with final performance values
    """
    # Get the last cycle for each experiment
    final_cycles = df.groupby('experiment_id')['cycle'].transform('max')
    final_data = df[df['cycle'] == final_cycles]
    
    # Aggregate across experiments for each group
    result_data = []
    for group_val in final_data[group_by].unique():
        group_data = final_data[final_data[group_by] == group_val]
        
        row = {group_by: group_val}
        for metric in metrics:
            if metric in group_data.columns:
                values = group_data[metric].dropna()
                if len(values) > 0:
                    row[f'{metric}_mean'] = values.mean()
                    row[f'{metric}_std'] = values.std()
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
        
        result_data.append(row)
    
    return pd.DataFrame(result_data)
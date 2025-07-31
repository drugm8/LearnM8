"""A/B Testing Visualization Functions for LearnM8.

This module provides 8 focused visualization functions that isolate single variables
to enable true A/B comparisons and reveal trends in active learning performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from scipy import stats

from .data_loader import load_cycle_data, filter_data, aggregate_by_cycle, get_final_performance
from .plot_utils import (
    setup_publication_style, get_color, plot_trajectory_with_ci,
    create_subplot_grid, format_axis_labels, create_legend_outside,
    save_plot, create_heatmap
)


def plot_learner_tournament(data_file: Path, output_dir: Path, 
                          acquisition: str = 'greedy', batch_size: float = 0.01,
                          initial_strategy: str = 'random') -> None:
    """Plot 1: Learner Tournament - Create separate plots for each metric.
    
    Fixed: acquisition strategy, batch size, initial strategy
    Variable: learner type
    Conclusion: Identifies which learner performs best for each metric type.
    """
    setup_publication_style()
    
    # Load and filter data
    df = load_cycle_data(data_file)
    df = filter_data(df, acquisition=acquisition, batch_size=batch_size, 
                     initial_strategy=initial_strategy)
    
    if df.empty:
        print(f"Warning: No data found for acquisition={acquisition}, batch_size={batch_size}")
        return
    
    # Define metrics to plot - all key performance metrics
    metrics = [
        ('r2_score_mean', 'R² Score', 'higher_better'),
        ('rmse_mean', 'RMSE', 'lower_better'),
        ('spearman_correlation_mean', 'Spearman Correlation', 'higher_better'),
        ('ef_1_0_mean', 'EF@1%', 'higher_better'),
        ('ef_0_1_mean', 'EF@0.1%', 'higher_better'),
        ('ef_5_0_mean', 'EF@5%', 'higher_better'),
        ('top_1_percent_overlap_mean', 'Top 1% Recovery', 'higher_better'),
        ('top_0_1_percent_overlap_mean', 'Top 0.1% Recovery', 'higher_better'),
        ('top_10_percent_overlap_mean', 'Top 10% Recovery', 'higher_better')
    ]
    
    learners = sorted([l for l in df['learner_type'].unique() if pd.notna(l)])
    
    # Create one plot per metric
    for metric, ylabel, direction in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric {metric} not available, skipping")
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        best_learner = None
        best_final_score = -np.inf if direction == 'higher_better' else np.inf
        
        # Plot each learner
        for learner in learners:
            learner_data = df[df['learner_type'] == learner]
            if learner_data.empty:
                continue
            
            # Clean and filter numeric data
            learner_data = learner_data.copy()
            learner_data = learner_data.dropna(subset=[metric, 'cycle'])
            
            # Ensure cycle is numeric
            learner_data['cycle'] = pd.to_numeric(learner_data['cycle'], errors='coerce')
            learner_data = learner_data.dropna(subset=['cycle'])
            
            if learner_data.empty:
                continue
                
            # Aggregate by cycle
            agg_data = aggregate_by_cycle(learner_data, 'learner_type', metric)
            if agg_data.empty:
                continue
            
            # Check if this is the best learner (by final performance)
            final_score = agg_data['mean'].iloc[-1] if len(agg_data) > 0 else (
                -np.inf if direction == 'higher_better' else np.inf)
            
            is_best = False
            if direction == 'higher_better' and pd.notna(final_score) and final_score > best_final_score:
                best_final_score = final_score
                best_learner = learner
                is_best = True
            elif direction == 'lower_better' and pd.notna(final_score) and final_score < best_final_score:
                best_final_score = final_score
                best_learner = learner
                is_best = True
                
            # Plot trajectory (make best learner line thicker)
            color = get_color(learner, 'learner')
            linewidth = 3 if is_best else 2
            plot_trajectory_with_ci(
                ax, agg_data['percent_explored'], agg_data['mean'], 
                agg_data['std'], learner, color
            )
        
        format_axis_labels(ax, '% Explored', ylabel, 
                          f'Learner Tournament - {ylabel}')
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Save with metric-specific filename
        metric_name = metric.replace('_mean', '').replace('_', '-')
        output_path = output_dir / f'learner_tournament_{metric_name}'
        save_plot(fig, str(output_path))
        print(f"Saved learner tournament {ylabel} plot to {output_path}.png")


def plot_acquisition_tournament(data_file: Path, output_dir: Path,
                              batch_size: float = 0.01, 
                              initial_strategy: str = 'random') -> None:
    """Plot 2: Acquisition Strategy Tournament - Create separate plots for each metric.
    
    Fixed: batch size, initial strategy
    Variable: acquisition strategy
    Conclusion: Reveals which acquisition strategies work best with which learners.
    """
    setup_publication_style()
    
    # Load and filter data
    df = load_cycle_data(data_file)
    df = filter_data(df, batch_size=batch_size, initial_strategy=initial_strategy)
    
    if df.empty:
        print(f"Warning: No data found for batch_size={batch_size}")
        return
    
    learners = sorted([l for l in df['learner_type'].unique() if pd.notna(l)])
    
    # Determine which column to use for acquisition strategy
    # Priority: custom_cycle_spec (if non-empty) > selection_strategy > acquisition_strategy
    acquisition_col = None
    strategies = []
    
    if 'custom_cycle_spec' in df.columns:
        # Use custom_cycle_spec when it's not empty/null
        custom_specs = [s for s in df['custom_cycle_spec'].unique() if pd.notna(s) and s != '']
        if custom_specs:
            acquisition_col = 'custom_cycle_spec'
            strategies = sorted(custom_specs)
    
    # Fallback to standard strategy columns if no custom specs
    if not strategies:
        if 'selection_strategy' in df.columns:
            acquisition_col = 'selection_strategy'
            strategies = sorted([s for s in df['selection_strategy'].unique() if pd.notna(s)])
        elif 'acquisition_strategy' in df.columns:
            acquisition_col = 'acquisition_strategy'
            strategies = sorted([s for s in df['acquisition_strategy'].unique() if pd.notna(s)])
    
    if not acquisition_col or not strategies:
        print("Warning: No acquisition strategy column found")
        return
    
    # Define key metrics to show for acquisition tournament
    key_metrics = [
        ('ef_1_0_mean', 'EF@1%'),
        ('r2_score_mean', 'R² Score'),
        ('top_1_percent_overlap_mean', 'Top 1% Recovery'),
        ('spearman_correlation_mean', 'Spearman Correlation')
    ]
    
    # Create one plot per metric, with subplots for each learner
    for metric, ylabel in key_metrics:
        if metric not in df.columns:
            print(f"Warning: Metric {metric} not available, skipping")
            continue
            
        # Create subplot grid for learners
        fig, axes = create_subplot_grid(len(learners), max_cols=3)
        
        for i, learner in enumerate(learners):
            if i >= len(axes):
                break
                
            ax = axes[i]
            learner_data = df[df['learner_type'] == learner]
            
            if learner_data.empty:
                ax.text(0.5, 0.5, f'{learner}\nNo data', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            best_strategy = None
            best_final_score = -np.inf
            
            # Plot each strategy
            for strategy in strategies:
                strategy_data = learner_data[learner_data[acquisition_col] == strategy]
                if strategy_data.empty:
                    continue
                
                # Clean and filter numeric data
                strategy_data = strategy_data.copy()
                strategy_data = strategy_data.dropna(subset=[metric, 'cycle'])
                
                # Ensure cycle is numeric
                strategy_data['cycle'] = pd.to_numeric(strategy_data['cycle'], errors='coerce')
                strategy_data = strategy_data.dropna(subset=['cycle'])
                
                if strategy_data.empty:
                    continue
                    
                # Aggregate by cycle
                agg_data = aggregate_by_cycle(strategy_data, acquisition_col, metric)
                if agg_data.empty:
                    continue
                
                # Check if this is the best strategy (by final performance)
                final_score = agg_data['mean'].iloc[-1] if len(agg_data) > 0 else -np.inf
                if pd.notna(final_score) and final_score > best_final_score:
                    best_final_score = final_score
                    best_strategy = strategy
                
                # Plot trajectory (make best strategy line thicker)
                color = get_color(strategy, 'strategy')
                linewidth = 3 if strategy == best_strategy else 2
                plot_trajectory_with_ci(
                    ax, agg_data['percent_explored'], agg_data['mean'], 
                    agg_data['std'], strategy, color
                )
            
            format_axis_labels(ax, '% Explored', ylabel, learner)
            
            # Add legend only to first subplot
            if i == 0:
                create_legend_outside(ax)
        
        plt.suptitle(f'Acquisition Strategy Tournament - {ylabel} (Batch: {batch_size})')
        plt.tight_layout()
        
        # Save with metric-specific filename
        metric_name = metric.replace('_mean', '').replace('_', '-')
        output_path = output_dir / f'acquisition_tournament_{metric_name}'
        save_plot(fig, str(output_path))
        print(f"Saved acquisition tournament {ylabel} plot to {output_path}.png")


def plot_batch_size_impact(data_file: Path, output_dir: Path,
                          learner: str = 'rf_ensemble',
                          acquisition: str = 'greedy') -> None:
    """Plot 3: Batch Size Impact - Compare different batch sizes.
    
    Fixed: learner, acquisition strategy
    Variable: batch size
    Conclusion: Determines optimal batch size for balancing performance with efficiency.
    """
    setup_publication_style()
    
    # Load and filter data
    df = load_cycle_data(data_file)
    df = filter_data(df, learner=learner, acquisition=acquisition)
    
    if df.empty:
        print(f"Warning: No data found for learner={learner}, acquisition={acquisition}")
        return
    
    # Check for batch size column
    batch_col = None
    if 'batch_size_fraction' in df.columns:
        batch_col = 'batch_size_fraction'
    elif 'batch_fraction_mean' in df.columns:
        batch_col = 'batch_fraction_mean'
    
    if batch_col is None:
        print("Warning: No batch size column found")
        return
    
    batch_sizes = sorted([b for b in df[batch_col].unique() if pd.notna(b)])
    metric = 'top_1_percent_overlap_mean'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each batch size
    for batch_size in batch_sizes:
        batch_data = df[df[batch_col] == batch_size]
        if batch_data.empty or metric not in batch_data.columns:
            continue
            
        # Aggregate by cycle  
        agg_data = aggregate_by_cycle(batch_data, batch_col, metric)
        if agg_data.empty:
            continue
        
        # Use different line styles for different batch sizes
        from .plot_utils import get_line_style
        line_style = get_line_style(batch_size)
        
        plot_trajectory_with_ci(
            ax, agg_data['percent_explored'], agg_data['mean'], 
            agg_data['std'], f'Batch: {batch_size}', 
            get_color('default', 'learner'), line_style
        )
    
    format_axis_labels(ax, '% Explored', 'Top 1% Recovery', 
                      f'Batch Size Impact ({learner}, {acquisition})')
    ax.legend()
    
    plt.tight_layout()
    
    output_path = output_dir / 'batch_size_impact'
    save_plot(fig, str(output_path))
    print(f"Saved batch size impact plot to {output_path}.png")


def plot_custom_cycle_comparison(data_file: Path, output_dir: Path,
                               learner: str = 'gp') -> None:
    """Plot 4: Custom Cycle Strategy Comparison.
    
    Fixed: learner
    Variable: cycle schedules
    Conclusion: Shows whether strategic transitions outperform single-strategy approaches.
    """
    setup_publication_style()
    
    # Load data
    df = load_cycle_data(data_file)
    df = filter_data(df, learner=learner)
    
    if df.empty:
        print(f"Warning: No data found for learner={learner}")
        return
    
    # Group by experiment type or custom cycle specification
    if 'custom_cycle_spec' in df.columns:
        cycle_specs = df['custom_cycle_spec'].unique()
        cycle_specs = [spec for spec in cycle_specs if pd.notna(spec) and spec != '']
    else:
        print("Warning: No custom cycle specifications found")
        return
    
    metrics = ['r2_score_mean', 'avg_score_selected_mean']
    metric_labels = ['R² Score', 'Average Score']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'Metric {metric}\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot each cycle specification
        for j, spec in enumerate(cycle_specs[:5]):  # Limit to 5 for readability
            spec_data = df[df['custom_cycle_spec'] == spec]
            if spec_data.empty:
                continue
                
            # Aggregate by cycle
            agg_data = aggregate_by_cycle(spec_data, 'custom_cycle_spec', metric)
            if agg_data.empty:
                continue
            
            color = plt.cm.tab10(j)
            plot_trajectory_with_ci(
                ax, agg_data['percent_explored'], agg_data['mean'], 
                agg_data['std'], spec[:20] + '...', color  # Truncate long specs
            )
        
        format_axis_labels(ax, '% Explored', ylabel, f'{ylabel} by Cycle Strategy')
        
        if i == 0:
            ax.legend(loc='best')
    
    plt.suptitle(f'Custom Cycle Strategy Comparison ({learner})')
    plt.tight_layout()
    
    output_path = output_dir / 'custom_cycle_comparison'
    save_plot(fig, str(output_path))
    print(f"Saved custom cycle comparison plot to {output_path}.png")


def plot_uncertainty_model_tournament(data_file: Path, output_dir: Path,
                                    acquisition: str = 'ucb',
                                    batch_size: float = 0.01) -> None:
    """Plot 5: Uncertainty-Enabled Model Tournament.
    
    Fixed: acquisition strategy (uncertainty-based), batch size
    Variable: uncertainty-enabled learners
    Conclusion: Shows how well uncertainty estimates correlate with performance.
    """
    setup_publication_style()
    
    # Load and filter data
    df = load_cycle_data(data_file)
    df = filter_data(df, acquisition=acquisition, batch_size=batch_size)
    
    if df.empty:
        print(f"Warning: No data found for acquisition={acquisition}")
        return
    
    # Filter for uncertainty-enabled models
    uncertainty_models = ['gp', 'mc_dropout', 'ensemble', 'rf_ensemble', 'mixed_ensemble']
    df = df[df['learner_type'].isin(uncertainty_models)]
    
    if df.empty:
        print("Warning: No uncertainty-enabled models found")
        return
    
    metrics = [
        ('uncertainty_mean_mean', 'Mean Uncertainty'),
        ('rmse_mean', 'RMSE'), 
        ('ef_1_0_mean', 'EF@1%')
    ]
    
    fig, axes = create_subplot_grid(len(metrics), max_cols=3)
    learners = sorted(df['learner_type'].unique())
    
    for i, (metric, ylabel) in enumerate(metrics):
        ax = axes[i]
        
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'Metric {metric}\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot each learner
        for learner in learners:
            learner_data = df[df['learner_type'] == learner]
            if learner_data.empty:
                continue
                
            # Aggregate by cycle
            agg_data = aggregate_by_cycle(learner_data, 'learner_type', metric)
            if agg_data.empty:
                continue
            
            color = get_color(learner, 'learner')
            plot_trajectory_with_ci(
                ax, agg_data['percent_explored'], agg_data['mean'], 
                agg_data['std'], learner, color
            )
        
        format_axis_labels(ax, '% Explored', ylabel, f'{ylabel} by Uncertainty Model')
        
        if i == 0:
            create_legend_outside(ax)
    
    plt.suptitle(f'Uncertainty Model Tournament (Strategy: {acquisition})')
    plt.tight_layout()
    
    output_path = output_dir / 'uncertainty_tournament'
    save_plot(fig, str(output_path))
    print(f"Saved uncertainty tournament plot to {output_path}.png")


def plot_initial_strategy_impact(data_file: Path, output_dir: Path,
                               learner: str = 'ensemble',
                               acquisition: str = 'greedy') -> None:
    """Plot 6: Initial Strategy Impact Analysis.
    
    Fixed: learner, main acquisition strategy
    Variable: initial strategy
    Conclusion: Reveals whether diverse initial sampling provides lasting benefits.
    """
    setup_publication_style()
    
    # Load and filter data
    df = load_cycle_data(data_file)
    df = filter_data(df, learner=learner, acquisition=acquisition)
    
    if df.empty or 'initial_strategy' not in df.columns:
        print(f"Warning: No data or initial_strategy column found")
        return
    
    initial_strategies = sorted(df['initial_strategy'].unique())
    metrics = ['intra_batch_diversity_mean', 'top_10_percent_overlap_mean']
    metric_labels = ['Intra-batch Diversity', 'Top 10% Recovery']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'Metric {metric}\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot each initial strategy
        for strategy in initial_strategies:
            strategy_data = df[df['initial_strategy'] == strategy]
            if strategy_data.empty:
                continue
                
            # Aggregate by cycle
            agg_data = aggregate_by_cycle(strategy_data, 'initial_strategy', metric)
            if agg_data.empty:
                continue
            
            color = get_color(strategy, 'strategy')
            plot_trajectory_with_ci(
                ax, agg_data['percent_explored'], agg_data['mean'], 
                agg_data['std'], strategy, color
            )
        
        format_axis_labels(ax, '% Explored', ylabel, f'{ylabel} by Initial Strategy')
        
        if i == 0:
            ax.legend()
    
    plt.suptitle(f'Initial Strategy Impact ({learner}, {acquisition})')
    plt.tight_layout()
    
    output_path = output_dir / 'initial_strategy_impact'
    save_plot(fig, str(output_path))
    print(f"Saved initial strategy impact plot to {output_path}.png")


def plot_performance_matrix(data_file: Path, output_dir: Path) -> None:
    """Plot 7: Performance Matrices - One matrix per key metric.
    
    Variable: all learner-strategy combinations  
    Conclusion: At-a-glance identification of best combinations for each metric.
    """
    setup_publication_style()
    
    # Load data
    df = load_cycle_data(data_file)
    
    # Get final performance for each experiment
    final_data = []
    for exp_id, exp_data in df.groupby('experiment_id'):
        if exp_data.empty:
            continue
        final_row = exp_data.iloc[-1]  # Get last cycle
        final_data.append(final_row)
    
    if not final_data:
        print("Warning: No final performance data found")
        return
        
    final_df = pd.DataFrame(final_data)
    
    # Define key metrics for individual matrices
    key_metrics = [
        ('r2_score_mean', 'R² Score', 'RdYlGn'),
        ('rmse_mean', 'RMSE', 'RdYlGn_r'), 
        ('ef_1_0_mean', 'EF@1%', 'RdYlGn'),
        ('top_1_percent_overlap_mean', 'Top 1% Recovery', 'RdYlGn'),
        ('spearman_correlation_mean', 'Spearman Correlation', 'RdYlGn')
    ]
    
    # Determine which column to use for acquisition strategy
    # Priority: custom_cycle_spec (if non-empty) > selection_strategy > acquisition_strategy
    acquisition_col = None
    
    if 'custom_cycle_spec' in final_df.columns:
        # Use custom_cycle_spec when it's not empty/null
        custom_specs = [s for s in final_df['custom_cycle_spec'].unique() if pd.notna(s) and s != '']
        if custom_specs:
            acquisition_col = 'custom_cycle_spec'
    
    # Fallback to standard strategy columns if no custom specs
    if not acquisition_col:
        if 'selection_strategy' in final_df.columns:
            acquisition_col = 'selection_strategy'
        elif 'acquisition_strategy' in final_df.columns:
            acquisition_col = 'acquisition_strategy'
    
    if not acquisition_col:
        print("Warning: No acquisition strategy column found")
        return
    
    # Create separate matrix for each metric
    for metric, title, cmap in key_metrics:
        if metric not in final_df.columns:
            print(f"Warning: Metric {metric} not found, skipping")
            continue
            
        # Create pivot table for this metric
        pivot_data = final_df.pivot_table(
            values=metric,
            index='learner_type',
            columns=acquisition_col, 
            aggfunc='mean'
        )
        
        if pivot_data.empty:
            continue
            
        # Create heatmap
        learners = list(pivot_data.index)
        strategies = list(pivot_data.columns)
        
        fig = create_heatmap(
            pivot_data.values, strategies, learners,
            f'Performance Matrix - {title}', 
            cmap=cmap, annot=True, fmt='.3f'
        )
        
        # Save with metric-specific filename
        metric_name = metric.replace('_mean', '').replace('_', '-')
        output_path = output_dir / f'performance_matrix_{metric_name}'
        save_plot(fig, str(output_path))
        print(f"Saved {title} performance matrix to {output_path}.png")


def plot_diversity_performance_tradeoff(data_file: Path, output_dir: Path,
                                      learner: str = 'rf_ensemble') -> None:
    """Plot 8: Diversity vs Performance Trade-off.
    
    Fixed: learner
    Variable: acquisition strategies
    Conclusion: Shows which strategies balance exploration and exploitation.
    """
    setup_publication_style()
    
    # Load and filter data
    df = load_cycle_data(data_file)
    df = filter_data(df, learner=learner)
    
    if df.empty:
        print(f"Warning: No data found for learner={learner}")
        return
    
    # Check for required metrics
    diversity_metric = 'intra_batch_diversity_mean'
    performance_metric = 'top_1_percent_overlap_mean'
    
    if diversity_metric not in df.columns or performance_metric not in df.columns:
        print(f"Warning: Required metrics not found")
        return
    
    # Determine which column to use for acquisition strategy
    # Priority: custom_cycle_spec (if non-empty) > selection_strategy > acquisition_strategy
    acquisition_col = None
    
    if 'custom_cycle_spec' in df.columns:
        # Use custom_cycle_spec when it's not empty/null
        custom_specs = [s for s in df['custom_cycle_spec'].unique() if pd.notna(s) and s != '']
        if custom_specs:
            acquisition_col = 'custom_cycle_spec'
    
    # Fallback to standard strategy columns if no custom specs
    if not acquisition_col:
        if 'selection_strategy' in df.columns:
            acquisition_col = 'selection_strategy'
        elif 'acquisition_strategy' in df.columns:
            acquisition_col = 'acquisition_strategy'
    
    if not acquisition_col:
        print("Warning: No acquisition strategy column found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    strategies = sorted([s for s in df[acquisition_col].unique() if pd.notna(s)])
    
    # Plot trajectory for each strategy
    for strategy in strategies:
        strategy_data = df[df[acquisition_col] == strategy]
        if strategy_data.empty:
            continue
        
        # Clean and filter numeric data
        strategy_data = strategy_data.copy()
        strategy_data = strategy_data.dropna(subset=[diversity_metric, performance_metric, 'cycle'])
        
        # Ensure cycle is numeric
        strategy_data['cycle'] = pd.to_numeric(strategy_data['cycle'], errors='coerce')
        strategy_data = strategy_data.dropna(subset=['cycle'])
        
        if strategy_data.empty:
            continue
        
        # Sort by cycle for trajectory
        strategy_data = strategy_data.sort_values('cycle')
        
        x_data = strategy_data[diversity_metric]
        y_data = strategy_data[performance_metric]
        
        if len(x_data) == 0 or len(y_data) == 0:
            continue
        
        color = get_color(strategy, 'strategy')
        
        # Plot points colored by cycle (early cycles lighter, later darker)
        cycles = strategy_data['cycle']
        scatter = ax.scatter(x_data, y_data, c=cycles, cmap='viridis', 
                           alpha=0.7, s=50, label=strategy)
        
        # Connect points with lines to show trajectory
        ax.plot(x_data, y_data, color=color, alpha=0.5, linewidth=1)
        
        # Add arrow to show direction
        if len(x_data) > 1:
            ax.annotate('', xy=(x_data.iloc[-1], y_data.iloc[-1]), 
                       xytext=(x_data.iloc[-2], y_data.iloc[-2]),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    format_axis_labels(ax, 'Intra-batch Diversity', 'Top 1% Recovery',
                      f'Diversity vs Performance Trade-off ({learner})')
    
    # Add colorbar for cycle progression
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cycle Number', rotation=270, labelpad=20)
    
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    plt.tight_layout()
    
    output_path = output_dir / 'diversity_performance_tradeoff'
    save_plot(fig, str(output_path))
    print(f"Saved diversity-performance trade-off plot to {output_path}.png")


def generate_all_ab_plots(data_file: Path, output_dir: Path) -> None:
    """Generate all 8 A/B testing plots.
    
    Args:
        data_file: Path to parameter_sweep_cycle_by_cycle.csv
        output_dir: Directory to save plots
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating A/B testing plots from {data_file}")
    print(f"Output directory: {output_dir}")
    
    # Generate all plots
    try:
        plot_learner_tournament(data_file, output_dir)
    except Exception as e:
        print(f"Error in learner tournament plot: {e}")
    
    try:
        plot_acquisition_tournament(data_file, output_dir)
    except Exception as e:
        print(f"Error in acquisition tournament plot: {e}")
    
    try:
        plot_batch_size_impact(data_file, output_dir)
    except Exception as e:
        print(f"Error in batch size impact plot: {e}")
    
    try:
        plot_custom_cycle_comparison(data_file, output_dir)
    except Exception as e:
        print(f"Error in custom cycle comparison plot: {e}")
    
    try:
        plot_uncertainty_model_tournament(data_file, output_dir)
    except Exception as e:
        print(f"Error in uncertainty tournament plot: {e}")
    
    try:
        plot_initial_strategy_impact(data_file, output_dir)
    except Exception as e:
        print(f"Error in initial strategy impact plot: {e}")
    
    try:
        plot_performance_matrix(data_file, output_dir)
    except Exception as e:
        print(f"Error in performance matrix plot: {e}")
    
    try:
        plot_diversity_performance_tradeoff(data_file, output_dir)
    except Exception as e:
        print(f"Error in diversity-performance trade-off plot: {e}")
    
    print("A/B testing visualization generation complete!")


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) != 3:
        print("Usage: python ab_testing_plots.py <data_file.csv> <output_dir>")
        sys.exit(1)
    
    data_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    generate_all_ab_plots(data_file, output_dir)
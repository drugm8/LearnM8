from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from datetime import datetime
from matplotlib.patches import Polygon


def create_top_k_heatmap(
    mean_matrix: pl.DataFrame,
    std_matrix: pl.DataFrame,
    k_label: str,
    metric_col: str,
    ax: plt.Axes
) -> plt.Axes:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'red_blue',
        ['#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']
    )

    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=75, vmax=100)

    mean_pd = mean_matrix.to_pandas()
    mask = mean_pd.isna()

    hm = sns.heatmap(
        mean_pd,
        annot=False,
        cmap=cmap,
        norm=norm,
        mask=mask,
        cbar_kws={'shrink': 0.8, 'label': 'Discovery Rate (%)'},
        linewidths=0.5,
        linecolor='white',
        square=False,
        vmin=0,
        vmax=100,
        ax=ax
    )

    for i in range(mean_matrix.height):
        for j in range(len(mean_matrix.columns)):
            mean_val = mean_matrix.row(i)[j]
            std_val = std_matrix.row(i)[j]
            if mean_val is not None and std_val is not None:
                text = f'{mean_val:.1f}±{std_val:.1f}'
                ax.text(j + 0.5, i + 0.5, text,
                       ha='center', va='center',
                       fontsize=8, fontweight='bold',
                       color='black' if mean_val > 50 else 'white')

    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Discovery Rate (%)', size=11)

    ax.set_title(f'{k_label} Discovery Rate (mean ± std)',
                 fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('Featurizer', fontsize=13, labelpad=8)
    ax.set_ylabel('Learner', fontsize=13, labelpad=8)

    ax.tick_params(axis='both', labelsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    return ax


def create_all_heatmaps(
    results_df: pl.DataFrame,
    output_dir: Path
) -> Path:
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    heatmap_configs = [
        ('top_0_1_pct_discovery', 'Top 0.1%'),
        ('top_1_pct_discovery', 'Top 1%'),
        ('top_10_discovery', 'Top 10'),
        ('top_100_discovery', 'Top 100')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=300)
    axes = axes.flatten()

    fig.suptitle('Learner-Featurizer Performance Matrix: Discovery Rates',
                 fontsize=24, fontweight='bold', y=0.995)

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.96,
        bottom=0.06,
        wspace=0.25,
        hspace=0.28
    )

    for idx, (metric_col, k_label) in enumerate(heatmap_configs):
        mean_col = f'{metric_col}_mean'
        std_col = f'{metric_col}_std'

        mean_pivot = results_df.pivot(
            values=mean_col,
            index='learner',
            columns='featurizer',
            aggregate_function='first'
        )

        std_pivot = results_df.pivot(
            values=std_col,
            index='learner',
            columns='featurizer',
            aggregate_function='first'
        )

        desired_order = ['none', 'morgan', 'maccs', 'ecfp6', 'descriptors', 'morgan_feat']
        existing_cols = [col for col in desired_order if col in mean_pivot.columns]
        mean_pivot = mean_pivot.select(existing_cols)
        std_pivot = std_pivot.select(existing_cols)

        create_top_k_heatmap(mean_pivot, std_pivot, k_label, metric_col, axes[idx])

    output_path = plots_dir / 'heatmap_combined.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Created combined heatmap: {output_path.name}")

    return output_path


def create_learner_cycle_plot(
    learner_results: Dict[str, Dict],
    output_path: Path,
    dpi: int = 300
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metrics = [
        ('top_0_1_pct_discovery', 'Top 0.1% Discovery Rate'),
        ('top_1_pct_discovery', 'Top 1% Discovery Rate'),
        ('top_10_discovery', 'Top 10 Discovery Rate'),
        ('top_100_discovery', 'Top 100 Discovery Rate')
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    featurizer_colors = {
        'none': colors[0],
        'morgan': colors[1],
        'maccs': colors[2],
        'ecfp6': colors[3],
        'descriptors': colors[4],
        'morgan_feat': colors[5]
    }

    for ax_idx, (metric_col, title) in enumerate(metrics):
        ax = axes[ax_idx]

        for learner_name, featurizer_results in sorted(learner_results.items()):
            for featurizer_name, result_data in sorted(featurizer_results.items()):
                aggregated = result_data['aggregated']
                cycle_metrics_mean = aggregated['cycle_metrics_mean']
                cycle_metrics_std = aggregated['cycle_metrics_std']

                cycles = [m['cycle'] for m in cycle_metrics_mean]
                mean_values = [m.get(metric_col, 0) for m in cycle_metrics_mean]
                std_values = [m.get(metric_col, 0) for m in cycle_metrics_std]

                label = f"{learner_name} ({featurizer_name})"
                ax.errorbar(cycles, mean_values, yerr=std_values,
                           marker='o', linewidth=1.5, label=label,
                           color=featurizer_colors.get(featurizer_name, 'gray'),
                           markersize=3, alpha=0.7, linestyle='-',
                           capsize=2, capthick=1)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Cycle', fontsize=10)
        ax.set_ylabel('Discovery Rate (%)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=100)

        if ax_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     fontsize=6, framealpha=0.9, ncol=1)

    plt.suptitle('Learner-Featurizer Performance Comparison: Greedy Acquisition',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"✓ Created cycle comparison plot: {output_path.name}")

    return output_path


def create_summary_report(
    results_df: pl.DataFrame,
    all_results: Dict[Tuple[str, str], Dict],
    heatmap_path: Path,
    cycle_plot_path: Path,
    output_dir: Path,
    config: Dict
) -> Path:
    output_dir = Path(output_dir)
    report_path = output_dir / 'validation_report.md'

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = [
        "# Learner-Featurizer Matrix Validation Report",
        "",
        f"**Generated:** {timestamp}",
        "",
        "## Experiment Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Dataset | {config['dataset_name']} |",
        f"| Compounds | {config.get('n_compounds', 'N/A')} |",
        f"| Cycles | {config['n_cycles']} |",
        f"| Batch Fraction | {config['batch_fraction']} (≈{config.get('batch_size', 'N/A')} compounds/cycle) |",
        f"| Acquisition | {config['acquisition']} |",
        f"| Random State | {config['random_state']} |",
        f"| Total Experiments | {len(results_df)} |",
        "",
        "## Summary Statistics",
        "",
        "### Best Performing Combinations",
        ""
    ]

    for metric in ['top_0_1_pct_discovery', 'top_1_pct_discovery', 'top_10_discovery', 'top_100_discovery']:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        best_idx = results_df[mean_col].idxmax()
        best_row = results_df.loc[best_idx]
        metric_name = metric.replace('_', ' ').title()
        lines.append(f"**{metric_name}:** {best_row['learner']} + {best_row['featurizer']} "
                    f"({best_row[mean_col]:.1f}±{best_row[std_col]:.1f}%)")

    lines.extend([
        "",
        "### Performance by Learner (Mean ± Std)",
        "",
        results_df.group_by('learner').agg([
            pl.col('top_0_1_pct_discovery_mean').mean().round(1).alias('top_0_1_pct_discovery_mean'),
            pl.col('top_1_pct_discovery_mean').mean().round(1).alias('top_1_pct_discovery_mean'),
            pl.col('top_10_discovery_mean').mean().round(1).alias('top_10_discovery_mean'),
            pl.col('top_100_discovery_mean').mean().round(1).alias('top_100_discovery_mean')
        ]).to_pandas().to_markdown(),
        "",
        "### Performance by Featurizer (Mean ± Std)",
        "",
        results_df.group_by('featurizer').agg([
            pl.col('top_0_1_pct_discovery_mean').mean().round(1).alias('top_0_1_pct_discovery_mean'),
            pl.col('top_1_pct_discovery_mean').mean().round(1).alias('top_1_pct_discovery_mean'),
            pl.col('top_10_discovery_mean').mean().round(1).alias('top_10_discovery_mean'),
            pl.col('top_100_discovery_mean').mean().round(1).alias('top_100_discovery_mean')
        ]).to_pandas().to_markdown(),
        "",
        "## Visualizations",
        "",
        "### Combined Heatmap: Top-K Discovery Rates",
        "",
        f"![Combined Discovery Rates Heatmap]({heatmap_path.relative_to(output_dir)})",
        ""
    ])

    lines.extend([
        "### Cycle-by-Cycle Performance",
        "",
        f"![Learner-Featurizer Cycle Comparison]({cycle_plot_path.relative_to(output_dir)})",
        "",
        "## Key Observations",
        ""
    ])

    learner_counts = results_df['learner'].nunique()
    featurizer_counts = results_df['featurizer'].nunique()

    lines.extend([
        f"- **Learners tested:** {learner_counts} ({', '.join(sorted(results_df['learner'].unique()))})",
        f"- **Featurizers tested:** {featurizer_counts} ({', '.join(sorted(results_df['featurizer'].unique()))})",
        f"- **Total experiments completed:** {len(results_df)}",
        f"- **Acquisition strategy:** {config['acquisition']}",
        "",
        "## Data Files",
        "",
        "Individual experiment results saved to:",
        "```",
        "validation/reports/learner_featurizer_matrix/data/{learner}_{featurizer}/",
        "  ├── compounds_final.csv",
        "  ├── cycle_metrics.csv",
        "  └── selection_history.csv",
        "```",
        "",
        f"**Report generated:** {timestamp}",
        ""
    ])

    report_path.write_text('\n'.join(lines))
    print(f"✓ Created summary report: {report_path.name}")

    return report_path


def generate_comprehensive_visualizations(
    all_results: Dict[Tuple[str, str], Dict],
    output_dir: Path,
    config: Dict
) -> Dict[str, Path]:
    output_dir = Path(output_dir)

    results_list = []
    for (learner, featurizer), result_data in all_results.items():
        aggregated = result_data['aggregated']
        final_metrics_mean = aggregated['cycle_metrics_mean'][-1]
        final_metrics_std = aggregated['cycle_metrics_std'][-1]

        results_list.append({
            'learner': learner,
            'featurizer': featurizer if featurizer is not None else 'none',
            'top_0_1_pct_discovery_mean': final_metrics_mean.get('top_0_1_pct_discovery', 0),
            'top_0_1_pct_discovery_std': final_metrics_std.get('top_0_1_pct_discovery', 0),
            'top_1_pct_discovery_mean': final_metrics_mean.get('top_1_pct_discovery', 0),
            'top_1_pct_discovery_std': final_metrics_std.get('top_1_pct_discovery', 0),
            'top_10_discovery_mean': final_metrics_mean.get('top_10_discovery', 0),
            'top_10_discovery_std': final_metrics_std.get('top_10_discovery', 0),
            'top_100_discovery_mean': final_metrics_mean.get('top_100_discovery', 0),
            'top_100_discovery_std': final_metrics_std.get('top_100_discovery', 0),
            'unlabeled_spearman_mean': final_metrics_mean.get('unlabeled_spearman_correlation', 0),
            'unlabeled_spearman_std': final_metrics_std.get('unlabeled_spearman_correlation', 0),
            'elapsed_time_mean': aggregated.get('elapsed_time_mean', 0),
            'elapsed_time_std': aggregated.get('elapsed_time_std', 0)
        })

    results_df = pl.DataFrame(results_list)

    heatmap_path = create_all_heatmaps(results_df, output_dir)

    learner_results = {}
    for (learner, featurizer), result_data in all_results.items():
        if learner not in learner_results:
            learner_results[learner] = {}
        featurizer_key = featurizer if featurizer is not None else 'none'
        learner_results[learner][featurizer_key] = result_data

    plots_dir = output_dir / 'plots'
    cycle_plot_path = create_learner_cycle_plot(
        learner_results,
        plots_dir / 'learner_featurizer_cycle_comparison.png'
    )

    report_path = create_summary_report(
        results_df,
        all_results,
        heatmap_path,
        cycle_plot_path,
        output_dir,
        config
    )

    return {
        'heatmap': heatmap_path,
        'cycle_plot': cycle_plot_path,
        'report': report_path
    }


def _format_time_label(seconds: float) -> str:
    """Format time values for axis labels."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def _identify_pareto_frontier(x_values: np.ndarray, y_values: np.ndarray, maximize_y: bool = True) -> np.ndarray:
    """Identify Pareto-optimal points (non-dominated solutions).

    Args:
        x_values: Cost values (minimize)
        y_values: Performance values
        maximize_y: If True, maximize y_values; if False, minimize

    Returns:
        Boolean mask of Pareto-optimal points
    """
    n_points = len(x_values)
    is_pareto = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if not is_pareto[i]:
            continue

        for j in range(n_points):
            if i == j:
                continue

            # Check if j dominates i
            if maximize_y:
                # j dominates i if: x[j] <= x[i] AND y[j] >= y[i] with at least one strict inequality
                if x_values[j] <= x_values[i] and y_values[j] >= y_values[i]:
                    if x_values[j] < x_values[i] or y_values[j] > y_values[i]:
                        is_pareto[i] = False
                        break
            else:
                # j dominates i if: x[j] <= x[i] AND y[j] <= y[i] with at least one strict inequality
                if x_values[j] <= x_values[i] and y_values[j] <= y_values[i]:
                    if x_values[j] < x_values[i] or y_values[j] < y_values[i]:
                        is_pareto[i] = False
                        break

    return is_pareto


def calculate_cumulative_timing(all_results: Dict[Tuple[str, str], Dict]) -> pl.DataFrame:
    """Calculate cumulative timing metrics from cycle results.

    Args:
        all_results: Dictionary mapping (learner, featurizer) → result data

    Returns:
        DataFrame with cumulative timing columns
    """
    timing_data = []

    for (learner, featurizer), result_data in all_results.items():
        aggregated = result_data['aggregated']
        cycle_metrics_mean = aggregated['cycle_metrics_mean']

        # Calculate cumulative timing
        cumulative_training = 0.0
        cumulative_prediction = 0.0
        cumulative_total = 0.0

        for cycle_metric in cycle_metrics_mean:
            cumulative_training += cycle_metric.get('training_time', 0.0)
            cumulative_prediction += cycle_metric.get('prediction_time', 0.0)
            cumulative_total += cycle_metric.get('total_time', 0.0)

        cumulative_training_prediction = cumulative_training + cumulative_prediction

        # Get final discovery rates
        final_metrics = cycle_metrics_mean[-1]

        timing_data.append({
            'learner': learner,
            'featurizer': featurizer if featurizer else 'none',
            'cumulative_training_time': cumulative_training,
            'cumulative_prediction_time': cumulative_prediction,
            'cumulative_total_time': cumulative_total,
            'cumulative_training_prediction_time': cumulative_training_prediction,
            'elapsed_time': aggregated.get('elapsed_time_mean', cumulative_total),
            'top_1_pct_discovery': final_metrics.get('top_1_pct_discovery', 0.0),
            'top_0_1_pct_discovery': final_metrics.get('top_0_1_pct_discovery', 0.0)
        })

    return pl.DataFrame(timing_data)


def create_cost_performance_heatmaps(
    timing_df: pl.DataFrame,
    output_dir: Path,
    performance_metric: str = 'top_1_pct_discovery'
) -> Path:
    """Create 4-panel heatmap showing cost metrics for each learner-featurizer combination.

    Args:
        timing_df: DataFrame with timing and performance data
        output_dir: Directory to save output
        performance_metric: Performance column name (top_1_pct_discovery or top_0_1_pct_discovery)

    Returns:
        Path to saved heatmap image
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    cost_metrics = [
        ('cumulative_training_time', 'Cumulative Training Time (s)'),
        ('cumulative_prediction_time', 'Cumulative Prediction Time (s)'),
        ('elapsed_time', 'Total Elapsed Time (s)'),
        ('cumulative_training_prediction_time', 'Cumulative Training+Prediction Time (s)')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=300)
    axes = axes.flatten()

    metric_label = 'Top 1%' if 'top_1' in performance_metric else 'Top 0.1%'
    fig.suptitle(f'Cost vs Performance Matrix: {metric_label} Discovery Rate',
                 fontsize=24, fontweight='bold', y=0.995)

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.96,
        bottom=0.06,
        wspace=0.25,
        hspace=0.28
    )

    for idx, (cost_col, cost_label) in enumerate(cost_metrics):
        ax = axes[idx]

        # Create pivot for cost
        cost_pivot = timing_df.pivot(
            values=cost_col,
            index='learner',
            columns='featurizer',
            aggregate_function='first'
        )

        # Create pivot for performance
        perf_pivot = timing_df.pivot(
            values=performance_metric,
            index='learner',
            columns='featurizer',
            aggregate_function='first'
        )

        # Reorder columns
        desired_order = ['none', 'morgan', 'maccs', 'ecfp6', 'descriptors', 'morgan_feat']
        existing_cols = [col for col in desired_order if col in cost_pivot.columns]
        cost_pivot = cost_pivot.select(existing_cols)
        perf_pivot = perf_pivot.select(existing_cols)

        # Convert to pandas for heatmap
        cost_pd = cost_pivot.to_pandas()

        # Create colormap for cost (lower is better - green to red)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'green_red',
            ['#2d7f2d', '#77b377', '#ffeb99', '#ff9966', '#d73027']
        )

        # Normalize colors
        vmin = cost_pd.min().min()
        vmax = cost_pd.max().max()

        # Create heatmap
        sns.heatmap(
            cost_pd,
            annot=False,
            cmap=cmap,
            mask=cost_pd.isna(),
            cbar_kws={'shrink': 0.8, 'label': cost_label},
            linewidths=0.5,
            linecolor='white',
            square=False,
            vmin=vmin,
            vmax=vmax,
            ax=ax
        )

        # Add custom annotations with cost and performance
        for i in range(cost_pivot.height):
            for j in range(len(cost_pivot.columns)):
                cost_val = cost_pivot.row(i)[j]
                perf_val = perf_pivot.row(i)[j]
                if cost_val is not None and perf_val is not None:
                    time_str = _format_time_label(cost_val)
                    text = f'{time_str}\n({perf_val:.1f}%)'
                    ax.text(j + 0.5, i + 0.5, text,
                           ha='center', va='center',
                           fontsize=8, fontweight='bold',
                           color='white')

        ax.set_title(cost_label, fontsize=16, fontweight='bold', pad=12)
        ax.set_xlabel('Featurizer', fontsize=13, labelpad=8)
        ax.set_ylabel('Learner', fontsize=13, labelpad=8)
        ax.tick_params(axis='both', labelsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    output_filename = f'cost_performance_heatmap_{performance_metric}.png'
    output_path = plots_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Created cost/performance heatmap: {output_path.name}")

    return output_path


def create_pareto_frontiers(
    timing_df: pl.DataFrame,
    output_dir: Path,
    performance_metric: str = 'top_1_pct_discovery'
) -> Path:
    """Create 4-panel Pareto frontier plots showing cost vs performance trade-offs.

    Args:
        timing_df: DataFrame with timing and performance data
        output_dir: Directory to save output
        performance_metric: Performance column name (top_1_pct_discovery or top_0_1_pct_discovery)

    Returns:
        Path to saved Pareto plot image
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    cost_metrics = [
        ('cumulative_training_time', 'Cumulative Training Time (s)'),
        ('cumulative_prediction_time', 'Cumulative Prediction Time (s)'),
        ('elapsed_time', 'Total Elapsed Time (s)'),
        ('cumulative_training_prediction_time', 'Cumulative Training+Prediction Time (s)')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=300)
    axes = axes.flatten()

    metric_label = 'Top 1%' if 'top_1' in performance_metric else 'Top 0.1%'
    fig.suptitle(f'Pareto Frontier Analysis: {metric_label} Discovery Rate',
                 fontsize=24, fontweight='bold', y=0.995)

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.96,
        bottom=0.06,
        wspace=0.25,
        hspace=0.28
    )

    # Color mapping for learners
    unique_learners = sorted(timing_df['learner'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_learners)))
    learner_colors = dict(zip(unique_learners, colors))

    # Marker mapping for featurizers
    featurizer_markers = {
        'none': 'o',
        'morgan': 's',
        'maccs': '^',
        'ecfp6': 'D',
        'descriptors': 'v',
        'morgan_feat': 'p'
    }

    for idx, (cost_col, cost_label) in enumerate(cost_metrics):
        ax = axes[idx]

        # Get cost and performance arrays
        x_values = timing_df[cost_col].to_numpy()
        y_values = timing_df[performance_metric].to_numpy()
        learners = timing_df['learner'].to_numpy()
        featurizers = timing_df['featurizer'].to_numpy()

        # Identify Pareto frontier
        pareto_mask = _identify_pareto_frontier(x_values, y_values, maximize_y=True)

        # Plot all points
        for i, (x, y, learner, featurizer) in enumerate(zip(x_values, y_values, learners, featurizers)):
            marker = featurizer_markers.get(featurizer, 'o')
            color = learner_colors[learner]
            alpha = 1.0 if pareto_mask[i] else 0.3
            size = 120 if pareto_mask[i] else 60
            edgecolor = 'black' if pareto_mask[i] else 'none'
            linewidth = 2 if pareto_mask[i] else 0

            ax.scatter(x, y, c=[color], marker=marker, s=size,
                      alpha=alpha, edgecolors=edgecolor, linewidth=linewidth,
                      label=f'{learner} ({featurizer})' if i < len(unique_learners) else '')

        # Draw Pareto frontier line
        pareto_points = np.column_stack([x_values[pareto_mask], y_values[pareto_mask]])
        sorted_indices = np.argsort(pareto_points[:, 0])
        pareto_sorted = pareto_points[sorted_indices]

        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
               'k--', linewidth=2, alpha=0.5, label='Pareto Frontier')

        # Annotate Pareto-optimal points
        for i, (is_pareto, x, y, learner, featurizer) in enumerate(zip(pareto_mask, x_values, y_values, learners, featurizers)):
            if is_pareto:
                label = f'{learner[:3]}+{featurizer[:3]}'
                ax.annotate(label, (x, y), xytext=(5, 5),
                           textcoords='offset points', fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

        ax.set_xlabel(cost_label, fontsize=13, labelpad=8)
        ax.set_ylabel(f'{metric_label} Discovery Rate (%)', fontsize=13, labelpad=8)
        ax.set_title(f'{cost_label.split("(")[0].strip()}', fontsize=16, fontweight='bold', pad=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=100)

    # Create custom legend
    legend_elements = []
    for learner, color in learner_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=8,
                                         label=learner))

    for featurizer, marker in featurizer_markers.items():
        if featurizer in timing_df['featurizer'].unique():
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w',
                                             markerfacecolor='gray', markersize=8,
                                             label=f'Feat: {featurizer}'))

    fig.legend(handles=legend_elements, loc='center right',
              fontsize=8, framealpha=0.9, ncol=1, bbox_to_anchor=(1.0, 0.5))

    output_filename = f'pareto_frontier_{performance_metric}.png'
    output_path = plots_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Created Pareto frontier plot: {output_path.name}")

    return output_path


def create_all_cost_performance_plots(
    all_results: Dict[Tuple[str, str], Dict],
    output_dir: Path,
    config: Dict
) -> Dict[str, List[Path]]:
    """Generate all cost/performance visualizations.

    Args:
        all_results: Dictionary mapping (learner, featurizer) → result data
        output_dir: Base output directory
        config: Configuration dictionary with experiment parameters

    Returns:
        Dictionary with keys 'heatmaps' and 'pareto_plots', each containing list of Paths
    """
    print("=" * 80)
    print("GENERATING COST/PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()

    # Calculate cumulative timing data
    print("Calculating cumulative timing metrics...")
    timing_df = calculate_cumulative_timing(all_results)
    print(f"✓ Processed timing data for {len(timing_df)} learner-featurizer combinations")
    print()

    # Generate heatmaps for both metrics
    print("Creating cost/performance heatmaps...")
    heatmap_paths = []
    for perf_metric in ['top_1_pct_discovery', 'top_0_1_pct_discovery']:
        heatmap_path = create_cost_performance_heatmaps(timing_df, output_dir, perf_metric)
        heatmap_paths.append(heatmap_path)
    print()

    # Generate Pareto frontiers for both metrics
    print("Creating Pareto frontier plots...")
    pareto_paths = []
    for perf_metric in ['top_1_pct_discovery', 'top_0_1_pct_discovery']:
        pareto_path = create_pareto_frontiers(timing_df, output_dir, perf_metric)
        pareto_paths.append(pareto_path)
    print()

    print("=" * 80)
    print("COST/PERFORMANCE ANALYSIS COMPLETE")
    print("=" * 80)
    print()

    return {
        'heatmaps': heatmap_paths,
        'pareto_plots': pareto_paths
    }

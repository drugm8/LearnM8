from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from datetime import datetime
from learnm8.api import LEARNER_DISPLAY_NAMES
from validation.lib.heatmap_utils import (
    get_purple_colormap,
    auto_scale_range,
    add_heatmap_annotations
)


def create_top_k_heatmap(
    mean_matrix: pl.DataFrame,
    std_matrix: pl.DataFrame,
    k_label: str,
    metric_col: str,
    ax: plt.Axes,
    learner_names: List[str] = None
) -> plt.Axes:
    """Create heatmap with mean values and std dev annotations.

    Args:
        mean_matrix: DataFrame with mean values
        std_matrix: DataFrame with std dev values (same shape as mean_matrix)
        k_label: Label for top-k (e.g., 'Top 10')
        metric_col: Metric column name
        ax: Matplotlib axes
        learner_names: List of learner names for y-axis labels (optional)

    Returns:
        Modified axes
    """
    cmap = get_purple_colormap()

    mean_pd = mean_matrix.to_pandas()
    std_pd = std_matrix.to_pandas()
    mask = mean_pd.isna()

    mean_values = mean_pd.values
    vmin, vmax = auto_scale_range(mean_values, percentile=5, padding=0.05)

    hm = sns.heatmap(
        mean_pd,
        annot=False,
        cmap=cmap,
        mask=mask,
        cbar_kws={'shrink': 0.8, 'label': 'Discovery Rate (%)'},
        linewidths=0.3,
        linecolor='white',
        square=False,
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    add_heatmap_annotations(
        ax=ax,
        data=mean_values,
        std_data=std_pd.values,
        colormap=cmap,
        vmin=vmin,
        vmax=vmax,
        format_string='{:.1f}',
        show_std=True,
        fontsize=8
    )

    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Discovery Rate (%)', size=11)

    ax.set_title(f'{k_label} Discovery Rate (mean ± std)',
                 fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.tick_params(axis='both', labelsize=11)
    ax.set_xticklabels(mean_matrix.columns, rotation=45, ha='right')

    if learner_names is not None:
        display_names = [LEARNER_DISPLAY_NAMES.get(name, name) for name in learner_names]
    else:
        display_names = [LEARNER_DISPLAY_NAMES.get(name, name) for name in mean_pd.index]
    ax.set_yticklabels(display_names, rotation=0)

    return ax


def create_all_heatmaps(
    results_df: pl.DataFrame,
    output_dir: Path,
    dataset_name: str = None
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

    title = 'Learner-Acquisition Performance Matrix: Discovery Rates'
    if dataset_name:
        title += f'\nDataset: {dataset_name}'

    fig.suptitle(title,
                 fontsize=24, fontweight='bold', y=0.995)

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.94,
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
            columns='acquisition',
            aggregate_function='first'
        )

        std_pivot = results_df.pivot(
            values=std_col,
            index='learner',
            columns='acquisition',
            aggregate_function='first'
        )

        learner_names = mean_pivot.get_column('learner').to_list()

        desired_order = ['greedy', 'random', 'simulated_annealing', 'ucb', 'ei', 'pi', 'thompson', 'entropy']
        existing_cols = [col for col in desired_order if col in mean_pivot.columns]
        mean_pivot = mean_pivot.select(existing_cols)
        std_pivot = std_pivot.select(existing_cols)

        create_top_k_heatmap(mean_pivot, std_pivot, k_label, metric_col, axes[idx], learner_names)

    output_path = plots_dir / 'heatmap_combined.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Created combined heatmap: {output_path.name}")

    return output_path


def create_greedy_cycle_plot(
    greedy_results: Dict[str, Dict],
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

    colors = plt.cm.tab20(np.linspace(0, 1, len(greedy_results)))

    for ax_idx, (metric_col, title) in enumerate(metrics):
        ax = axes[ax_idx]

        for learner_idx, (learner_name, result_data) in enumerate(sorted(greedy_results.items())):
            aggregated = result_data['aggregated']
            cycle_metrics_mean = aggregated['cycle_metrics_mean']
            cycle_metrics_std = aggregated['cycle_metrics_std']

            cycles = [m['cycle'] for m in cycle_metrics_mean]
            mean_values = [m.get(metric_col, 0) for m in cycle_metrics_mean]
            std_values = [m.get(metric_col, 0) for m in cycle_metrics_std]

            ax.errorbar(cycles, mean_values, yerr=std_values,
                       marker='o', linewidth=2, label=learner_name,
                       color=colors[learner_idx], markersize=4,
                       alpha=0.8, capsize=3, capthick=1)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Cycle', fontsize=10)
        ax.set_ylabel('Discovery Rate (%)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=100)

        if ax_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     fontsize=8, framealpha=0.9)

    plt.suptitle('Learner Performance Comparison: Greedy Acquisition',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"✓ Created cycle comparison plot: {output_path.name}")

    return output_path


def create_summary_report(
    results_df: pl.DataFrame,
    greedy_results: Dict[str, Dict],
    heatmap_path: Path,
    cycle_plot_path: Path,
    output_dir: Path,
    config: Dict
) -> Path:
    output_dir = Path(output_dir)
    report_path = output_dir / 'validation_report.md'

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = [
        "# Learner-Acquisition Matrix Validation Report",
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
        f"| Featurizer | {config['featurizer_type']} |",
        f"| Random Seeds | {config['random_seeds']} |",
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
        best_idx = results_df[mean_col].arg_max()
        best_row = results_df.row(best_idx, named=True)
        metric_name = metric.replace('_', ' ').title()
        lines.append(f"**{metric_name}:** {best_row['learner']} + {best_row['acquisition']} "
                    f"({best_row[mean_col]:.1f}±{best_row[std_col]:.1f}%)")

    lines.extend([
        "",
        "### Performance Summary Table (Mean ± Std)",
        "",
        results_df.group_by('learner').agg([
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
        "### Cycle-by-Cycle Performance: Greedy Acquisition",
        "",
        f"![Greedy Cycle Comparison]({cycle_plot_path.relative_to(output_dir)})",
        "",
        "## Key Observations",
        ""
    ])

    uncertainty_learners = results_df.filter(pl.col('acquisition').is_in(['ucb', 'ei', 'pi', 'thompson', 'entropy']))['learner'].unique().to_list()
    basic_learners = results_df.filter(~pl.col('learner').is_in(uncertainty_learners))['learner'].unique().to_list()

    lines.extend([
        f"- **Uncertainty-capable learners:** {', '.join(sorted(uncertainty_learners))}",
        f"- **Basic learners (greedy/random only):** {', '.join(sorted(basic_learners))}",
        f"- **Total experiments completed:** {len(results_df)}",
        "",
        "## Data Files",
        "",
        "Individual experiment results saved to:",
        "```",
        "validation/reports/learner_acquisition_matrix/data/{learner}_{acquisition}/",
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
    for (learner, acquisition), result_data in all_results.items():
        aggregated = result_data['aggregated']
        final_metrics_mean = aggregated['cycle_metrics_mean'][-1]
        final_metrics_std = aggregated['cycle_metrics_std'][-1]

        results_list.append({
            'learner': learner,
            'acquisition': acquisition,
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

    heatmap_path = create_all_heatmaps(results_df, output_dir, config.get('dataset_name'))

    greedy_results = {
        learner: result_data
        for (learner, acquisition), result_data in all_results.items()
        if acquisition == 'greedy'
    }

    plots_dir = output_dir / 'plots'
    cycle_plot_path = create_greedy_cycle_plot(
        greedy_results,
        plots_dir / 'greedy_cycle_comparison.png'
    )

    report_path = create_summary_report(
        results_df,
        greedy_results,
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

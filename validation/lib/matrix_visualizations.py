from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from datetime import datetime
from learnm8.api import LEARNER_DISPLAY_NAMES
from learnm8.visualization import style
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
    style.apply()
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
        cbar_kws={
            'shrink': 0.8,
            'label': 'Initial-pool top-set recovered (%)',
        },
        linewidths=0.3,
        linecolor=style.BACKGROUND,
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
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Initial-pool top-set recovered (%)', size=9)

    ax.set_title(f'{k_label} recovered from initial pool (mean ± SD)',
                 fontsize=11, fontweight='bold', pad=9, loc='left')
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.tick_params(axis='both', labelsize=8)
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
    style.apply()
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    heatmap_configs = [
        ('top_0_1_pct_discovery', 'Top 0.1%'),
        ('top_1_pct_discovery', 'Top 1%'),
        ('top_10_discovery', 'Top 10'),
        ('top_100_discovery', 'Top 100')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=style.OUTPUT_DPI)
    axes = axes.flatten()

    title = 'Learner–acquisition performance: initial-pool recovery'
    if dataset_name:
        title += f'\nDataset: {dataset_name}'

    fig.suptitle(title, fontsize=15, fontweight='bold')

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
    fig.savefig(output_path, dpi=style.OUTPUT_DPI, bbox_inches='tight', pad_inches=0.08)
    plt.close()

    print(f"✓ Created combined heatmap: {output_path.name}")

    return output_path


def create_greedy_cycle_plot(
    greedy_results: Dict[str, Dict],
    output_path: Path,
    dpi: int = style.OUTPUT_DPI
) -> Path:
    style.apply()
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

    colors = style.categorical_colors(len(greedy_results))

    for ax_idx, (metric_col, title) in enumerate(metrics):
        ax = axes[ax_idx]

        for learner_idx, (learner_name, result_data) in enumerate(sorted(greedy_results.items())):
            aggregated = result_data['aggregated']
            cycle_metrics_mean = aggregated['cycle_metrics_mean']
            cycle_metrics_std = aggregated['cycle_metrics_std']

            cycles = [m['cycle'] for m in cycle_metrics_mean]
            mean_values = [m.get(metric_col, 0) for m in cycle_metrics_mean]
            std_values = [m.get(metric_col, 0) for m in cycle_metrics_std]

            ax.errorbar(
                       cycles, mean_values, yerr=std_values,
                       marker=style.CURVE_MARKERS[learner_idx % len(style.CURVE_MARKERS)],
                       linestyle=style.CURVE_LINESTYLES[
                           learner_idx % len(style.CURVE_LINESTYLES)
                       ],
                       linewidth=1.8, label=learner_name,
                       color=colors[learner_idx], markersize=4,
                       alpha=0.8, capsize=3, capthick=1)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Cycle', fontsize=10)
        ax.set_ylabel('Initial-pool top-set recovered (%)', fontsize=10)
        style.style_axes(ax)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=100)

        if ax_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     fontsize=8, framealpha=0.9)

    fig.suptitle('Learner performance · greedy acquisition', fontsize=15, fontweight='bold')
    fig.savefig(output_path, dpi=style.OUTPUT_DPI, bbox_inches='tight', pad_inches=0.08)
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
        f"| Featurizer | {config['featurizer']} |",
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


# ── Functions absorbed from featurizer_visualizations.py ──────────────────────


def featurizer_create_all_heatmaps(
    results_df: pl.DataFrame,
    output_dir: Path,
    dataset_name: str = None
) -> Path:
    style.apply()
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    heatmap_configs = [
        ('top_0_1_pct_discovery', 'Top 0.1%'),
        ('top_1_pct_discovery', 'Top 1%'),
        ('top_10_discovery', 'Top 10'),
        ('top_100_discovery', 'Top 100')
    ]

    n_featurizers = results_df['featurizer'].n_unique()
    fig_width = max(20, 1.2 * n_featurizers)
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 16), dpi=style.OUTPUT_DPI)
    axes = axes.flatten()

    title = 'Learner-Featurizer Performance Matrix: Discovery Rates'
    if dataset_name:
        title += f'\nDataset: {dataset_name}'

    fig.suptitle(title, fontsize=15, fontweight='bold')

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

        learner_names = mean_pivot.get_column('learner').to_list()

        featurizer_cols = [c for c in mean_pivot.columns if c != 'learner']
        ordered_cols = (['none'] if 'none' in featurizer_cols else []) + sorted(c for c in featurizer_cols if c != 'none')
        mean_pivot = mean_pivot.select(ordered_cols)
        std_pivot = std_pivot.select(ordered_cols)

        create_top_k_heatmap(mean_pivot, std_pivot, k_label, metric_col, axes[idx], learner_names)

    output_path = plots_dir / 'heatmap_combined.png'
    fig.savefig(output_path, dpi=style.OUTPUT_DPI, bbox_inches='tight', pad_inches=0.08)
    plt.close()

    print(f"✓ Created combined heatmap: {output_path.name}")

    return output_path


def create_learner_cycle_plot(
    learner_results: Dict[str, Dict],
    output_path: Path,
    dpi: int = style.OUTPUT_DPI
) -> Path:
    style.apply()
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

    colors = style.categorical_colors(6)
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
                           color=featurizer_colors.get(featurizer_name, style.MUTED),
                           markersize=3, alpha=0.7, linestyle='-',
                           capsize=2, capthick=1)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Cycle', fontsize=10)
        ax.set_ylabel('Discovery Rate (%)', fontsize=10)
        style.style_axes(ax)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=100)

        if ax_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     fontsize=6, framealpha=0.9, ncol=1)

    fig.suptitle(
        'Learner · featurizer performance · greedy acquisition',
        fontsize=15,
        fontweight='bold',
    )
    fig.savefig(output_path, dpi=style.OUTPUT_DPI, bbox_inches='tight', pad_inches=0.08)
    plt.close()

    print(f"✓ Created cycle comparison plot: {output_path.name}")

    return output_path


def featurizer_create_summary_report(
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

    learner_counts = results_df['learner'].n_unique()
    featurizer_counts = results_df['featurizer'].n_unique()

    lines.extend([
        f"- **Learners tested:** {learner_counts} ({', '.join(sorted(results_df['learner'].unique().to_list()))})",
        f"- **Featurizers tested:** {featurizer_counts} ({', '.join(sorted(results_df['featurizer'].unique().to_list()))})",
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


def featurizer_generate_comprehensive_visualizations(
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

    heatmap_path = featurizer_create_all_heatmaps(results_df, output_dir, config.get('dataset_name'))

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

    report_path = featurizer_create_summary_report(
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
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def calculate_cumulative_timing(all_results: Dict[Tuple[str, str], Dict]) -> pl.DataFrame:
    timing_data = []

    for (learner, featurizer), result_data in all_results.items():
        aggregated = result_data['aggregated']
        cycle_metrics_mean = aggregated['cycle_metrics_mean']

        cumulative_training = 0.0
        cumulative_prediction = 0.0
        cumulative_total = 0.0

        for cycle_metric in cycle_metrics_mean:
            cumulative_training += cycle_metric.get('training_time') or 0.0
            cumulative_prediction += cycle_metric.get('prediction_time') or 0.0
            cumulative_total += cycle_metric.get('total_time') or 0.0

        cumulative_training_prediction = cumulative_training + cumulative_prediction

        final_metrics = cycle_metrics_mean[-1]

        timing_data.append({
            'learner': learner,
            'featurizer': featurizer if featurizer else 'none',
            'cumulative_training_time': cumulative_training,
            'cumulative_prediction_time': cumulative_prediction,
            'cumulative_total_time': cumulative_total,
            'cumulative_training_prediction_time': cumulative_training_prediction,
            'elapsed_time': aggregated.get('elapsed_time_mean', cumulative_total),
            'top_0_1_pct_discovery': final_metrics.get('top_0_1_pct_discovery', 0.0),
            'top_1_pct_discovery': final_metrics.get('top_1_pct_discovery', 0.0),
            'top_10_discovery': final_metrics.get('top_10_discovery', 0.0),
            'top_100_discovery': final_metrics.get('top_100_discovery', 0.0),
            'top_0_1_pct_recovery': final_metrics.get('top_0_1_pct_discovery', 0.0),
            'top_1_pct_recovery': final_metrics.get('top_1_pct_discovery', 0.0),
            'top_10_recovery': final_metrics.get('top_10_discovery', 0.0),
            'top_100_recovery': final_metrics.get('top_100_discovery', 0.0),
        })

    return pl.DataFrame(timing_data)


def _get_metric_label(performance_metric: str) -> str:
    metric_labels = {
        'top_0_1_pct_recovery': 'Top 0.1%',
        'top_1_pct_recovery': 'Top 1%',
        'top_10_recovery': 'Top 10',
        'top_100_recovery': 'Top 100',
    }
    return metric_labels.get(performance_metric, performance_metric)


def create_time_heatmap(
    timing_df: pl.DataFrame,
    output_dir: Path,
    performance_metric: str = 'top_10_recovery',
    dataset_name: str = None
) -> Path:
    style.apply()
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_label = _get_metric_label(performance_metric)

    index_col = 'config_name' if 'config_name' in timing_df.columns else 'learner'

    time_pivot = timing_df.pivot(
        values='cumulative_training_prediction_time',
        index=index_col,
        columns='featurizer',
        aggregate_function='first'
    )

    row_names = time_pivot.get_column(index_col).to_list()

    featurizer_cols = [c for c in time_pivot.columns if c != index_col]
    ordered_cols = (['none'] if 'none' in featurizer_cols else []) + sorted(c for c in featurizer_cols if c != 'none')
    time_pivot = time_pivot.select(ordered_cols)

    time_pd = time_pivot.to_pandas()

    n_featurizers = len(ordered_cols)
    fig_width = max(12, 0.8 * n_featurizers)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 10), dpi=style.OUTPUT_DPI)

    title = f'Time Matrix: Cumulative Training + Prediction Time (seconds)\n{metric_label} Discovery Rate'
    if dataset_name:
        title += f'\nDataset: {dataset_name}'

    fig.suptitle(title, fontsize=14, fontweight='bold')

    cmap = get_purple_colormap(reverse=True)

    time_values = time_pd.values
    vmin, vmax = auto_scale_range(time_values, percentile=5, padding=0.05)

    sns.heatmap(
        time_pd,
        annot=False,
        cmap=cmap,
        mask=time_pd.isna(),
        cbar_kws={'shrink': 0.8, 'label': 'Time (seconds)'},
        linewidths=0.3,
        linecolor=style.BACKGROUND,
        square=False,
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    add_heatmap_annotations(
        ax=ax,
        data=time_values,
        std_data=None,
        colormap=cmap,
        vmin=vmin,
        vmax=vmax,
        format_string='{:.1f}s',
        show_std=False,
        fontsize=9
    )

    ax.set_xlabel('Featurizer', fontsize=13, labelpad=8)
    ax.set_ylabel('Configuration', fontsize=13, labelpad=8)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xticklabels(time_pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(row_names, rotation=0)

    output_filename = f'time_heatmap_{performance_metric}.png'
    output_path = plots_dir / output_filename
    fig.savefig(output_path, dpi=style.OUTPUT_DPI, bbox_inches='tight', pad_inches=0.08)
    plt.close()

    print(f"✓ Created time heatmap: {output_path.name}")

    return output_path


def create_performance_heatmap(
    timing_df: pl.DataFrame,
    output_dir: Path,
    performance_metric: str = 'top_10_recovery',
    dataset_name: str = None
) -> Path:
    style.apply()
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_label = _get_metric_label(performance_metric)

    index_col = 'config_name' if 'config_name' in timing_df.columns else 'learner'

    perf_pivot = timing_df.pivot(
        values=performance_metric,
        index=index_col,
        columns='featurizer',
        aggregate_function='first'
    )

    row_names = perf_pivot.get_column(index_col).to_list()

    featurizer_cols = [c for c in perf_pivot.columns if c != index_col]
    ordered_cols = (['none'] if 'none' in featurizer_cols else []) + sorted(c for c in featurizer_cols if c != 'none')
    perf_pivot = perf_pivot.select(ordered_cols)

    perf_pd = perf_pivot.to_pandas()

    n_featurizers = len(ordered_cols)
    fig_width = max(12, 0.8 * n_featurizers)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 10), dpi=style.OUTPUT_DPI)

    title = f'Performance Matrix: {metric_label} Discovery Rate (%)'
    if dataset_name:
        title += f'\nDataset: {dataset_name}'

    fig.suptitle(title, fontsize=14, fontweight='bold')

    cmap = get_purple_colormap()

    perf_values = perf_pd.values
    vmin, vmax = auto_scale_range(perf_values, percentile=5, padding=0.05)

    sns.heatmap(
        perf_pd,
        annot=False,
        cmap=cmap,
        mask=perf_pd.isna(),
        cbar_kws={'shrink': 0.8, 'label': 'Discovery Rate (%)'},
        linewidths=0.3,
        linecolor=style.BACKGROUND,
        square=False,
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    add_heatmap_annotations(
        ax=ax,
        data=perf_values,
        std_data=None,
        colormap=cmap,
        vmin=vmin,
        vmax=vmax,
        format_string='{:.1f}%',
        show_std=False,
        fontsize=9
    )

    ax.set_xlabel('Featurizer', fontsize=13, labelpad=8)
    ax.set_ylabel('Configuration', fontsize=13, labelpad=8)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xticklabels(perf_pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(row_names, rotation=0)

    output_filename = f'performance_heatmap_{performance_metric}.png'
    output_path = plots_dir / output_filename
    fig.savefig(output_path, dpi=style.OUTPUT_DPI, bbox_inches='tight', pad_inches=0.08)
    plt.close()

    print(f"✓ Created performance heatmap: {output_path.name}")

    return output_path


def create_efficiency_heatmap(
    timing_df: pl.DataFrame,
    output_dir: Path,
    performance_metric: str = 'top_10_recovery',
    dataset_name: str = None
) -> Path:
    style.apply()
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_label = _get_metric_label(performance_metric)

    index_col = 'config_name' if 'config_name' in timing_df.columns else 'learner'

    timing_with_efficiency = timing_df.with_columns([
        (pl.col(performance_metric) / pl.col('cumulative_training_prediction_time')).alias('efficiency')
    ])

    efficiency_pivot = timing_with_efficiency.pivot(
        values='efficiency',
        index=index_col,
        columns='featurizer',
        aggregate_function='first'
    )

    row_names = efficiency_pivot.get_column(index_col).to_list()

    featurizer_cols = [c for c in efficiency_pivot.columns if c != index_col]
    ordered_cols = (['none'] if 'none' in featurizer_cols else []) + sorted(c for c in featurizer_cols if c != 'none')
    efficiency_pivot = efficiency_pivot.select(ordered_cols)

    efficiency_pd = efficiency_pivot.to_pandas()

    n_featurizers = len(ordered_cols)
    fig_width = max(12, 0.8 * n_featurizers)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 10), dpi=style.OUTPUT_DPI)

    title = f'Efficiency Matrix: {metric_label} Discovery Rate per Second (%/s)'
    if dataset_name:
        title += f'\nDataset: {dataset_name}'

    fig.suptitle(title, fontsize=14, fontweight='bold')

    cmap = get_purple_colormap()

    efficiency_values = efficiency_pd.values
    vmin, vmax = auto_scale_range(efficiency_values, percentile=5, padding=0.05)

    sns.heatmap(
        efficiency_pd,
        annot=False,
        cmap=cmap,
        mask=efficiency_pd.isna(),
        cbar_kws={'shrink': 0.8, 'label': 'Efficiency (%/s)'},
        linewidths=0.3,
        linecolor=style.BACKGROUND,
        square=False,
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    add_heatmap_annotations(
        ax=ax,
        data=efficiency_values,
        std_data=None,
        colormap=cmap,
        vmin=vmin,
        vmax=vmax,
        format_string='{:.2f}',
        show_std=False,
        fontsize=9
    )

    ax.set_xlabel('Featurizer', fontsize=13, labelpad=8)
    ax.set_ylabel('Configuration', fontsize=13, labelpad=8)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xticklabels(efficiency_pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(row_names, rotation=0)

    output_filename = f'efficiency_heatmap_{performance_metric}.png'
    output_path = plots_dir / output_filename
    fig.savefig(output_path, dpi=style.OUTPUT_DPI, bbox_inches='tight', pad_inches=0.08)
    plt.close()

    print(f"✓ Created efficiency heatmap: {output_path.name}")

    return output_path


def create_all_cost_performance_plots(
    all_results: Dict[Tuple[str, str], Dict],
    output_dir: Path,
    config: Dict
) -> Dict[str, List[Path]]:
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    timing_df = calculate_cumulative_timing(all_results)

    if len(timing_df) == 0:
        print("Warning: No successful experiments to visualize")
        return {'heatmaps': []}

    all_heatmaps = []
    dataset_name = config.get('dataset_name', 'validation')

    for metric in ['top_0_1_pct_recovery', 'top_1_pct_recovery', 'top_10_recovery', 'top_100_recovery']:
        time_path = create_time_heatmap(
            timing_df,
            output_dir,
            performance_metric=metric,
            dataset_name=dataset_name
        )
        all_heatmaps.append(time_path)

        perf_path = create_performance_heatmap(
            timing_df,
            output_dir,
            performance_metric=metric,
            dataset_name=dataset_name
        )
        all_heatmaps.append(perf_path)

        efficiency_path = create_efficiency_heatmap(
            timing_df,
            output_dir,
            performance_metric=metric,
            dataset_name=dataset_name
        )
        all_heatmaps.append(efficiency_path)

    print(f"✓ Generated {len(all_heatmaps)} cost/performance heatmaps")

    return {'heatmaps': all_heatmaps}

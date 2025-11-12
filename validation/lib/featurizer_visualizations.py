from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from datetime import datetime


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

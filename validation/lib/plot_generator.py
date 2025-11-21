import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from learnm8.visualization import create_dashboard_animation_from_csv


def create_comprehensive_validation_plot(
    result: Dict[str, Any],
    strategy_config: Dict[str, Any],
    param_value: float,
    output_path: Path,
    dpi: int = 300,
    dataset_name: Optional[str] = None,
    learner_name: Optional[str] = None
) -> Path:
    param_name = strategy_config['param_name']
    strategy_name = strategy_config['name']

    df = result['compounds_df']
    cycle_metrics_df = pl.DataFrame(result['cycle_metrics'])

    # Auto-detect learner capabilities from data
    has_uncertainty = False
    sample_cycle = 1
    pred_col = f'prediction_cycle_{sample_cycle}'
    unc_col = f'uncertainty_cycle_{sample_cycle}'
    if pred_col in df.columns and unc_col in df.columns:
        has_uncertainty = len(df[unc_col].drop_nulls()) > 0

    # Auto-detect pruning from metrics
    has_pruning = False
    if 'cumulative_pruned' in cycle_metrics_df.columns:
        has_pruning = cycle_metrics_df['cumulative_pruned'].max() > 0

    # Determine layout based on what we're showing
    # Always show top 4 plots (uncertainty scatter or prediction violin)
    n_rows = 0
    cycle_viz_rows = [0, 1]
    n_rows = 2
    pruning_row = None
    metrics_row = None

    if has_pruning:
        pruning_row = n_rows
        n_rows += 1

    metrics_row = n_rows
    n_rows += 1

    # Create figure with appropriate size
    fig_height = 3 + (n_rows * 2.8)
    fig = plt.figure(figsize=(12, fig_height))

    # Build height ratios: cycle plots = 1/3, metrics = 2/3
    # Top 4 plots across 2 rows = 0.5 units each (total 1.0 for 1/3)
    # Metrics plot = 2.0 units (for 2/3)
    height_ratios = [0.5, 0.5]  # Top 4 cycle visualization plots (1/3 total)
    if has_pruning:
        height_ratios.append(0.6)
    height_ratios.append(2.0)  # Discovery and score metrics (2/3 total)

    gs = gridspec.GridSpec(n_rows, 2, height_ratios=height_ratios, hspace=0.4, wspace=0.3, top=0.96)

    # Plot top 4 cycle visualizations (uncertainty scatter or prediction violin)
    cycles_to_show = [1, 3, 6, 9]
    final_cycle = int(cycle_metrics_df['cycle'].max())
    cycles_to_show = [c for c in cycles_to_show if c <= final_cycle]

    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    if has_uncertainty:
        # Uncertainty scatter plots
        for idx, cycle in enumerate(cycles_to_show):
            row, col = subplot_positions[idx]
            ax = fig.add_subplot(gs[row, col])

            pred_col = f'prediction_cycle_{cycle}'
            unc_col = f'uncertainty_cycle_{cycle}'

            if pred_col not in df.columns or unc_col not in df.columns:
                ax.text(0.5, 0.5, f'Missing data\nfor cycle {cycle}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Cycle {cycle}')
                continue

            # Get all compounds with predictions at this cycle (labeled or unlabeled)
            compounds_with_pred = df.filter(
                pl.col(pred_col).is_not_null() & pl.col(unc_col).is_not_null()
            )

            # Plot all compounds as density hexbin (no subsampling)
            hb = ax.hexbin(
                compounds_with_pred[pred_col].to_numpy(),
                compounds_with_pred[unc_col].to_numpy(),
                gridsize=50,           # Balance between detail and performance
                cmap='Greys',          # Neutral, shows density clearly
                mincnt=1,              # Show even single points
                norm=LogNorm(),        # Logarithmic scale makes low-density areas visible
                alpha=0.7,             # Slightly more opaque for better visibility
                edgecolors='none',     # Clean appearance
                linewidths=0,
                zorder=1               # Behind selected points
            )

            selected_this_cycle = df.filter(
                (pl.col('status') == 'labeled') & (pl.col('selected_cycle').cast(pl.Int64) == cycle)
            )

            if len(selected_this_cycle) > 0:
                ax.scatter(
                    selected_this_cycle[pred_col].to_numpy(),
                    selected_this_cycle[unc_col].to_numpy(),
                    c='#7e62df',
                    alpha=0.75,
                    s=35,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f'Selected (n={len(selected_this_cycle)})',
                    zorder=10             # Ensure visibility on top of hexbin
                )

            ax.set_xlabel('Prediction', fontsize=9)
            if col == 0:
                ax.set_ylabel('Uncertainty', fontsize=9)
            ax.set_title(f'Cycle {cycle}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, linewidth=0.5)
            if len(selected_this_cycle) > 0:
                ax.legend(fontsize=7, loc='best', framealpha=0.9)
    else:
        # Horizontal violin plots showing prediction distribution and selection
        for idx, cycle in enumerate(cycles_to_show):
            row, col = subplot_positions[idx]
            ax = fig.add_subplot(gs[row, col])

            pred_col = f'prediction_cycle_{cycle}'

            if pred_col not in df.columns:
                ax.text(0.5, 0.5, f'Missing data\nfor cycle {cycle}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Cycle {cycle}')
                continue

            # Get all compounds with predictions at this cycle
            compounds_with_pred = df.filter(pl.col(pred_col).is_not_null())

            if len(compounds_with_pred) == 0:
                ax.text(0.5, 0.5, f'No predictions\nfor cycle {cycle}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Cycle {cycle}')
                continue

            all_predictions = compounds_with_pred[pred_col].to_numpy()

            # Get selected compounds in this specific cycle
            selected_this_cycle = df.filter(
                (pl.col('status') == 'labeled') &
                (pl.col('selected_cycle').cast(pl.Int64) == cycle)
            )

            # Create horizontal violin plot
            parts = ax.violinplot([all_predictions], positions=[0], vert=False,
                                 widths=0.7, showmeans=False, showmedians=False,
                                 showextrema=False)

            # Style the violin
            for pc in parts['bodies']:
                pc.set_facecolor('#d0c9e8')
                pc.set_edgecolor('#7e62df')
                pc.set_alpha(0.6)
                pc.set_linewidth(1.5)

            # Show selected predictions as a rug plot or strip
            if len(selected_this_cycle) > 0:
                selected_preds = selected_this_cycle[pred_col].to_numpy()

                # Add jitter to y-position for better visibility
                n_selected = len(selected_preds)
                jitter = np.random.RandomState(42 + cycle).uniform(-0.15, 0.15, n_selected)

                ax.scatter(selected_preds, jitter,
                          c='#7e62df', s=30, alpha=0.8,
                          edgecolors='black', linewidths=0.5,
                          zorder=10, label=f'Selected (n={n_selected})')

                # Add vertical lines showing selection range
                if len(selected_preds) > 1:
                    sel_min, sel_max = selected_preds.min(), selected_preds.max()
                    ax.axvline(sel_min, color='#7e62df', linestyle='--',
                              linewidth=1.5, alpha=0.5, zorder=5)
                    ax.axvline(sel_max, color='#7e62df', linestyle='--',
                              linewidth=1.5, alpha=0.5, zorder=5)

            ax.set_xlabel('Prediction', fontsize=9)
            ax.set_yticks([])
            ax.set_ylim(-0.5, 0.5)
            ax.set_title(f'Cycle {cycle}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, linewidth=0.5, axis='x')

            if len(selected_this_cycle) > 0:
                ax.legend(fontsize=7, loc='upper right', framealpha=0.9)

    # Plot pruning timeline if pruning was used
    if has_pruning:
        ax_prune = fig.add_subplot(gs[pruning_row, :])
        ax_prune_twin = ax_prune.twinx()

        cycles = cycle_metrics_df['cycle'].to_numpy()
        pool_sizes = cycle_metrics_df['pool_size'].to_numpy()
        cumulative_labeled = cycle_metrics_df['cumulative_labeled'].to_numpy()

        ax_prune.plot(cycles, pool_sizes, color='#7e62df', linewidth=2.5,
                     marker='o', markersize=5, label='Unlabeled Pool Size')
        ax_prune.fill_between(cycles, 0, pool_sizes, color='#7e62df', alpha=0.15)

        ax_prune_twin.plot(cycles, cumulative_labeled, color='#e67e22', linewidth=2.5,
                          marker='s', markersize=5, label='Cumulative Labeled', linestyle='--')

        ax_prune.set_xlabel('Cycle', fontsize=11)
        ax_prune.set_ylabel('Unlabeled Pool Size', fontsize=11, color='#7e62df')
        ax_prune.tick_params(axis='y', labelcolor='#7e62df')
        ax_prune_twin.set_ylabel('Cumulative Labeled', fontsize=11, color='#e67e22')
        ax_prune_twin.tick_params(axis='y', labelcolor='#e67e22')

        ax_prune.grid(True, alpha=0.3, linewidth=0.75)
        ax_prune.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax_prune_twin.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Create two separate subplots for discovery and score metrics with shared x-axis
    # Use a nested GridSpec for the metrics row to create two plot areas
    metrics_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[metrics_row, :],
                                                   hspace=0.05, height_ratios=[1, 1])

    ax_discovery = fig.add_subplot(metrics_gs[0])
    ax_score = fig.add_subplot(metrics_gs[1], sharex=ax_discovery)

    discovery_metrics = [
        ('top_10_discovery', 'Top-10', '#452997', '-'),
        ('top_100_discovery', 'Top-100', '#5b3fb5', '-'),
        ('top_1000_discovery', 'Top-1000', '#7e62df', '-'),
        ('top_0_1_pct_discovery', 'Top-0.1%', '#a48dd7', '--'),
        ('top_1_pct_discovery', 'Top-1%', '#c8b5e7', '--'),
        ('top_10_pct_discovery', 'Top-10%', '#e3d8f0', ':')
    ]

    score_metrics = [
        ('cumulative_avg_score_ratio', 'Cumulative Avg', '#7e62df', '-'),
        ('batch_avg_score_ratio', 'Batch Avg', '#b7a5df', '-')
    ]

    cycles = cycle_metrics_df['cycle'].to_numpy()
    batch_sizes = cycle_metrics_df['selected_count'].to_numpy()
    cumulative_compounds = np.cumsum(batch_sizes)
    cumulative_pct = (cumulative_compounds / len(df)) * 100

    xtick_labels = []
    for i, cycle in enumerate(cycles):
        n_compounds = int(cumulative_compounds[i])
        xtick_labels.append(f'C{cycle}: {cumulative_pct[i]:.1f}%\n({n_compounds})')

    # Plot discovery metrics
    for metric, label, color, style in discovery_metrics:
        if metric in cycle_metrics_df.columns:
            values = cycle_metrics_df[metric].to_numpy()
            ax_discovery.plot(cumulative_pct, values, color=color, linestyle=style,
                            linewidth=2, marker='o', markersize=4, label=label)

    ax_discovery.set_ylabel('Discovery Rate (%)', fontsize=11)
    ax_discovery.set_ylim(0, 100)
    ax_discovery.grid(True, alpha=0.3, linewidth=0.75)
    ax_discovery.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)
    ax_discovery.tick_params(labelbottom=False)

    # Plot score metrics
    score_values_all = []
    for metric, label, color, style in score_metrics:
        if metric in cycle_metrics_df.columns:
            values = cycle_metrics_df[metric].to_numpy()
            score_values_all.extend(values)
            ax_score.plot(cumulative_pct, values, color=color, linestyle=style,
                         linewidth=2, marker='o', markersize=4, label=label)

    ax_score.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Random baseline')

    # Dynamic y-axis for score ratio
    if score_values_all:
        score_min = min(score_values_all)
        score_max = max(score_values_all)
        y_min = max(0.8, score_min - 0.2)
        y_max = min(3.0, score_max + 0.5)
        ax_score.set_ylim(y_min, y_max)

    ax_score.set_xlabel('Cumulative % Compounds Evaluated', fontsize=11)
    ax_score.set_ylabel('Score Ratio', fontsize=11)
    ax_score.set_xticks(cumulative_pct)
    ax_score.set_xticklabels(xtick_labels, fontsize=8, rotation=0)
    ax_score.grid(True, alpha=0.3, linewidth=0.75)
    ax_score.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)

    # Add a shared title for both metric plots
    ax_discovery.text(0.5, 1.15, 'Discovery and Score Metrics',
                     ha='center', va='bottom', transform=ax_discovery.transAxes,
                     fontsize=12, fontweight='bold')

    # Generate title
    if dataset_name and learner_name:
        title = f'{dataset_name} {learner_name} Validation'
    elif param_name and param_value is not None:
        title = f'{strategy_name} Validation: {param_name}={param_value}'
    else:
        title = f'{strategy_name} Validation'

    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.998)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Comprehensive validation plot saved: {output_path.resolve()}")
    return output_path


def create_animations(
    results: Dict[str, Any],
    strategy_config: Dict[str, Any],
    output_dir: Path,
    fps: int = 2,
    dpi: int = 100
) -> List[Path]:
    param_name = strategy_config['param_name']
    param_values = strategy_config['param_values']
    strategy_name = strategy_config['name']

    print(f"\n  Creating animations for {strategy_name}...")

    animations_dir = output_dir / 'animations'
    animations_dir.mkdir(parents=True, exist_ok=True)

    animation_paths = []

    for param_value in param_values:
        result = results[param_value]
        data_dir = Path(result['output_dir'])

        if not data_dir.exists():
            print(f"    Warning: Data directory not found for {param_name}={param_value}")
            continue

        required_files = ['cycle_metrics.csv', 'compounds_final.csv']
        if not all((data_dir / f).exists() for f in required_files):
            print(f"    Warning: Missing required files for {param_name}={param_value}")
            continue

        anim_filename = f"{param_name}_{param_value}.gif"
        anim_path = animations_dir / anim_filename

        try:
            create_dashboard_animation_from_csv(
                output_dir=str(data_dir),
                output_file=str(anim_path),
                format='gif',
                fps=fps,
                dpi=dpi,
                downsample_scatter=5000
            )
            animation_paths.append(anim_path)
            print(f"    ✓ Animation created: {anim_path.resolve()}")

        except FileNotFoundError as e:
            print(f"    Warning: FFmpeg not found - skipping animation for {param_name}={param_value}")
        except Exception as e:
            print(f"    Warning: Failed to create animation for {param_name}={param_value}: {e}")

    if animation_paths:
        print(f"  ✓ Created {len(animation_paths)} animations")
    else:
        print(f"  Warning: No animations created")

    return animation_paths


def create_embedding_plots(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray],
    selected_indices: List[int],
    method_name: str,
    output_dir: Path
) -> Path:
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    plt.style.use('default')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if labels is not None:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        scatter = axes[0].scatter(embeddings[:, 0], embeddings[:, 1],
                                c=labels, alpha=0.6, s=20, cmap='tab10')
        axes[0].set_title(f"After Clustering ({n_clusters} clusters)")
        if n_clusters <= 10:
            plt.colorbar(scatter, ax=axes[0], label='Cluster')
    else:
        axes[0].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=20, c='blue')
        axes[0].set_title("No Clustering Info Available")
    axes[0].set_xlabel("Component 1")
    axes[0].set_ylabel("Component 2")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.3, s=20,
                   color='lightgray', label='Unselected')
    if len(selected_indices) > 0:
        axes[1].scatter(embeddings[selected_indices, 0], embeddings[selected_indices, 1],
                       alpha=0.8, s=50, color='red', label='Selected')
    axes[1].set_title(f"Selected Compounds ({len(selected_indices)} selected)")
    axes[1].set_xlabel("Component 1")
    axes[1].set_ylabel("Component 2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if labels is not None:
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(unique_labels) > 0:
            n_clusters = len(unique_labels)

            if n_clusters > 50:
                from scipy.stats import gaussian_kde
                if len(counts) > 1:
                    kde = gaussian_kde(counts)
                    x_range = np.linspace(counts.min(), counts.max(), 200)
                    kde_values = kde(x_range)
                    axes[2].fill_between(x_range, kde_values, alpha=0.7, color='skyblue')
                    axes[2].plot(x_range, kde_values, color='darkblue', linewidth=2)
                    axes[2].set_xlabel("Cluster Size")
                    axes[2].set_ylabel("Density")
                else:
                    axes[2].axvline(counts[0], color='darkblue', linewidth=3)
                    axes[2].set_xlabel("Cluster Size")
                    axes[2].set_ylabel("Density")
                axes[2].set_title(f"Cluster Size Distribution (KDE, {n_clusters} clusters)")
            else:
                axes[2].bar(range(len(unique_labels)), counts, color='skyblue', edgecolor='darkblue')
                axes[2].set_xlabel("Cluster ID")
                axes[2].set_ylabel("Number of Compounds")
                axes[2].set_title(f"Cluster Size Distribution ({n_clusters} clusters)")

                if n_clusters <= 20:
                    axes[2].set_xticks(range(0, n_clusters, max(1, n_clusters // 10)))
                else:
                    axes[2].set_xticks(range(0, n_clusters, n_clusters // 10))

            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, "No valid clusters", ha='center', va='center',
                        transform=axes[2].transAxes)
            axes[2].set_title("Cluster Size Distribution")
    else:
        axes[2].text(0.5, 0.5, "No clustering information", ha='center', va='center',
                    transform=axes[2].transAxes)
        axes[2].set_title("Cluster Size Distribution")

    plt.suptitle(f"{method_name} - Analysis ({len(embeddings)} compounds)", fontsize=16)
    plt.tight_layout()

    plot_file = output_dir / f"{method_name.lower()}_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {plot_file}")

    plt.close()

    return plot_file


def create_pruning_efficiency_timeline(
    all_results: Dict[float, Dict[str, Any]],
    strategy_labels: Dict[float, str],
    output_path: Path,
    figsize: tuple = (12, 6),
    dpi: int = 300
) -> Path:
    strategy_colors = {
        0.0: '#cabde7',
        0.15: '#7e62df',
        0.3: '#452997'
    }

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    for pruning_frac in sorted(all_results.keys()):
        result = all_results[pruning_frac]
        cycle_metrics = pl.DataFrame(result['cycle_metrics'])

        cycles = cycle_metrics['cycle'].to_numpy()
        pool_sizes = cycle_metrics['pool_size'].to_numpy()
        discovery = cycle_metrics.get('top_0_1_pct_discovery', cycle_metrics.get('top_100_discovery', [0]*len(cycles))).to_numpy()

        label = strategy_labels[pruning_frac]
        color = strategy_colors.get(pruning_frac, '#95a5a6')

        ax1.plot(cycles, pool_sizes, color=color, linewidth=2.5, label=f'{label} (Pool)', marker='o', markersize=4)
        ax1.fill_between(cycles, 0, pool_sizes, color=color, alpha=0.15)

        ax2.plot(cycles, discovery, color=color, linewidth=2, linestyle='--',
                marker='s', markersize=4, alpha=0.7)

    ax1.set_xlabel('Cycle', fontsize=13)
    ax1.set_ylabel('Unlabeled Pool Size (compounds)', fontsize=13, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, linewidth=0.75)

    ax2.set_ylabel('Top-0.1% Discovery Rate (%)', fontsize=13, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, 100)

    ax1.set_title('Pruning Efficiency: Pool Size Reduction vs Discovery Progress',
                  fontsize=14, fontweight='bold', pad=15)

    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Pruning efficiency timeline saved: {output_path.resolve()}")
    return output_path


def create_discovery_efficiency_scatter(
    all_results: Dict[float, Dict[str, Any]],
    strategy_labels: Dict[float, str],
    output_path: Path,
    figsize: tuple = (10, 7),
    dpi: int = 300
) -> Path:
    strategy_colors = {
        0.0: '#cabde7',
        0.15: '#7e62df',
        0.3: '#452997'
    }

    fig, ax = plt.subplots(figsize=figsize)

    for pruning_frac in sorted(all_results.keys()):
        result = all_results[pruning_frac]
        cycle_metrics = pl.DataFrame(result['cycle_metrics'])

        cumulative_labeled = cycle_metrics['cumulative_labeled'].to_numpy()
        discovery = cycle_metrics['top_100_discovery'].to_numpy()
        cycles = cycle_metrics['cycle'].to_numpy()

        label = strategy_labels[pruning_frac]
        color = strategy_colors.get(pruning_frac, '#95a5a6')

        ax.plot(cumulative_labeled, discovery, color=color, linewidth=2.5,
               label=label, marker='o', markersize=6, alpha=0.8)

        for i, cycle in enumerate(cycles):
            if cycle % 3 == 0:
                ax.annotate(f'C{cycle}',
                           (cumulative_labeled[i], discovery[i]),
                           textcoords="offset points", xytext=(0,8),
                           ha='center', fontsize=8, alpha=0.7)

    ax.set_xlabel('Cumulative Compounds Evaluated', fontsize=13)
    ax.set_ylabel('Top-100 Discovery Rate (%)', fontsize=13)
    ax.set_title('Discovery Efficiency: How Many Compounds Needed to Find Top-100?',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.75)
    ax.legend(fontsize=11, framealpha=0.9, loc='lower right')
    ax.set_ylim(0, 100)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Discovery efficiency scatter saved: {output_path.resolve()}")
    return output_path


def create_model_quality_facets(
    all_results: Dict[float, Dict[str, Any]],
    strategy_labels: Dict[float, str],
    output_path: Path,
    figsize: tuple = (14, 10),
    dpi: int = 300
) -> Path:
    strategy_colors = {
        0.0: '#cabde7',
        0.15: '#7e62df',
        0.3: '#452997'
    }

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Model Quality on Unlabeled Pool: Ranking Performance',
                 fontsize=16, fontweight='bold', y=0.995)

    metrics = [
        ('unlabeled_top_100_overlap', 'Top-100 Overlap (%)', axes[0, 0]),
        ('unlabeled_top_1000_overlap', 'Top-1000 Overlap (%)', axes[0, 1]),
        ('unlabeled_spearman', 'Spearman Correlation', axes[1, 0]),
        ('spearman_correlation', 'Overall Spearman', axes[1, 1])
    ]

    for metric_name, ylabel, ax in metrics:
        for pruning_frac in sorted(all_results.keys()):
            result = all_results[pruning_frac]
            cycle_metrics = pl.DataFrame(result['cycle_metrics'])

            if metric_name not in cycle_metrics.columns:
                continue

            cycles = cycle_metrics['cycle'].to_numpy()
            values = cycle_metrics[metric_name].to_numpy()

            label = strategy_labels[pruning_frac]
            color = strategy_colors.get(pruning_frac, '#95a5a6')

            ax.plot(cycles, values, color=color, linewidth=2.5,
                   label=label, marker='o', markersize=5)

        ax.set_xlabel('Cycle', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.75)
        ax.legend(fontsize=9, framealpha=0.9)

        if 'overlap' in metric_name.lower():
            ax.set_ylim(0, 100)
        elif 'spearman' in metric_name.lower():
            ax.set_ylim(0, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Model quality facets saved: {output_path.resolve()}")
    return output_path


def create_uncertainty_prediction_snapshots(
    result: Dict[str, Any],
    strategy_label: str,
    output_path: Path,
    cycles_to_show: List[int] = [1, 3, 6, 9],
    figsize: tuple = (12, 10),
    dpi: int = 300
) -> Path:
    df = result['compounds_df']
    cycle_metrics = pl.DataFrame(result['cycle_metrics'])

    final_cycle = cycle_metrics['cycle'].max()
    cycles_to_show = [c for c in cycles_to_show if c <= final_cycle]

    unlabeled_bg = df.filter(pl.col('status') == 'unlabeled')
    if len(unlabeled_bg) > 5000:
        unlabeled_bg = unlabeled_bg.sample(n=5000, seed=123)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{strategy_label}: Prediction vs Uncertainty Evolution',
                 fontsize=16, fontweight='bold', y=0.995)

    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, cycle in enumerate(cycles_to_show):
        row, col = subplot_positions[idx]
        ax = axes[row, col]

        pred_col = f'prediction_cycle_{cycle}'
        unc_col = f'uncertainty_cycle_{cycle}'

        if pred_col not in df.columns or unc_col not in df.columns:
            ax.text(0.5, 0.5, f'Missing data\nfor cycle {cycle}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Cycle {cycle}')
            continue

        if unc_col in unlabeled_bg.columns:
            ax.scatter(
                unlabeled_bg[pred_col],
                unlabeled_bg[unc_col],
                c='#CCCCCC',
                alpha=0.25,
                s=8,
                edgecolors='none',
                label='Unlabeled'
            )

        pruned_at_cycle = df.filter((pl.col('status') == 'pruned') & (pl.col('pruned_cycle') == cycle))
        if len(pruned_at_cycle) > 0:
            if len(pruned_at_cycle) > 2000:
                pruned_at_cycle = pruned_at_cycle.sample(n=2000, seed=123)
            ax.scatter(
                pruned_at_cycle[pred_col],
                pruned_at_cycle[unc_col],
                c='#d89ba5',
                alpha=0.5,
                s=20,
                edgecolors='darkred',
                linewidths=0.3,
                label=f'Pruned this cycle (n={len(pruned_at_cycle)})'
            )

        selected_up_to = df.filter((pl.col('status') == 'labeled') & (pl.col('selected_cycle') <= cycle))
        if len(selected_up_to) > 0:
            ax.scatter(
                selected_up_to[pred_col],
                selected_up_to[unc_col],
                c='#7e62df',
                alpha=0.75,
                s=35,
                edgecolors='black',
                linewidths=0.5,
                label=f'Selected (n={len(selected_up_to)})'
            )

        ax.set_xlabel('Prediction', fontsize=11)
        if col == 0:
            ax.set_ylabel('Uncertainty', fontsize=11)
        ax.set_title(f'Cycle {cycle}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=8, loc='best', framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Uncertainty-prediction snapshots saved: {output_path.resolve()}")
    return output_path


def create_score_ratio_evolution(
    all_results: Dict[float, Dict[str, Any]],
    strategy_labels: Dict[float, str],
    output_path: Path,
    figsize: tuple = (12, 6),
    dpi: int = 300
) -> Path:
    strategy_colors = {
        0.0: '#cabde7',
        0.15: '#7e62df',
        0.3: '#452997'
    }

    fig, ax = plt.subplots(figsize=figsize)

    for pruning_frac in sorted(all_results.keys()):
        result = all_results[pruning_frac]
        cycle_metrics = pl.DataFrame(result['cycle_metrics'])

        cycles = cycle_metrics['cycle'].to_numpy()
        cumulative_ratio = cycle_metrics.get('cumulative_avg_score_ratio', [1.0]*len(cycles)).to_numpy()
        batch_ratio = cycle_metrics.get('batch_avg_score_ratio', [1.0]*len(cycles)).to_numpy()

        label = strategy_labels[pruning_frac]
        color = strategy_colors.get(pruning_frac, '#95a5a6')

        ax.plot(cycles, cumulative_ratio, color=color, linewidth=2.5,
               label=f'{label} (Cumulative)', marker='o', markersize=5, linestyle='-')
        ax.plot(cycles, batch_ratio, color=color, linewidth=1.5,
               marker='s', markersize=4, linestyle='--', alpha=0.6)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5,
              label='Random baseline')

    ax.set_xlabel('Cycle', fontsize=13)
    ax.set_ylabel('Average Score Ratio', fontsize=13)
    ax.set_title('Selection Quality: Score Ratio Evolution (Higher = Better Selection)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.75)
    ax.legend(fontsize=10, framealpha=0.9, loc='best')
    ax.set_ylim(0.8, 2.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Score ratio evolution saved: {output_path.resolve()}")
    return output_path

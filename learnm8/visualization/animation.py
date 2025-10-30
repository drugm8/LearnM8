import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from .utils import downsample_for_viz, detect_benchmark_mode, format_metric_value

logger = logging.getLogger(__name__)


def load_csv_data(output_dir: str) -> Dict[str, pd.DataFrame]:
    output_path = Path(output_dir)

    data = {}

    predictions_file = output_path / 'compounds_final.csv'
    if predictions_file.exists():
        data['predictions'] = pd.read_csv(predictions_file, comment='#')
    else:
        raise FileNotFoundError(f"Required file not found: {predictions_file}")

    metrics_file = output_path / 'cycle_metrics.csv'
    if metrics_file.exists():
        data['metrics'] = pd.read_csv(metrics_file, comment='#')
    else:
        raise FileNotFoundError(f"Required file not found: {metrics_file}")

    selection_file = output_path / 'selection_history.csv'
    if selection_file.exists():
        data['selections'] = pd.read_csv(selection_file, comment='#')
    else:
        logger.warning(f"Selection history not found: {selection_file}")
        data['selections'] = pd.DataFrame()

    return data


def _get_strategy_color_map(strategies):
    """Create a color map for acquisition strategies."""
    strategy_colors = {
        'random': 'yellow',
        'greedy': 'red',
        'diverse': 'blue',
        'ucb': 'orange',
        'ei': 'purple',
        'pi': 'green',
        'thompson': 'cyan',
        'simulated_annealing': 'magenta',
        'bitbirch': 'brown'
    }

    color_map = {}
    fallback_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fallback_idx = 0

    for strategy in strategies:
        if strategy in strategy_colors:
            color_map[strategy] = strategy_colors[strategy]
        else:
            color_map[strategy] = fallback_colors[fallback_idx % len(fallback_colors)]
            fallback_idx += 1

    return color_map


def create_dashboard_animation_from_csv(
    output_dir: str,
    output_file: Optional[str] = None,
    format: str = 'mp4',
    fps: int = 2,
    dpi: int = 100,
    downsample_scatter: int = 5000
) -> animation.Animation:
    data = load_csv_data(output_dir)

    predictions_df = data['predictions']
    metrics_df = data['metrics']
    selections_df = data['selections']

    is_benchmark = detect_benchmark_mode(metrics_df)

    pred_cols = [col for col in predictions_df.columns if col.startswith('prediction_cycle_')]
    unc_cols = [col for col in predictions_df.columns if col.startswith('uncertainty_cycle_')]

    # Detect target column (oracle values) by elimination
    system_cols = {'ID', 'SMILES', 'status', 'labeled_cycle', 'selected_cycle', 'pruned_cycle'}
    target_col = None
    for col in predictions_df.columns:
        if col not in system_cols and not col.startswith('prediction_cycle_') and not col.startswith('uncertainty_cycle_'):
            target_col = col
            break

    if target_col is None:
        logger.warning("Could not detect target column for true values")
    else:
        logger.info(f"Detected target column: {target_col}")

    n_cycles = len(pred_cols)
    if n_cycles == 0:
        raise ValueError("No prediction cycles found in data")

    logger.info(f"Creating animation for {n_cycles} cycles (benchmark mode: {is_benchmark})")

    global_pred_min = float('inf')
    global_pred_max = float('-inf')
    global_unc_max = 0.0

    for pred_col in pred_cols:
        valid_preds = predictions_df[pred_col].values
        valid_preds = valid_preds[~np.isnan(valid_preds)]
        if len(valid_preds) > 0:
            global_pred_min = min(global_pred_min, valid_preds.min())
            global_pred_max = max(global_pred_max, valid_preds.max())

    for unc_col in unc_cols:
        valid_uncs = predictions_df[unc_col].values
        valid_uncs = valid_uncs[~np.isnan(valid_uncs)]
        if len(valid_uncs) > 0:
            global_unc_max = max(global_unc_max, valid_uncs.max())

    if global_pred_min == float('inf'):
        global_pred_min = 0.0
        global_pred_max = 1.0
    if global_unc_max == 0.0:
        global_unc_max = 1.0

    pred_range = global_pred_max - global_pred_min
    ax2_xlim = (global_pred_min - pred_range * 0.05, global_pred_max + pred_range * 0.05)
    ax2_ylim = (0, global_unc_max * 1.1)

    unique_strategies = []
    strategy_color_map = {}
    if not selections_df.empty and 'strategy' in selections_df.columns:
        unique_strategies = selections_df['strategy'].unique().tolist()
        strategy_color_map = _get_strategy_color_map(unique_strategies)

    if is_benchmark:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ((ax1, ax2), (ax3, ax4)) = axes
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ((ax1, ax2), (ax3, ax4)) = axes

    ax1.set_title('Uncertainty vs Prediction')
    ax1.set_xlabel('Predicted Activity')
    ax1.set_ylabel('Uncertainty')

    ax2.set_title('Discovery Metrics')
    ax2.set_xlabel('Data Explored')
    ax2.set_ylabel('Discovery Rate (%)')

    ax3.set_title('Cumulative Best Value Found')
    ax3.set_xlabel('Data Explored')
    ax3.set_ylabel('Best Value (Lower is Better)')

    ax4.set_title('Model Ranking Metrics')
    ax4.set_xlabel('Data Explored')
    ax4.set_ylabel('Metric Value')

    # Panel A: Uncertainty vs Prediction scatter
    scatter1 = ax1.scatter([], [], c=[], cmap='viridis', alpha=0.6, s=20)

    strategy_scatters = {}
    if unique_strategies:
        for strategy in unique_strategies:
            color = strategy_color_map[strategy]
            scatter = ax1.scatter([], [], c=color, marker='x', s=100, linewidths=2, label=strategy, alpha=0.8)
            strategy_scatters[strategy] = scatter

    if strategy_scatters:
        ax1.legend(loc='upper right')

    # Panel B: Discovery metrics (6 line plots)
    line_top10_disc, = ax2.plot([], [], 'r-o', linewidth=2, markersize=4, label='Top-10')
    line_top100_disc, = ax2.plot([], [], 'b-s', linewidth=2, markersize=4, label='Top-100')
    line_top01pct_disc, = ax2.plot([], [], 'g-^', linewidth=2, markersize=4, label='Top-0.1%')
    line_top1pct_disc, = ax2.plot([], [], 'm-d', linewidth=2, markersize=4, label='Top-1%')

    ax2_twin = ax2.twinx()
    line_batch_ratio, = ax2_twin.plot([], [], 'c--', linewidth=2, markersize=4, label='Batch Ratio')
    line_cumul_ratio, = ax2_twin.plot([], [], 'orange', linestyle='--', linewidth=2, markersize=4, label='Cumul Ratio')
    ax2_twin.set_ylabel('Avg Score Ratio', color='darkcyan', fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor='darkcyan')

    ax2.legend(loc='upper left', fontsize=8)
    ax2_twin.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(0, 100)

    # Panel D: Model ranking metrics (3 line plots)
    line_spearman, = ax4.plot([], [], 'b-o', linewidth=2, markersize=4, label='Spearman ρ')

    ax4_twin1 = ax4.twinx()
    line_top1000_overlap, = ax4_twin1.plot([], [], 'g-s', linewidth=2, markersize=4, label='Top-1000 Overlap')

    ax4_twin2 = ax4.twinx()
    ax4_twin2.spines['right'].set_position(('outward', 60))
    line_top100_overlap, = ax4_twin2.plot([], [], 'r-^', linewidth=2, markersize=4, label='Top-100 Overlap')

    ax4.set_ylabel('Spearman ρ', color='blue', fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4.set_ylim(-1, 1)

    ax4_twin1.set_ylabel('Top-1000 Overlap (%)', color='green', fontweight='bold')
    ax4_twin1.tick_params(axis='y', labelcolor='green')
    ax4_twin1.set_ylim(0, 100)

    ax4_twin2.set_ylabel('Top-100 Overlap (%)', color='red', fontweight='bold')
    ax4_twin2.tick_params(axis='y', labelcolor='red')
    ax4_twin2.set_ylim(0, 100)

    lines = [line_spearman] + [line_top1000_overlap] + [line_top100_overlap]
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center left', fontsize=8)

    metric_text = None
    regression_line = None

    def init():
        scatter1.set_offsets(np.empty((0, 2)))
        scatter1.set_array(np.array([]))
        for scatter in strategy_scatters.values():
            scatter.set_offsets(np.empty((0, 2)))

        # Panel B: Discovery metrics
        line_top10_disc.set_data([], [])
        line_top100_disc.set_data([], [])
        line_top01pct_disc.set_data([], [])
        line_top1pct_disc.set_data([], [])
        line_batch_ratio.set_data([], [])
        line_cumul_ratio.set_data([], [])

        # Panel D: Model ranking metrics
        line_spearman.set_data([], [])
        line_top1000_overlap.set_data([], [])
        line_top100_overlap.set_data([], [])

        artists = [scatter1]
        artists.extend(strategy_scatters.values())
        artists.extend([line_top10_disc, line_top100_disc, line_top01pct_disc, line_top1pct_disc,
                       line_batch_ratio, line_cumul_ratio,
                       line_spearman, line_top1000_overlap, line_top100_overlap])
        return tuple(artists)

    def update(cycle_idx):
        nonlocal metric_text, regression_line

        cycle_col = pred_cols[cycle_idx]
        cycle_num = int(cycle_col.split('_')[-1])

        predictions = predictions_df[cycle_col].values

        # Panel A: Uncertainty vs Prediction scatter
        if cycle_idx < len(unc_cols):
            unc_col = unc_cols[cycle_idx]
            uncertainties = predictions_df[unc_col].values

            valid_mask = ~np.isnan(predictions) & ~np.isnan(uncertainties)
            pred_clean = predictions[valid_mask]
            unc_clean = uncertainties[valid_mask]

            if len(pred_clean) > 0:
                if len(pred_clean) > downsample_scatter:
                    sample_indices = np.random.choice(len(pred_clean), downsample_scatter, replace=False)
                    pred_clean = pred_clean[sample_indices]
                    unc_clean = unc_clean[sample_indices]

                scatter1.set_offsets(np.c_[pred_clean, unc_clean])
                scatter1.set_array(pred_clean)

                ax1.set_xlim(ax2_xlim)
                ax1.set_ylim(ax2_ylim)

                if not selections_df.empty and strategy_scatters:
                    selected_up_to_cycle = selections_df[selections_df['cycle'] <= cycle_num]

                    for strategy in unique_strategies:
                        strategy_scatter = strategy_scatters[strategy]
                        strategy_selections = selected_up_to_cycle[selected_up_to_cycle['strategy'] == strategy]

                        if len(strategy_selections) > 0:
                            selected_preds = strategy_selections['prediction_at_selection'].values
                            selected_uncs = strategy_selections['uncertainty_at_selection'].values

                            valid_sel_mask = ~np.isnan(selected_preds) & ~np.isnan(selected_uncs)
                            selected_preds = selected_preds[valid_sel_mask]
                            selected_uncs = selected_uncs[valid_sel_mask]

                            if len(selected_preds) > 0:
                                strategy_scatter.set_offsets(np.c_[selected_preds, selected_uncs])
                            else:
                                strategy_scatter.set_offsets(np.empty((0, 2)))
                        else:
                            strategy_scatter.set_offsets(np.empty((0, 2)))
                else:
                    for scatter in strategy_scatters.values():
                        scatter.set_offsets(np.empty((0, 2)))
        else:
            scatter1.set_offsets(np.empty((0, 2)))
            for scatter in strategy_scatters.values():
                scatter.set_offsets(np.empty((0, 2)))

        # Panel B: Discovery Metrics
        cycles = np.arange(cycle_idx + 1)
        top10_values = metrics_df['top_10_discovery'].iloc[:cycle_idx + 1].values if 'top_10_discovery' in metrics_df.columns else np.zeros(cycle_idx + 1)
        top100_values = metrics_df['top_100_discovery'].iloc[:cycle_idx + 1].values if 'top_100_discovery' in metrics_df.columns else np.zeros(cycle_idx + 1)
        top01pct_values = metrics_df['top_0_1_pct_discovery'].iloc[:cycle_idx + 1].values if 'top_0_1_pct_discovery' in metrics_df.columns else np.zeros(cycle_idx + 1)
        top1pct_values = metrics_df['top_1_pct_discovery'].iloc[:cycle_idx + 1].values if 'top_1_pct_discovery' in metrics_df.columns else np.zeros(cycle_idx + 1)
        batch_ratio_values = metrics_df['batch_avg_score_ratio'].iloc[:cycle_idx + 1].values if 'batch_avg_score_ratio' in metrics_df.columns else np.ones(cycle_idx + 1)
        cumul_ratio_values = metrics_df['cumulative_avg_score_ratio'].iloc[:cycle_idx + 1].values if 'cumulative_avg_score_ratio' in metrics_df.columns else np.ones(cycle_idx + 1)

        # Calculate percentage explored for x-axis coordinates
        n_total = len(predictions_df)
        pct_explored = []
        for i in range(cycle_idx + 1):
            cumulative = metrics_df['cumulative_labeled'].iloc[i] if 'cumulative_labeled' in metrics_df.columns else 0
            pct = (cumulative / n_total) * 100 if n_total > 0 else 0
            pct_explored.append(pct)

        line_top10_disc.set_data(pct_explored, top10_values)
        line_top100_disc.set_data(pct_explored, top100_values)
        line_top01pct_disc.set_data(pct_explored, top01pct_values)
        line_top1pct_disc.set_data(pct_explored, top1pct_values)
        line_batch_ratio.set_data(pct_explored, batch_ratio_values)
        line_cumul_ratio.set_data(pct_explored, cumul_ratio_values)

        max_pct = pct_explored[-1] if len(pct_explored) > 0 else 5.0
        ax2.set_xlim(0, max_pct * 1.1)
        ax2.set_xlabel('Data Explored (%)', fontsize=10)
        ax2.set_xticks(pct_explored)
        x_labels = [f'{pct:.1f}%\n(c{c})' for c, pct in zip(cycles, pct_explored)]
        ax2.set_xticklabels(x_labels, fontsize=8, rotation=0)

        # Panel C: Cumulative Best Value Found
        ax3.clear()
        ax3.set_title('Cumulative Best Value Found')
        ax3.set_xlabel('Data Explored (%)', fontsize=10)
        ax3.set_ylabel('Best Value (Lower is Better)')

        if not selections_df.empty:
            selected_up_to_cycle = selections_df[selections_df['cycle'] <= cycle_num]
            if len(selected_up_to_cycle) > 0:
                cumulative_best = []
                best_so_far = float('inf')

                for c in range(cycle_num + 1):
                    cycle_selections = selections_df[selections_df['cycle'] == c]
                    if len(cycle_selections) > 0:
                        cycle_values = cycle_selections['measured_value'].values
                        cycle_best = np.nanmin(cycle_values)
                        best_so_far = min(best_so_far, cycle_best)
                    cumulative_best.append(best_so_far)

                pct_range = pct_explored[:cycle_num + 1]
                ax3.plot(pct_range, cumulative_best, 'b-', linewidth=2, marker='o', markersize=4, label='Best Found')

                if target_col and target_col in predictions_df.columns:
                    true_best = predictions_df[target_col].min()
                    if not np.isnan(true_best):
                        ax3.axhline(y=true_best, color='g', linestyle='--', linewidth=2, alpha=0.7, label='True Best')

                max_pct = pct_explored[-1] if len(pct_explored) > 0 else 5.0
                ax3.set_xlim(0, max_pct * 1.1)
                if len(cumulative_best) > 0:
                    y_range = cumulative_best[0] - cumulative_best[-1]
                    ax3.set_ylim(cumulative_best[-1] - y_range * 0.2, cumulative_best[0] + y_range * 0.2)
                ax3.legend(loc='upper right')
                ax3.grid(True, alpha=0.3)

                # Update x-axis labels for Panel C
                ax3.set_xticks(pct_range)
                x_labels_c = [f'{pct:.1f}%\n(c{c})' for c, pct in zip(range(len(pct_range)), pct_range)]
                ax3.set_xticklabels(x_labels_c, fontsize=8, rotation=0)

        # Panel D: Model Ranking Metrics
        spearman_values = metrics_df['unlabeled_spearman_correlation'].iloc[:cycle_idx + 1].values if 'unlabeled_spearman_correlation' in metrics_df.columns else np.zeros(cycle_idx + 1)
        top1000_overlap_values = metrics_df['unlabeled_top_1000_overlap'].iloc[:cycle_idx + 1].values if 'unlabeled_top_1000_overlap' in metrics_df.columns else np.zeros(cycle_idx + 1)
        top100_overlap_values = metrics_df['unlabeled_top_100_overlap'].iloc[:cycle_idx + 1].values if 'unlabeled_top_100_overlap' in metrics_df.columns else np.zeros(cycle_idx + 1)

        line_spearman.set_data(pct_explored, spearman_values)
        line_top1000_overlap.set_data(pct_explored, top1000_overlap_values)
        line_top100_overlap.set_data(pct_explored, top100_overlap_values)

        ax4.set_xlim(0, max_pct * 1.1)
        ax4.set_xlabel('Data Explored (%)', fontsize=10)
        ax4.set_xticks(pct_explored)
        ax4.set_xticklabels(x_labels, fontsize=8, rotation=0)

        fig.suptitle(f'Active Learning Progress - Cycle {cycle_num}', fontsize=14, fontweight='bold')

        artists = [scatter1]
        artists.extend(strategy_scatters.values())
        artists.extend([line_top10_disc, line_top100_disc, line_top01pct_disc, line_top1pct_disc,
                       line_batch_ratio, line_cumul_ratio,
                       line_spearman, line_top1000_overlap, line_top100_overlap])

        return tuple(artists)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_cycles,
        interval=1000 // fps,
        blit=False,
        repeat=True
    )

    plt.tight_layout()

    if output_file:
        _save_animation(anim, output_file, format, fps, dpi)

    return anim


def _save_animation(anim: animation.Animation, output_path: str, format: str, fps: int, dpi: int) -> None:
    from pathlib import Path

    try:
        if format == 'mp4':
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=1800, codec='h264')
                anim.save(output_path, writer=writer, dpi=dpi)
                logger.info(f"Saved animation as MP4: {output_path}")
            except FileNotFoundError:
                logger.warning("FFmpeg not found. Falling back to GIF format.")
                logger.info("To install FFmpeg: conda install ffmpeg")

                gif_path = str(Path(output_path).with_suffix('.gif'))
                writer = animation.PillowWriter(fps=fps)
                anim.save(gif_path, writer=writer, dpi=dpi)
                logger.info(f"Saved animation as GIF instead: {gif_path}")
                print(f"\n⚠️  FFmpeg not found. Saved as GIF: {gif_path}")
                print("To save as MP4 in the future, install FFmpeg: conda install ffmpeg\n")

        elif format == 'gif':
            writer = animation.PillowWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=dpi)
            logger.info(f"Saved animation as GIF: {output_path}")
        elif format == 'html':
            with open(output_path, 'w') as f:
                f.write(anim.to_html5_video())
            logger.info(f"Saved animation as HTML: {output_path}")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'mp4', 'gif', or 'html'")
    except Exception as e:
        if 'FFmpeg' not in str(e):
            logger.error(f"Failed to save animation: {e}")
            raise

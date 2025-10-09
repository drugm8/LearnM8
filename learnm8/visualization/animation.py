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

    predictions_file = output_path / 'predictions_by_cycle.csv'
    if predictions_file.exists():
        data['predictions'] = pd.read_csv(predictions_file)
    else:
        raise FileNotFoundError(f"Required file not found: {predictions_file}")

    metrics_file = output_path / 'cycle_metrics.csv'
    if metrics_file.exists():
        data['metrics'] = pd.read_csv(metrics_file, comment='#')
    else:
        raise FileNotFoundError(f"Required file not found: {metrics_file}")

    selection_file = output_path / 'selection_history.csv'
    if selection_file.exists():
        data['selections'] = pd.read_csv(selection_file)
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

    ax1.set_title('Predicted vs True Activity' if is_benchmark else 'Prediction Evolution')
    ax1.set_xlabel('True Activity' if is_benchmark else 'Cycle')
    ax1.set_ylabel('Predicted Activity')

    ax2.set_title('Uncertainty vs Prediction')
    ax2.set_xlabel('Predicted Activity')
    ax2.set_ylabel('Uncertainty')

    ax3.set_title('Cumulative Best Value Found')
    ax3.set_xlabel('Cycle')
    ax3.set_ylabel('Best Value (Lower is Better)')

    ax4.set_title('Cumulative Performance')
    ax4.set_xlabel('Cycle')
    ax4.set_ylabel('Metric Value')

    scatter1 = ax1.scatter([], [], c=[], cmap='RdYlGn', alpha=0.6, s=20)
    scatter2 = ax2.scatter([], [], c=[], cmap='viridis', alpha=0.6, s=20)

    strategy_scatters = {}
    if unique_strategies:
        for strategy in unique_strategies:
            color = strategy_color_map[strategy]
            scatter = ax2.scatter([], [], c=color, marker='x', s=100, linewidths=2, label=strategy, alpha=0.8)
            strategy_scatters[strategy] = scatter

    line_rmse, = ax4.plot([], [], 'b-', linewidth=2, label='RMSE')
    line_r2, = ax4.plot([], [], 'g-', linewidth=2, label='R²')

    if strategy_scatters:
        ax2.legend(loc='upper right')

    if is_benchmark and 'top_10_percent_overlap' in metrics_df.columns:
        ax4_twin = ax4.twinx()
        line_top_k, = ax4_twin.plot([], [], 'r-', linewidth=2, label='Top-10% Recovery')
        ax4_twin.set_ylabel('Top-10% Recovery (%)', color='r')
        ax4_twin.tick_params(axis='y', labelcolor='r')
    else:
        ax4_twin = None
        line_top_k = None

    ax4.legend(loc='upper left')
    if ax4_twin:
        ax4_twin.legend(loc='upper right')

    metric_text = None
    regression_line = None

    def init():
        scatter1.set_offsets(np.empty((0, 2)))
        scatter1.set_array(np.array([]))
        scatter2.set_offsets(np.empty((0, 2)))
        scatter2.set_array(np.array([]))
        for scatter in strategy_scatters.values():
            scatter.set_offsets(np.empty((0, 2)))
        line_rmse.set_data([], [])
        line_r2.set_data([], [])
        if line_top_k:
            line_top_k.set_data([], [])

        artists = [scatter1, scatter2, line_rmse, line_r2]
        artists.extend(strategy_scatters.values())
        return tuple(artists)

    def update(cycle_idx):
        nonlocal metric_text, regression_line

        cycle_col = pred_cols[cycle_idx]
        cycle_num = int(cycle_col.split('_')[-1])

        predictions = predictions_df[cycle_col].values
        true_values = predictions_df['final_oracle_value'].values if 'final_oracle_value' in predictions_df.columns else None

        if is_benchmark and true_values is not None:
            valid_mask = ~np.isnan(predictions) & ~np.isnan(true_values)
            pred_clean = predictions[valid_mask]
            true_clean = true_values[valid_mask]

            if len(pred_clean) > 0:
                if len(pred_clean) > downsample_scatter:
                    sample_indices = np.random.choice(len(pred_clean), downsample_scatter, replace=False)
                    pred_clean = pred_clean[sample_indices]
                    true_clean = true_clean[sample_indices]

                scatter1.set_offsets(np.c_[true_clean, pred_clean])
                scatter1.set_array(pred_clean)

                all_values = np.concatenate([pred_clean, true_clean])
                vmin, vmax = all_values.min(), all_values.max()
                margin = (vmax - vmin) * 0.1 if vmax > vmin else 1.0
                ax1.set_xlim(vmin - margin, vmax + margin)
                ax1.set_ylim(vmin - margin, vmax + margin)

                while len(ax1.lines) > 0:
                    ax1.lines[0].remove()
                ax1.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin], 'k--', alpha=0.5, linewidth=1)

                if len(pred_clean) > 1:
                    coeffs = np.polyfit(true_clean, pred_clean, 1)
                    poly_line = np.poly1d(coeffs)
                    x_fit = np.array([vmin - margin, vmax + margin])
                    y_fit = poly_line(x_fit)
                    regression_line, = ax1.plot(x_fit, y_fit, 'r-', alpha=0.6, linewidth=2, label='Best Fit')

                if cycle_idx < len(metrics_df):
                    r2 = metrics_df['r2_score'].iloc[cycle_idx]
                    spearman = metrics_df['spearman_correlation'].iloc[cycle_idx] if 'spearman_correlation' in metrics_df.columns else None

                    if metric_text is not None:
                        metric_text.remove()

                    metric_str = f"R² = {r2:.3f}"
                    if spearman is not None and not np.isnan(spearman):
                        metric_str += f"\nρ = {spearman:.3f}"

                    metric_text = ax1.text(0.02, 0.98, metric_str,
                                          transform=ax1.transAxes,
                                          fontsize=12, fontweight='bold',
                                          verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
        else:
            cycles_array = np.arange(cycle_idx + 1)
            pred_means = []
            for i in range(cycle_idx + 1):
                col_preds = predictions_df[pred_cols[i]].values
                pred_means.append(np.nanmean(col_preds))
            scatter1.set_offsets(np.c_[cycles_array, pred_means])
            scatter1.set_array(np.array(pred_means))
            ax1.set_xlim(-0.5, n_cycles + 0.5)

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

                scatter2.set_offsets(np.c_[pred_clean, unc_clean])
                scatter2.set_array(pred_clean)

                ax2.set_xlim(ax2_xlim)
                ax2.set_ylim(ax2_ylim)

                if not selections_df.empty and strategy_scatters:
                    selected_up_to_cycle = selections_df[selections_df['selected_cycle'] <= cycle_num]

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
            scatter2.set_offsets(np.empty((0, 2)))
            for scatter in strategy_scatters.values():
                scatter.set_offsets(np.empty((0, 2)))

        ax3.clear()
        ax3.set_title('Cumulative Best Value Found')
        ax3.set_xlabel('Cycle')
        ax3.set_ylabel('Best Value (Lower is Better)')

        if not selections_df.empty:
            selected_up_to_cycle = selections_df[selections_df['selected_cycle'] <= cycle_num]
            if len(selected_up_to_cycle) > 0:
                cumulative_best = []
                best_so_far = float('inf')

                for c in range(cycle_num + 1):
                    cycle_selections = selections_df[selections_df['selected_cycle'] == c]
                    if len(cycle_selections) > 0:
                        cycle_values = cycle_selections['oracle_measured_value'].values
                        cycle_best = np.nanmin(cycle_values)
                        best_so_far = min(best_so_far, cycle_best)
                    cumulative_best.append(best_so_far)

                cycles_range = np.arange(len(cumulative_best))
                ax3.plot(cycles_range, cumulative_best, 'b-', linewidth=2, marker='o', markersize=4, label='Best Found')

                if 'final_oracle_value' in predictions_df.columns:
                    true_best = predictions_df['final_oracle_value'].min()
                    if not np.isnan(true_best):
                        ax3.axhline(y=true_best, color='g', linestyle='--', linewidth=2, alpha=0.7, label='True Best')

                ax3.set_xlim(-0.5, n_cycles + 0.5)
                if len(cumulative_best) > 0:
                    y_range = cumulative_best[0] - cumulative_best[-1]
                    ax3.set_ylim(cumulative_best[-1] - y_range * 0.2, cumulative_best[0] + y_range * 0.2)
                ax3.legend(loc='upper right')
                ax3.grid(True, alpha=0.3)

        cycles = np.arange(cycle_idx + 1)
        rmse_values = metrics_df['rmse'].iloc[:cycle_idx + 1].values
        r2_values = metrics_df['r2_score'].iloc[:cycle_idx + 1].values

        line_rmse.set_data(cycles, rmse_values)
        line_r2.set_data(cycles, r2_values)

        ax4.set_xlim(-0.5, n_cycles + 0.5)
        ax4.set_ylim(0, max(rmse_values.max() * 1.1, 1.0))

        if line_top_k and 'top_10_percent_overlap' in metrics_df.columns:
            top_k_values = metrics_df['top_10_percent_overlap'].iloc[:cycle_idx + 1].values
            line_top_k.set_data(cycles, top_k_values)
            ax4_twin.set_ylim(0, 100)

        fig.suptitle(f'Active Learning Progress - Cycle {cycle_num}', fontsize=14, fontweight='bold')

        artists = [scatter1, scatter2, line_rmse, line_r2]
        artists.extend(strategy_scatters.values())
        if line_top_k:
            artists.append(line_top_k)

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

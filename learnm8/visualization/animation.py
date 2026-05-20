import logging
import re
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


_BENCHMARK_COLS = ('top_10_percent_overlap', 'ef_1_0', 'ground_truth_ef_1_0')


def detect_benchmark_mode(metrics_df: pl.DataFrame) -> bool:
    return any(col in metrics_df.columns for col in _BENCHMARK_COLS)


_PARQUET_PATTERN = re.compile(r'prediction_cycle_(\d+)\.parquet$')


def _discover_prediction_parquets(output_dir: Path) -> list[tuple[int, Path]]:
    """Discover all prediction parquet files in output_dir, sorted by cycle.

    Returns a list of (cycle_number, path) tuples sorted by cycle number.
    """
    paths = []
    for path in output_dir.glob('prediction_cycle_*.parquet'):
        match = _PARQUET_PATTERN.search(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    paths.sort(key=lambda x: x[0])
    return paths


def _read_results_file(path: Path, comment_prefix: str = '#') -> pl.DataFrame:
    """Auto-detect and read a results file as .parquet or .csv.

    Args:
        path: Path to the results file (with or without extension).
        comment_prefix: Comment prefix for CSV metadata comments.

    Returns:
        DataFrame loaded from the file.

    Raises:
        FileNotFoundError: If the file cannot be found with any extension.
    """
    if path.exists():
        if path.suffix == '.parquet':
            return pl.scan_parquet(path).collect()
        if path.suffix == '.csv':
            return pl.read_csv(path, ignore_errors=True, comment_prefix=comment_prefix)
        raise FileNotFoundError(f"Unrecognized results file extension: {path.suffix}")

    parquet_path = path.with_suffix('.parquet')
    if parquet_path.exists():
        return pl.scan_parquet(parquet_path).collect()

    csv_path = path.with_suffix('.csv')
    if csv_path.exists():
        return pl.read_csv(csv_path, ignore_errors=True, comment_prefix=comment_prefix)

    raise FileNotFoundError(
        f"Results file not found: {path} (tried .parquet and .csv extensions)"
    )


def load_csv_data(output_dir: str) -> dict:
    """Load visualization data from CSV and parquet files.

    Args:
        output_dir: Directory containing output files and prediction parquets

    Returns:
        Dictionary with:
            - 'compounds': narrow master DataFrame
            - 'metrics': cycle_metrics DataFrame
            - 'selections': selection_history DataFrame
            - 'parquets': list of (cycle_number, Path) for prediction parquets

    Raises:
        FileNotFoundError: If required files are missing
    """
    output_path = Path(output_dir)

    data = {}

    compounds_file = output_path / 'compounds_final'
    data['compounds'] = _read_results_file(compounds_file)

    metrics_file = output_path / 'cycle_metrics'
    data['metrics'] = _read_results_file(metrics_file)

    selection_file = output_path / 'selection_history'
    try:
        data['selections'] = _read_results_file(selection_file)
    except FileNotFoundError:
        logger.warning(f"Selection history not found: {selection_file}")
        data['selections'] = pl.DataFrame()

    data['parquets'] = _discover_prediction_parquets(output_path)
    if not data['parquets']:
        raise FileNotFoundError(
            f"No prediction parquet files found in {output_path}. "
            f"Expected prediction_cycle_*.parquet (post-migration format)."
        )

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
    output_file: str | None = None,
    format: str = 'mp4',
    fps: int = 2,
    dpi: int = 100,
    downsample_scatter: int = 5000
) -> animation.Animation:
    data = load_csv_data(output_dir)

    compounds_df = data['compounds']
    metrics_df = data['metrics']
    selections_df = data['selections']
    parquets: list[tuple[int, Path]] = data['parquets']

    is_benchmark = detect_benchmark_mode(metrics_df)

    # Detect target column (oracle values) by elimination — narrow master DF
    # has only the system columns plus the target column.
    system_cols = {'ID', 'SMILES', 'status', 'labeled_cycle', 'selected_cycle', 'pruned_cycle'}
    target_col = None
    for col in compounds_df.columns:
        if col not in system_cols:
            target_col = col
            break

    if target_col is None:
        logger.warning("Could not detect target column for true values")
    else:
        logger.info(f"Detected target column: {target_col}")

    n_cycles = len(parquets)
    if n_cycles == 0:
        raise ValueError("No prediction cycles found in data")

    logger.info(f"Creating animation for {n_cycles} cycles (benchmark mode: {is_benchmark})")

    # Compute global axis ranges via lazy scan over each parquet, one at a time,
    # so we never materialise all cycles' predictions at once.
    global_pred_min = float('inf')
    global_pred_max = float('-inf')
    global_unc_max = 0.0

    for _, parquet_path in parquets:
        agg = (
            pl.scan_parquet(parquet_path)
            .select([
                pl.col('prediction').min().alias('pmin'),
                pl.col('prediction').max().alias('pmax'),
                pl.col('uncertainty').max().alias('umax')
                if 'uncertainty' in pl.scan_parquet(parquet_path).collect_schema().names()
                else pl.lit(None).alias('umax'),
            ])
            .collect()
        )
        if agg.height > 0:
            row = agg.row(0, named=True)
            if row['pmin'] is not None:
                global_pred_min = min(global_pred_min, float(row['pmin']))
            if row['pmax'] is not None:
                global_pred_max = max(global_pred_max, float(row['pmax']))
            if row.get('umax') is not None:
                global_unc_max = max(global_unc_max, float(row['umax']))

    if global_pred_min == float('inf'):
        global_pred_min = 0.0
        global_pred_max = 1.0

    has_uncertainty = global_unc_max > 0.0
    if not has_uncertainty:
        global_unc_max = 1.0

    pred_range = global_pred_max - global_pred_min
    ax2_xlim = (global_pred_min - pred_range * 0.05, global_pred_max + pred_range * 0.05)
    ax2_ylim = (0, global_unc_max * 1.1)

    # Compute global ratio bounds for Panel B twin axis
    global_ratio_lo = 0.0
    global_ratio_hi = 1.0
    if 'batch_score_improvement_ratio' in metrics_df.columns:
        batch_r = metrics_df.get_column('batch_score_improvement_ratio').to_numpy()
        cumul_r = metrics_df.get_column('cumulative_score_improvement_ratio').to_numpy()
        all_ratios = np.concatenate([batch_r, cumul_r])
        all_ratios = all_ratios[np.isfinite(all_ratios)]
        if len(all_ratios) > 0:
            global_ratio_lo = min(0.0, float(np.nanmin(all_ratios)))
            global_ratio_hi = float(np.nanmax(all_ratios))
    ratio_pad = (global_ratio_hi - global_ratio_lo) * 0.1 if global_ratio_hi > global_ratio_lo else 0.1

    # Get unique strategies for color mapping
    unique_strategies = []
    strategy_color_map = {}
    if selections_df.height > 0 and 'strategy' in selections_df.columns:
        unique_strategies = selections_df.get_column('strategy').unique().to_list()
        strategy_color_map = _get_strategy_color_map(unique_strategies)

    if is_benchmark:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ((ax1, ax2), (ax3, ax4)) = axes
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ((ax1, ax2), (ax3, ax4)) = axes

    if has_uncertainty:
        ax1.set_title('Uncertainty vs Prediction')
        ax1.set_xlabel('Predicted Activity')
        ax1.set_ylabel('Uncertainty')
    else:
        ax1.set_title('Prediction Distribution')
        ax1.set_xlabel('Predicted Activity')
        ax1.set_ylabel('Count')

    ax2.set_title('Discovery Metrics')
    ax2.set_xlabel('Data Explored')
    ax2.set_ylabel('Discovery Rate (%)')

    ax3.set_title('Cumulative Best Value Found')
    ax3.set_xlabel('Data Explored')
    ax3.set_ylabel('Best Value (Lower is Better)')

    ax4.set_title('Model Ranking Metrics')
    ax4.set_xlabel('Data Explored')
    ax4.set_ylabel('Metric Value')

    # Panel A scatter/histogram setup
    scatter1 = None
    strategy_scatters: dict = {}
    if has_uncertainty:
        scatter1 = ax1.scatter([], [], c=[], cmap='viridis', alpha=0.6, s=20)
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

    lines = [line_spearman, line_top1000_overlap, line_top100_overlap]
    labels = [line.get_label() for line in lines]
    ax4.legend(lines, labels, loc='center left', fontsize=8)

    metric_text = None
    regression_line = None

    def init():
        if scatter1 is not None:
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

        artists = [scatter1] if scatter1 is not None else []
        artists.extend(strategy_scatters.values())
        artists.extend([line_top10_disc, line_top100_disc, line_top01pct_disc, line_top1pct_disc,
                       line_batch_ratio, line_cumul_ratio,
                       line_spearman, line_top1000_overlap, line_top100_overlap])
        return tuple(artists)

    def update(cycle_idx):
        nonlocal metric_text, regression_line

        cycle_num, parquet_path = parquets[cycle_idx]

        # Lazy scan: load only this cycle's predictions, never all cycles at once.
        cycle_lf = pl.scan_parquet(parquet_path)
        cycle_schema = cycle_lf.collect_schema().names()
        select_cols = ['prediction']
        if 'uncertainty' in cycle_schema:
            select_cols.append('uncertainty')
        cycle_df = cycle_lf.select(select_cols).collect()
        predictions = cycle_df.get_column('prediction').to_numpy()

        # Panel A: scatter (with uncertainty) or histogram (without)
        if has_uncertainty:
            if 'uncertainty' in cycle_df.columns:
                uncertainties = cycle_df.get_column('uncertainty').to_numpy()

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

                    if selections_df.height > 0 and strategy_scatters:
                        selected_up_to_cycle = selections_df.filter(pl.col('cycle') <= cycle_num)
                        for strategy in unique_strategies:
                            strategy_scatter = strategy_scatters[strategy]
                            strategy_selections = selected_up_to_cycle.filter(pl.col('strategy') == strategy)
                            if strategy_selections.height > 0:
                                selected_preds = strategy_selections.get_column('prediction_at_selection').to_numpy()
                                selected_uncs = strategy_selections.get_column('uncertainty_at_selection').to_numpy()
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
            else:
                scatter1.set_offsets(np.empty((0, 2)))
                for scatter in strategy_scatters.values():
                    scatter.set_offsets(np.empty((0, 2)))
        else:
            # No uncertainty: show prediction histogram
            ax1.clear()
            ax1.set_title('Prediction Distribution')
            ax1.set_xlabel('Predicted Activity')
            ax1.set_ylabel('Count')
            pred_clean = predictions[np.isfinite(predictions)]
            if len(pred_clean) > 0:
                ax1.hist(pred_clean, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
                # Mark selected compounds with vertical lines if available
                if selections_df.height > 0:
                    sel_ids_up_to = selections_df.filter(pl.col('cycle') <= cycle_num)
                    if sel_ids_up_to.height > 0:
                        sel_preds = sel_ids_up_to.get_column('prediction_at_selection').to_numpy()
                        sel_preds = sel_preds[np.isfinite(sel_preds)]
                        if len(sel_preds) > 0:
                            for sp in sel_preds:
                                ax1.axvline(x=sp, color='red', alpha=0.15, linewidth=1)
            ax1.set_xlim(ax2_xlim)

        # Panel B: Discovery Metrics - extract data up to current cycle
        # Use cycle_num + 1 (not cycle_idx + 1) because metrics span cycles 0..N
        # while parquets start at cycle 1 (cycle 0 has no predictions).
        n_rows = cycle_num + 1
        cycles = np.arange(n_rows)

        # Helper to safely extract metric column
        def get_metric_values(col_name, default_value=0.0):
            if col_name in metrics_df.columns:
                return metrics_df.head(n_rows).get_column(col_name).to_numpy()
            return np.full(n_rows, default_value)

        top10_values = get_metric_values('top_10_discovery', 0.0)
        top100_values = get_metric_values('top_100_discovery', 0.0)
        top01pct_values = get_metric_values('top_0_1_pct_discovery', 0.0)
        top1pct_values = get_metric_values('top_1_pct_discovery', 0.0)
        batch_ratio_values = get_metric_values('batch_score_improvement_ratio', 1.0)
        cumul_ratio_values = get_metric_values('cumulative_score_improvement_ratio', 1.0)

        # Calculate percentage explored for x-axis coordinates
        n_total = compounds_df.height
        pct_explored = []
        if 'cumulative_labeled' in metrics_df.columns:
            cumulative_values = metrics_df.head(n_rows).get_column('cumulative_labeled').to_numpy()
            for cumulative in cumulative_values:
                pct = (cumulative / n_total) * 100 if n_total > 0 else 0
                pct_explored.append(pct)
        else:
            pct_explored = [0.0] * n_rows

        line_top10_disc.set_data(pct_explored, top10_values)
        line_top100_disc.set_data(pct_explored, top100_values)
        line_top01pct_disc.set_data(pct_explored, top01pct_values)
        line_top1pct_disc.set_data(pct_explored, top1pct_values)
        line_batch_ratio.set_data(pct_explored, batch_ratio_values)
        line_cumul_ratio.set_data(pct_explored, cumul_ratio_values)
        ax2_twin.set_ylim(global_ratio_lo - ratio_pad, global_ratio_hi + ratio_pad)

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

        if selections_df.height > 0:
            selected_up_to_cycle = selections_df.filter(pl.col('cycle') <= cycle_num)
            if selected_up_to_cycle.height > 0:
                cumulative_best = []
                best_so_far = float('inf')

                for c in range(cycle_num + 1):
                    cycle_selections = selections_df.filter(pl.col('cycle') == c)
                    if cycle_selections.height > 0:
                        cycle_values = cycle_selections.get_column('measured_value').to_numpy()
                        cycle_best = np.nanmin(cycle_values)
                        best_so_far = min(best_so_far, cycle_best)
                    cumulative_best.append(best_so_far)

                # Align pct_explored (which covers cycles 0..cycle_num) with
                # cumulative_best — both have length cycle_num+1.
                pct_range = pct_explored[:len(cumulative_best)]
                ax3.plot(pct_range, cumulative_best, 'b-', linewidth=2, marker='o', markersize=4, label='Best Found')

                if target_col and target_col in compounds_df.columns:
                    true_best = compounds_df.get_column(target_col).min()
                    if true_best is not None and not np.isnan(true_best):
                        ax3.axhline(y=true_best, color='g', linestyle='--', linewidth=2, alpha=0.7, label='True Best')

                max_pct = pct_explored[-1] if len(pct_explored) > 0 else 5.0
                ax3.set_xlim(0, max_pct * 1.1)
                if len(cumulative_best) > 1:
                    first_real = cumulative_best[0]
                    if not np.isfinite(first_real):
                        # First cycle(s) may lack selection_history entries;
                        # use the first finite value as the top of the range.
                        finite_vals = [v for v in cumulative_best if np.isfinite(v)]
                        first_real = finite_vals[0] if finite_vals else cumulative_best[-1]
                    y_lo = cumulative_best[-1]
                    y_hi = first_real
                    y_range = y_hi - y_lo
                    if y_range <= 0:
                        y_range = abs(y_lo) * 0.1 if y_lo != 0 else 1.0
                    ax3.set_ylim(y_lo - y_range * 0.2, y_hi + y_range * 0.2)
                ax3.legend(loc='upper right')
                ax3.grid(True, alpha=0.3)

                # Update x-axis labels for Panel C
                ax3.set_xticks(pct_range)
                x_labels_c = [f'{pct:.1f}%\n(c{c})' for c, pct in enumerate(pct_range)]
                ax3.set_xticklabels(x_labels_c, fontsize=8, rotation=0)

        # Panel D: Model Ranking Metrics
        spearman_values = get_metric_values('unlabeled_spearman_correlation', 0.0)
        top1000_overlap_values = get_metric_values('unlabeled_top_1000_overlap', 0.0)
        top100_overlap_values = get_metric_values('unlabeled_top_100_overlap', 0.0)

        line_spearman.set_data(pct_explored, spearman_values)
        line_top1000_overlap.set_data(pct_explored, top1000_overlap_values)
        line_top100_overlap.set_data(pct_explored, top100_overlap_values)

        ax4.set_xlim(0, max_pct * 1.1)
        ax4.set_xlabel('Data Explored (%)', fontsize=10)
        ax4.set_xticks(pct_explored)
        ax4.set_xticklabels(x_labels, fontsize=8, rotation=0)

        fig.suptitle(f'Active Learning Progress - Cycle {cycle_num}', fontsize=14, fontweight='bold')

        artists = [scatter1] if scatter1 is not None else []
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


def create_static_dashboard(
    output_dir: str,
    output_file: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Create a single-frame static dashboard showing the final state of all cycles.

    Unlike the animated dashboard, this renders all data at once in a single
    static 2x2 figure saved as a PNG. Useful for reports and quick inspection.

    Args:
        output_dir: Directory containing run output files.
        output_file: Path to save the PNG. If None, the figure is returned.
        dpi: Resolution for the saved image.

    Returns:
        matplotlib Figure.
    """
    data = load_csv_data(output_dir)

    compounds_df = data['compounds']
    metrics_df = data['metrics']
    selections_df = data['selections']
    parquets: list[tuple[int, Path]] = data['parquets']

    system_cols = {'ID', 'SMILES', 'status', 'labeled_cycle', 'selected_cycle', 'pruned_cycle'}
    target_col = None
    for col in compounds_df.columns:
        if col not in system_cols:
            target_col = col
            break

    n_cycles = len(parquets)
    if n_cycles == 0:
        raise ValueError('No prediction cycles found in data')

    # Collect all-cycle data
    all_predictions = []
    all_uncertainties = []
    has_unc = False
    for _, pq_path in parquets:
        schema = pl.scan_parquet(pq_path).collect_schema().names()
        cols = ['prediction']
        if 'uncertainty' in schema:
            cols.append('uncertainty')
            has_unc = True
        df_chunk = pl.scan_parquet(pq_path).select(cols).collect()
        all_predictions.append(df_chunk.get_column('prediction').to_numpy())
        if 'uncertainty' in schema:
            all_uncertainties.append(df_chunk.get_column('uncertainty').to_numpy())

    stacked_preds = np.concatenate(all_predictions)
    stacked_unc = np.concatenate(all_uncertainties) if all_uncertainties else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ((ax1, ax2), (ax3, ax4)) = axes

    # Panel A
    if has_unc:
        ax1.set_title('Uncertainty vs Prediction')
        ax1.set_xlabel('Predicted Activity')
        ax1.set_ylabel('Uncertainty')
        valid = np.isfinite(stacked_preds) & np.isfinite(stacked_unc)
        ax1.scatter(stacked_preds[valid], stacked_unc[valid], c=stacked_preds[valid],
                    cmap='viridis', alpha=0.6, s=10)
    else:
        ax1.set_title('Prediction Distribution')
        ax1.set_xlabel('Predicted Activity')
        ax1.set_ylabel('Count')
        clean = stacked_preds[np.isfinite(stacked_preds)]
        if len(clean) > 0:
            ax1.hist(clean, bins=50, color='steelblue', alpha=0.7, edgecolor='white')

    # Panel B: Discovery metrics (final state = all cycles)
    n_rows = len(metrics_df)
    pct_explored = []
    n_total = compounds_df.height
    if 'cumulative_labeled' in metrics_df.columns:
        for cumulative in metrics_df.get_column('cumulative_labeled').to_list():
            pct_explored.append((cumulative / n_total) * 100 if n_total > 0 else 0)
    else:
        pct_explored = list(range(n_rows))

    def _get(name, default=0.0):
        if name in metrics_df.columns:
            return metrics_df.get_column(name).to_numpy()
        return np.full(n_rows, default)

    ax2.plot(pct_explored, _get('top_10_discovery'), 'r-o', linewidth=2, markersize=4, label='Top-10')
    ax2.plot(pct_explored, _get('top_100_discovery'), 'b-s', linewidth=2, markersize=4, label='Top-100')
    ax2.plot(pct_explored, _get('top_0_1_pct_discovery'), 'g-^', linewidth=2, markersize=4, label='Top-0.1%')
    ax2.plot(pct_explored, _get('top_1_pct_discovery'), 'm-d', linewidth=2, markersize=4, label='Top-1%')

    ax2_twin = ax2.twinx()
    ax2_twin.plot(pct_explored, _get('batch_score_improvement_ratio', 1.0), 'c--', linewidth=2, markersize=4, label='Batch Ratio')
    ax2_twin.plot(pct_explored, _get('cumulative_score_improvement_ratio', 1.0), 'orange', linestyle='--', linewidth=2, markersize=4, label='Cumul Ratio')
    ax2_twin.set_ylabel('Avg Score Ratio', color='darkcyan', fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor='darkcyan')

    ax2.set_title('Discovery Metrics')
    ax2.set_xlabel('Data Explored (%)')
    ax2.set_ylabel('Discovery Rate (%)')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left', fontsize=8)
    ax2_twin.legend(loc='upper right', fontsize=8)

    # Panel C: Cumulative best
    ax3.set_title('Cumulative Best Value Found')
    ax3.set_xlabel('Data Explored (%)')
    ax3.set_ylabel('Best Value (lower is better)')
    if selections_df.height > 0:
        best_so_far = float('inf')
        cumulative_best = []
        cycle_pcts = []
        for c in range(n_cycles + 1):
            cyc_sel = selections_df.filter(pl.col('cycle') == c)
            if cyc_sel.height > 0:
                cyc_best = np.nanmin(cyc_sel.get_column('measured_value').to_numpy())
                best_so_far = min(best_so_far, cyc_best)
                if c < len(pct_explored):
                    cumulative_best.append(best_so_far)
                    cycle_pcts.append(pct_explored[c])
        if cumulative_best:
            ax3.plot(cycle_pcts, cumulative_best, 'b-', linewidth=2, marker='o', markersize=4, label='Best Found')
            if target_col and target_col in compounds_df.columns:
                true_best = compounds_df.get_column(target_col).min()
                if true_best is not None and not np.isnan(true_best):
                    ax3.axhline(y=true_best, color='g', linestyle='--', linewidth=2, alpha=0.7, label='True Best')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)

    # Panel D: Model ranking
    ax4.set_title('Model Ranking Metrics')
    ax4.set_xlabel('Data Explored (%)')
    ax4.set_ylabel('Spearman ρ', color='blue', fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4.set_ylim(-1, 1)
    ax4.plot(pct_explored, _get('unlabeled_spearman_correlation'), 'b-o', linewidth=2, markersize=4, label='Spearman ρ')

    ax4_twin1 = ax4.twinx()
    ax4_twin1.set_ylabel('Top-1000 Overlap (%)', color='green', fontweight='bold')
    ax4_twin1.tick_params(axis='y', labelcolor='green')
    ax4_twin1.set_ylim(0, 100)
    ax4_twin1.plot(pct_explored, _get('unlabeled_top_1000_overlap'), 'g-s', linewidth=2, markersize=4, label='Top-1000 Overlap')

    ax4_twin2 = ax4.twinx()
    ax4_twin2.spines['right'].set_position(('outward', 60))
    ax4_twin2.set_ylabel('Top-100 Overlap (%)', color='red', fontweight='bold')
    ax4_twin2.tick_params(axis='y', labelcolor='red')
    ax4_twin2.set_ylim(0, 100)
    ax4_twin2.plot(pct_explored, _get('unlabeled_top_100_overlap'), 'r-^', linewidth=2, markersize=4, label='Top-100 Overlap')

    lines4 = [ax4.get_lines()[0], ax4_twin1.get_lines()[0], ax4_twin2.get_lines()[0]]
    labels4 = [line.get_label() for line in lines4]
    ax4.legend(lines4, labels4, loc='center left', fontsize=8)

    n_labeled = pct_explored[-1] if pct_explored else 5.0
    fig.suptitle(f'Active Learning Dashboard — {n_labeled:.1f}% explored ({n_cycles} cycles)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        logger.info(f'Saved static dashboard: {output_file}')

    return fig


def generate_all_plots(
    output_dir: str,
    output_prefix: str | None = None,
    fps: int = 2,
    dpi: int = 120,
) -> dict[str, str]:
    """Generate all visualization outputs for a run: MP4, GIF, HTML, PNG.

    Convenience wrapper that calls both the animated dashboard and static
    dashboard generators. Returns a dict mapping format/type to output path.

    Args:
        output_dir: Directory containing run output files.
        output_prefix: Prefix for output files. If None, uses ``output_dir`` itself.
        fps: Frames per second for animation.
        dpi: Resolution for saved outputs.

    Returns:
        Dict with keys like 'mp4', 'gif', 'html', 'png' mapping to file paths.
    """
    base = Path(output_prefix) if output_prefix else Path(output_dir) / 'dashboard'

    results: dict[str, str] = {}

    # Animated
    create_dashboard_animation_from_csv(
        output_dir, output_file=str(base.with_suffix('.mp4')), format='mp4', fps=fps, dpi=dpi
    )
    results['mp4'] = str(base.with_suffix('.mp4'))

    create_dashboard_animation_from_csv(
        output_dir, output_file=str(base.with_suffix('.gif')), format='gif', fps=fps, dpi=dpi
    )
    results['gif'] = str(base.with_suffix('.gif'))

    create_dashboard_animation_from_csv(
        output_dir, output_file=str(base.with_suffix('.html')), format='html', fps=fps, dpi=dpi
    )
    results['html'] = str(base.with_suffix('.html'))

    # Static
    create_static_dashboard(
        output_dir, output_file=str(base.with_suffix('.png')), dpi=dpi
    )
    results['png'] = str(base.with_suffix('.png'))

    return results


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
                logger.warning("FFmpeg not found. Saved as GIF: %s", gif_path)
                logger.info("To save as MP4 in the future, install FFmpeg: conda install ffmpeg")

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
    except (ValueError, RuntimeError, OSError) as e:
        if 'FFmpeg' not in str(e):
            logger.error(f"Failed to save animation: {e}")
            raise

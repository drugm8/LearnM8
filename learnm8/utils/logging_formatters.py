"""
Logging formatters for LearnM8 active learning experiments.

Provides clean, reusable formatting functions for console output including:
- Cycle schedule presentation
- Duration formatting
- Rich metrics tables
- Experiment summaries

All functions return formatted strings suitable for logging at INFO level.
"""

import logging
from typing import Any
import polars as pl

logger = logging.getLogger(__name__)


def format_cycle_schedule(cycle_num: int, config: 'CycleConfig', pool_size: int) -> str:
    """
    Format cycle configuration for INFO logging.

    Creates a human-readable description of a cycle configuration including
    strategy, batch size, and pruning parameters.

    Args:
        cycle_num: Starting cycle number (1-indexed)
        config: CycleConfig object with strategy and parameters
        pool_size: Original pool size for batch calculation

    Returns:
        Formatted string describing the cycle configuration

    Examples:
        >>> config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.01)
        >>> format_cycle_schedule(1, config, 10000)
        'Cycle 1: GREEDY acquisition (100 compounds, 1.0% batch size)'

        >>> config = CycleConfig('ucb', n_cycles=5, batch_fraction=0.005,
        ...                      pruning_strategy='score_based',
        ...                      pruning_params={'pruning_fraction': 0.3})
        >>> format_cycle_schedule(1, config, 10000)
        'Cycles 1-5: UCB acquisition (50 compounds per cycle, 0.5% batch size), 30% pruned per cycle'
    """
    batch_size = int(pool_size * config.batch_fraction)

    if config.n_cycles == 1:
        msg = f"Cycle {cycle_num}: {config.strategy.upper()} acquisition "
        msg += f"({batch_size} compounds, {config.batch_fraction*100:.1f}% batch size)"
    else:
        end_cycle = cycle_num + config.n_cycles - 1
        msg = f"Cycles {cycle_num}-{end_cycle}: {config.strategy.upper()} acquisition "
        msg += f"({batch_size} compounds per cycle, {config.batch_fraction*100:.1f}% batch size)"

    if config.pruning_strategy:
        prune_frac = config.pruning_params.get('pruning_fraction', 0.3) if config.pruning_params else 0.3
        msg += f", {prune_frac*100:.0f}% pruned per cycle"

    return msg


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Converts seconds to appropriate time units (seconds, minutes, or hours)
    for display in logs and summaries.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string

    Examples:
        >>> format_duration(45.7)
        '45.7 seconds'

        >>> format_duration(154)
        '2 minutes 34 seconds'

        >>> format_duration(7384)
        '2 hours 3 minutes'
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes} minutes {secs} seconds"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours} hours {minutes} minutes"


def format_cycle_metrics_table(
	metrics: dict[str, Any],
	oracle_type: str = 'auto',
	previous_metrics: dict[str, Any] | None = None,
	score_direction: str = 'higher'
) -> str:
	"""
	Format cycle metrics as compact key-value lines with change indicators.

	Creates clean multi-line output automatically adapting for benchmark vs run modes.
	Uses colored arrows: [green]↑[/green] (improvement), [red]↓[/red] (worsening),
	[yellow]→[/yellow] (stagnant) with Rich markup.

	Args:
		metrics: Metrics dictionary from evaluate_cycle
		oracle_type: Oracle type for mode-specific formatting ('benchmark' or 'run')
		previous_metrics: Previous cycle metrics for change indicators
		score_direction: 'higher' or 'lower' - indicates optimization direction

	Returns:
		Formatted string for console output with Rich markup

	Note:
		This function provides mode-aware display:
		- Benchmark mode: Shows Discovery (2 rows), Ranking, and Selection metrics
		- Run mode: Shows only Selection metrics
	"""
	lines = []

	cycle = metrics.get('cycle', '?')
	batch_size = metrics.get('batch_size', '?')
	cumulative_labeled = metrics.get('cumulative_labeled', '?')
	is_benchmark = (oracle_type == 'benchmark')

	# Metrics where lower is better (error metrics, similarity to previous)
	bad_metrics = {'rmse', 'mae', 'mse', 'inter_cycle_similarity'}
	# Add avg_score_selected to bad_metrics if score_direction is 'lower'
	if score_direction == 'lower':
		bad_metrics = bad_metrics | {'avg_score_selected'}

	def get_change(key: str, current_val: float, is_pct: bool = False) -> str:
		"""Return Rich-markup colored change indicator with stagnation support."""
		if previous_metrics is None or key not in previous_metrics:
			return ""

		prev_val = previous_metrics.get(key)
		if prev_val is None or current_val is None:
			return ""

		try:
			diff = float(current_val) - float(prev_val)
		except (ValueError, TypeError):
			return ""

		# Determine stagnation threshold
		if is_pct or 'discovery' in key or 'overlap' in key:
			stagnation_threshold = 1.0
		else:
			stagnation_threshold = 0.01

		# Check for stagnation
		if abs(diff) < stagnation_threshold:
			return "[yellow]→[/yellow]"

		is_higher_better = key not in bad_metrics
		is_improvement = (diff > 0) if is_higher_better else (diff < 0)

		return "[green]↑[/green]" if is_improvement else "[red]↓[/red]"

	def format_metric(key: str, label: str, digits: int = 1, is_pct: bool = False) -> str:
		"""Format a single metric with optional change indicator."""
		val = metrics.get(key)
		if val is None:
			return f"{label}:N/A"

		change = get_change(key, val, is_pct)
		if is_pct:
			return f"{label}:{val:.{digits}f}%{change}"
		else:
			return f"{label}:{val:.{digits}f}{change}"

	# Header
	lines.append(f"Cycle {cycle} | {batch_size} selected | {cumulative_labeled} labeled")
	lines.append("─" * 60)

	if is_benchmark:
		# Discovery metrics - split into two rows
		discovery_row1 = [
			format_metric('top_10_discovery', 'Top10', 1, True),
			format_metric('top_100_discovery', 'Top100', 1, True),
			format_metric('top_1000_discovery', 'Top1K', 1, True),
		]
		discovery_row2 = [
			format_metric('top_0_1_pct_discovery', 'Top0.1%', 1, True),
			format_metric('top_1_pct_discovery', 'Top1%', 1, True),
		]
		lines.append(f"Discovery │ {' '.join(discovery_row1)}")
		lines.append(f"          │ {' '.join(discovery_row2)}")

		# Ranking line
		ranking_parts = [
			format_metric('unlabeled_spearman_correlation', 'Spearman', 3, False),
			format_metric('unlabeled_ef_1_0', 'EF@1%', 2, False),
			format_metric('unlabeled_ef_5_0', 'EF@5%', 2, False),
		]
		lines.append(f"Ranking   │ {' '.join(ranking_parts)}")

	# Selection line
	selection_parts = [format_metric('avg_score_selected', 'Avg', 2, False)]

	if metrics.get('uncertainty_mean') is not None:
		selection_parts.append(format_metric('uncertainty_mean', 'Unc', 3, False))
	if metrics.get('intra_batch_diversity') is not None:
		selection_parts.append(format_metric('intra_batch_diversity', 'Div', 2, False))
	if metrics.get('batch_novelty_score') is not None:
		selection_parts.append(format_metric('batch_novelty_score', 'Nov', 2, False))

	lines.append(f"Selection │ {' '.join(selection_parts)}")

	return "\n".join(lines)


def format_experiment_summary(
    compounds_df: pl.DataFrame,
    duration: float,
    total_cycles: int
) -> list[str]:
    """
    Generate experiment completion summary lines.

    Creates formatted summary statistics for the completed experiment including
    compound counts, percentages, and duration.

    Args:
        compounds_df: Master DataFrame with all compounds and final states
        duration: Total experiment duration in seconds
        total_cycles: Total number of cycles executed

    Returns:
        List of formatted strings for logging at INFO level

    Example:
        >>> df = pl.DataFrame({
        ...     'ID': range(1000),
        ...     'status': ['labeled'] * 300 + ['pruned'] * 200 + ['unlabeled'] * 500
        ... })
        >>> lines = format_experiment_summary(df, 3723.5, 10)
        >>> for line in lines:
        ...     print(line)
        Total compounds labeled: 300 of 1000 (30.0%)
        Total compounds pruned: 200 (20.0%)
        Final unlabeled pool: 500 compounds (50.0%)
        Duration: 1 hour 2 minutes
    """
    total_labeled = compounds_df.filter(pl.col('status') == 'labeled').height
    total_pruned = compounds_df.filter(pl.col('status') == 'pruned').height
    total_unlabeled = compounds_df.filter(pl.col('status') == 'unlabeled').height
    total_compounds = len(compounds_df)

    lines = []

    lines.append(
        f"Total compounds labeled: {total_labeled} of {total_compounds} "
        f"({total_labeled/total_compounds*100:.1f}%)"
    )

    if total_pruned > 0:
        lines.append(
            f"Total compounds pruned: {total_pruned} "
            f"({total_pruned/total_compounds*100:.1f}%)"
        )

    lines.append(
        f"Final unlabeled pool: {total_unlabeled} compounds "
        f"({total_unlabeled/total_compounds*100:.1f}%)"
    )

    duration_str = format_duration(duration)
    lines.append(f"Duration: {duration_str}")

    return lines

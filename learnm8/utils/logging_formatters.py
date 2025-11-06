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
from typing import Dict, Any, Optional, List
import pandas as pd

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
	metrics: Dict[str, Any],
	oracle_type: str = 'auto',
	previous_metrics: Optional[Dict[str, Any]] = None
) -> str:
	"""
	Format cycle metrics as simple lines with optional change indicators.

	Creates clean line-by-line output automatically adapting for benchmark vs run modes.

	Args:
		metrics: Metrics dictionary from evaluate_cycle
		oracle_type: Oracle type for mode-specific formatting ('benchmark' or 'run')
		previous_metrics: Previous cycle metrics for change indicators

	Returns:
		Formatted string for console output

	Note:
		This function provides mode-aware display:
		- Benchmark mode: Shows Selection, Discovery, and Ranking metrics
		- Run mode: Shows only Selection Quality metrics
	"""
	lines = []
	
	cycle = metrics.get('cycle', '?')
	batch_size = metrics.get('batch_size', '?')
	is_benchmark = (oracle_type == 'benchmark')
	
	def get_change(key: str, is_higher_better: bool = True) -> str:
		"""Return change indicator if metrics improved/worsened."""
		if previous_metrics is None or key not in metrics or key not in previous_metrics:
			return ""
		
		current = metrics[key]
		previous = previous_metrics[key]
		
		if current is None or previous is None:
			return ""
		
		diff = current - previous
		if abs(diff) < 0.001:
			return ""
		
		is_improvement = (diff > 0) if is_higher_better else (diff < 0)
		symbol = "↑" if diff > 0 else "↓"
		return f" {symbol}" if is_improvement else f" {symbol}"
	
	lines.append(f"\nCycle {cycle} ({batch_size} selected)")
	lines.append("=" * 50)
	
	lines.append("\nSelection Quality:")
	if metrics.get('avg_score_selected') is not None:
		val = metrics['avg_score_selected']
		change = get_change('avg_score_selected', True)
		lines.append(f"  Batch Avg: {val:.3f}{change}")
	
	if metrics.get('avg_score_ground_truth') is not None:
		lines.append(f"  GT Avg: {metrics['avg_score_ground_truth']:.3f}")
	
	if metrics.get('cumulative_labeled') is not None:
		lines.append(f"  Total Labeled: {metrics['cumulative_labeled']}")
	
	if metrics.get('diversity_score') is not None:
		lines.append(f"  Diversity: {metrics['diversity_score']:.3f}")
	
	if is_benchmark:
		lines.append("\nDiscovery Metrics:")
		
		if metrics.get('top_10_discovery') is not None:
			val = metrics['top_10_discovery']
			change = get_change('top_10_discovery', True)
			lines.append(f"  Top-10: {val:.1f}%{change}")
		
		if metrics.get('top_100_discovery') is not None:
			val = metrics['top_100_discovery']
			change = get_change('top_100_discovery', True)
			lines.append(f"  Top-100: {val:.1f}%{change}")
		
		if metrics.get('top_1k_discovery') is not None:
			val = metrics['top_1k_discovery']
			change = get_change('top_1k_discovery', True)
			lines.append(f"  Top-1K: {val:.1f}%{change}")
		
		if metrics.get('enrichment_factor_10') is not None:
			val = metrics['enrichment_factor_10']
			change = get_change('enrichment_factor_10', True)
			lines.append(f"  EF@10: {val:.2f}{change}")
		
		if metrics.get('score_ratio') is not None:
			val = metrics['score_ratio']
			change = get_change('score_ratio', True)
			lines.append(f"  Score Ratio: {val:.2f}{change}")
		
		lines.append("\nRanking (Unlabeled):")
		
		if metrics.get('unlabeled_top_10_discovery') is not None:
			lines.append(f"  Unlbl Top-10: {metrics['unlabeled_top_10_discovery']:.1f}%")
		
		if metrics.get('unlabeled_enrichment_factor_10') is not None:
			val = metrics['unlabeled_enrichment_factor_10']
			change = get_change('unlabeled_enrichment_factor_10', True)
			lines.append(f"  Unlbl EF@10: {val:.2f}{change}")
		
		if metrics.get('spearman_correlation') is not None:
			val = metrics['spearman_correlation']
			change = get_change('spearman_correlation', True)
			lines.append(f"  Spearman: {val:.3f}{change}")
	
	return "\n".join(lines)


def format_experiment_summary(
    compounds_df: pd.DataFrame,
    duration: float,
    total_cycles: int
) -> List[str]:
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
        >>> df = pd.DataFrame({
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
    total_labeled = len(compounds_df[compounds_df['status'] == 'labeled'])
    total_pruned = len(compounds_df[compounds_df['status'] == 'pruned'])
    total_unlabeled = len(compounds_df[compounds_df['status'] == 'unlabeled'])
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

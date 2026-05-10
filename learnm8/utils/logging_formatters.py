"""
Logging formatters for LearnM8 active learning experiments.

Provides clean, reusable formatting functions for console output including:
- Cycle schedule presentation
- Duration formatting
- Experiment summaries

All functions return formatted strings suitable for logging at INFO level.
"""

import logging

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

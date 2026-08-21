"""Configuration dataclasses and parsing for flexible cycle scheduling.

Supports two APIs:
1. Simple API: Individual parameters (strategy, n_cycles, batch_fraction)
2. Advanced API: List of CycleConfig objects with full control

Both APIs produce the same output: List[CycleConfig] with n_cycles=1 (expanded).

Examples:
    # Simple API
    schedule = parse_cycle_schedule(strategy='greedy', n_cycles=10)

    # Advanced API
    schedule = parse_cycle_schedule(cycles=[
        CycleConfig('random', n_cycles=3, batch_fraction=0.02),
        CycleConfig('greedy', n_cycles=5, batch_fraction=0.01)
    ])

    # String specification (for CLI)
    configs = parse_cycle_spec('random:0.02 greedy:0.01*5')
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CycleConfig:
    """Configuration for a single cycle or group of cycles.

    Attributes:
        strategy: Acquisition strategy name (e.g., 'greedy', 'ucb', 'random', 'ei')
        n_cycles: Number of cycles with this configuration (default 1)
        batch_fraction: Fraction of original pool to select per cycle
        pruning_strategy: Pruning strategy name (e.g., 'score')
        pruning_params: Parameters for pruning strategy (e.g., {'pruning_fraction': 0.3})
        acquisition_params: Parameters for acquisition strategy (e.g., {'beta': 2.0})

    Example:
        >>> config = CycleConfig('greedy', n_cycles=5, batch_fraction=0.01)
        >>> config_with_pruning = CycleConfig(
        ...     'greedy', n_cycles=3, batch_fraction=0.01,
        ...     pruning_strategy='score',
        ...     pruning_params={'pruning_fraction': 0.3}
        ... )
    """
    strategy: str
    n_cycles: int = 1
    batch_fraction: float | None = None
    pruning_strategy: str | None = None
    pruning_params: dict | None = None
    acquisition_params: dict | None = None

    def __post_init__(self):
        """Validate that batch_fraction is provided."""
        if self.batch_fraction is None:
            raise ValueError(
                "CycleConfig requires batch_fraction. "
                "Specify the fraction of the pool to select per cycle (e.g., batch_fraction=0.01 for 1%)."
            )
        if not (0 < float(self.batch_fraction) <= 1.0):
            raise ValueError(
                f"batch_fraction must be between 0 (exclusive) and 1 (inclusive), "
                f"got {self.batch_fraction}. Use a value like 0.01 (1%) or 0.1 (10%)."
            )


def parse_cycle_spec(spec: str) -> list[dict[str, Any]]:
    """Parse compact string specification into list of dicts.

    Format specification:
    - Each part separated by whitespace
    - Format: "strategy:fraction" or "strategy:fraction*count"
    - strategy: acquisition strategy name
    - fraction: batch_fraction (float between 0 and 1)
    - count: number of cycles (optional, default 1)

    Args:
        spec: String specification like "random:0.02 greedy:0.01*5 ucb:0.01*3"

    Returns:
        List of dicts with keys: strategy, batch_fraction, n_cycles

    Examples:
        >>> parse_cycle_spec("random:0.02")
        [{'strategy': 'random', 'batch_fraction': 0.02, 'n_cycles': 1}]

        >>> parse_cycle_spec("greedy:0.01*5")
        [{'strategy': 'greedy', 'batch_fraction': 0.01, 'n_cycles': 5}]

        >>> parse_cycle_spec("random:0.02 greedy:0.01*5 ucb:0.01*3")
        [
            {'strategy': 'random', 'batch_fraction': 0.02, 'n_cycles': 1},
            {'strategy': 'greedy', 'batch_fraction': 0.01, 'n_cycles': 5},
            {'strategy': 'ucb', 'batch_fraction': 0.01, 'n_cycles': 3}
        ]
    """
    parts = spec.split()
    if not parts:
        return []

    configs = []
    for part in parts:
        try:
            if ':' not in part:
                raise ValueError(
                    f"Invalid cycle specification '{part}'. "
                    f"Expected format: 'strategy:fraction' or 'strategy:fraction*count'. "
                    f"Example: 'greedy:0.01' or 'greedy:0.01*5'"
                )

            strategy, rest = part.split(':', 1)

            if '*' in rest:
                fraction_str, count_str = rest.split('*', 1)
                batch_fraction = float(fraction_str)
                n_cycles = int(count_str)
            else:
                batch_fraction = float(rest)
                n_cycles = 1

            if not (0 < batch_fraction <= 1.0):
                raise ValueError(
                    f"batch_fraction must be between 0 (exclusive) and 1 (inclusive), "
                    f"got {batch_fraction} in specification '{part}'. "
                    f"Use a value like 0.01 (1%) or 0.1 (10%)."
                )
            if n_cycles < 1:
                raise ValueError(
                    f"n_cycles must be >= 1, got {n_cycles} in specification '{part}'."
                )

            configs.append({
                'strategy': strategy,
                'batch_fraction': batch_fraction,
                'n_cycles': n_cycles
            })

        except ValueError as e:
            raise ValueError(
                f"Invalid cycle specification '{part}'. "
                f"Expected format: 'strategy:fraction' or 'strategy:fraction*count'. "
                f"Example: 'greedy:0.01' or 'greedy:0.01*5'. "
                f"Error: {e}"
            ) from e

    return configs


def parse_cycle_schedule(
    cycles: list[CycleConfig] | None = None,
    strategy: str = 'greedy',
    n_cycles: int = 10,
    batch_fraction: float = 0.01,
    initial_strategy: str = 'random',
    acquisition_params: dict | None = None,
    pruning_strategy: str | None = None,
    pruning_params: dict | None = None,
) -> list[CycleConfig]:
    """Convert either advanced API (cycles list) or simple API (individual parameters) to unified List[CycleConfig].

    This is the main entry point for cycle configuration. It supports two APIs:

    1. Advanced API: Provide cycles list with full control over each cycle group
    2. Simple API: Provide individual parameters for standard patterns

    Both APIs produce the same output: List[CycleConfig] with n_cycles=1 (expanded).

    Args:
        cycles: Advanced API - List of CycleConfig objects
        strategy: Simple API - Acquisition strategy for cycles 1+
        n_cycles: Simple API - Total number of cycles
        batch_fraction: Simple API - Batch fraction for ALL cycles (including cycle 0)
        initial_strategy: Simple API - Strategy for cycle 0 (default: 'random')
        acquisition_params: Parameters for acquisition strategy
        pruning_strategy: Pruning strategy name
        pruning_params: Parameters for pruning strategy

    Returns:
        List of CycleConfig objects, each with n_cycles=1 (expanded from multi-cycle configs)

    Examples:
        >>> # Simple API - same batch size, different strategies
        >>> schedule = parse_cycle_schedule(
        ...     strategy='greedy',
        ...     n_cycles=10,
        ...     batch_fraction=0.01,
        ...     initial_strategy='random'
        ... )
        >>> # Returns: [CycleConfig('random', 1, batch_fraction=0.01),
        >>> #           CycleConfig('greedy', 1, batch_fraction=0.01), ...] (10 total)

        >>> # Advanced API
        >>> schedule = parse_cycle_schedule(cycles=[
        ...     CycleConfig('random', n_cycles=3, batch_fraction=0.02),
        ...     CycleConfig('greedy', n_cycles=5, batch_fraction=0.01,
        ...                 pruning_strategy='score',
        ...                 pruning_params={'pruning_fraction': 0.3})
        ... ])
        >>> # Returns: 8 CycleConfig objects, each with n_cycles=1
    """
    logger.debug(f"Parsing cycle schedule: cycles={cycles is not None}, strategy='{strategy}', n_cycles={n_cycles}")

    if cycles is not None:
        if not isinstance(cycles, list):
            raise ValueError(
                f"cycles must be a list of CycleConfig objects, got {type(cycles).__name__}. "
                f"Example: cycles=[CycleConfig('greedy', n_cycles=5, batch_fraction=0.01)]"
            )

        expanded = []
        for config in cycles:
            if not isinstance(config, CycleConfig):
                raise ValueError(
                    f"Each element in cycles must be a CycleConfig instance, "
                    f"got {type(config).__name__}. "
                    f"Example: CycleConfig('greedy', n_cycles=5, batch_fraction=0.01)"
                )

            # Pruning is a run-level knob (the CLI exposes it only as a top-level
            # flag), so fall back to the top-level values whenever a block does
            # not set its own. An explicit per-block strategy always wins.
            if config.pruning_strategy is not None:
                block_pruning_strategy = config.pruning_strategy
                block_pruning_params = config.pruning_params
            else:
                block_pruning_strategy = pruning_strategy
                block_pruning_params = pruning_params

            # Same story for acquisition parameters: the CLI exposes them only
            # as a top-level flag (--acquisition-params), so a block without
            # its own value inherits them. An explicit per-block dict wins.
            if config.acquisition_params is not None:
                block_acquisition_params = config.acquisition_params
            else:
                block_acquisition_params = acquisition_params

            for _ in range(config.n_cycles):
                new_config = CycleConfig(
                    strategy=config.strategy,
                    n_cycles=1,
                    batch_fraction=config.batch_fraction,
                    pruning_strategy=block_pruning_strategy,
                    pruning_params=block_pruning_params,
                    acquisition_params=block_acquisition_params
                )
                expanded.append(new_config)
                logger.debug(f"Created CycleConfig(strategy='{new_config.strategy}', n_cycles={new_config.n_cycles}, "
                            f"batch_fraction={new_config.batch_fraction}, pruning={new_config.pruning_strategy or 'disabled'})")

        total_cycles = len(expanded)
        logger.debug(f"Schedule parsed: {len(cycles)} config blocks, {total_cycles} total cycles")
        return expanded

    schedule = []

    # Cycle 0: initial_strategy with same batch_fraction
    config_0 = CycleConfig(
        strategy=initial_strategy,
        n_cycles=1,
        batch_fraction=batch_fraction,
        acquisition_params=acquisition_params,
        pruning_strategy=pruning_strategy,
        pruning_params=pruning_params,
    )
    schedule.append(config_0)
    logger.debug(f"Created CycleConfig(strategy='{config_0.strategy}', n_cycles={config_0.n_cycles}, "
                f"batch_fraction={config_0.batch_fraction}, pruning={config_0.pruning_strategy or 'disabled'})")

    # Cycles 1+: main strategy with same batch_fraction
    for _ in range(n_cycles - 1):
        config_i = CycleConfig(
            strategy=strategy,
            n_cycles=1,
            batch_fraction=batch_fraction,
            acquisition_params=acquisition_params,
            pruning_strategy=pruning_strategy,
            pruning_params=pruning_params,
        )
        schedule.append(config_i)
        logger.debug(f"Created CycleConfig(strategy='{config_i.strategy}', n_cycles={config_i.n_cycles}, "
                    f"batch_fraction={config_i.batch_fraction}, pruning={config_i.pruning_strategy or 'disabled'})")

    total_cycles = len(schedule)
    logger.debug(f"Schedule parsed: simple API, {total_cycles} total cycles")
    return schedule

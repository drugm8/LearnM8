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

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class CycleConfig:
    """Configuration for a single cycle or group of cycles.

    Attributes:
        strategy: Acquisition strategy name (e.g., 'greedy', 'ucb', 'random', 'ei', 'bitbirch')
        n_cycles: Number of cycles with this configuration (default 1)
        batch_fraction: Fraction of original pool to select per cycle
        pruning_strategy: Pruning strategy name (e.g., 'score')
        pruning_params: Parameters for pruning strategy (e.g., {'pruning_fraction': 0.3})
        acquisition_params: Parameters for acquisition strategy (e.g., {'exploration_weight': 2.0})

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
    batch_fraction: Optional[float] = None
    pruning_strategy: Optional[str] = None
    pruning_params: Optional[Dict] = None
    acquisition_params: Optional[Dict] = None

    def __post_init__(self):
        """Validate that batch_fraction is provided."""
        if self.batch_fraction is None:
            raise ValueError("Must provide batch_fraction")
        if not (0 < float(self.batch_fraction) <= 1.0):
            raise ValueError(f"batch_fraction must be between 0 and 1, got {self.batch_fraction}")


def parse_cycle_spec(spec: str) -> List[Dict[str, Any]]:
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
                raise ValueError(f"batch_fraction must be between 0 and 1, got {batch_fraction}")
            if n_cycles < 1:
                raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")

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
            )

    return configs


def parse_cycle_schedule(
    cycles: Optional[List[CycleConfig]] = None,
    strategy: str = 'greedy',
    n_cycles: int = 10,
    batch_fraction: float = 0.01,
    initial_strategy: str = 'random',
    initial_batch_fraction: float = 0.01,
    **kwargs
) -> List[CycleConfig]:
    """Convert either advanced API (cycles list) or simple API (individual parameters) to unified List[CycleConfig].

    This is the main entry point for cycle configuration. It supports two APIs:

    1. Advanced API: Provide cycles list with full control over each cycle group
    2. Simple API: Provide individual parameters for standard patterns

    Both APIs produce the same output: List[CycleConfig] with n_cycles=1 (expanded).

    Args:
        cycles: Advanced API - List of CycleConfig objects
        strategy: Simple API - Acquisition strategy for main cycles
        n_cycles: Simple API - Total number of cycles
        batch_fraction: Simple API - Batch fraction for main cycles
        initial_strategy: Simple API - Strategy for first cycle (typically 'random')
        initial_batch_fraction: Simple API - Batch fraction for first cycle
        **kwargs: Additional parameters passed to CycleConfig (pruning_strategy, acquisition_params, etc.)

    Returns:
        List of CycleConfig objects, each with n_cycles=1 (expanded from multi-cycle configs)

    Examples:
        >>> # Simple API
        >>> schedule = parse_cycle_schedule(
        ...     strategy='greedy',
        ...     n_cycles=10,
        ...     batch_fraction=0.01,
        ...     initial_strategy='random',
        ...     initial_batch_fraction=0.02
        ... )
        >>> # Returns: [CycleConfig('random', 1, batch_fraction=0.02),
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
    VALID_CYCLE_PARAMS = {'pruning_strategy', 'pruning_params', 'acquisition_params'}

    cycle_kwargs = {}
    invalid_kwargs = {}

    for key, value in kwargs.items():
        if key in VALID_CYCLE_PARAMS:
            cycle_kwargs[key] = value
        else:
            invalid_kwargs[key] = value

    if invalid_kwargs:
        import warnings
        warnings.warn(
            f"Invalid cycle parameters ignored: {', '.join(invalid_kwargs.keys())}. "
            f"Valid cycle parameters are: {', '.join(sorted(VALID_CYCLE_PARAMS))}. "
            f"These parameters have no effect and will be removed in a future version.",
            UserWarning,
            stacklevel=2
        )
        logger.warning(f"Invalid cycle parameters ignored: {', '.join(invalid_kwargs.keys())}")

    if cycles is not None:
        if not isinstance(cycles, list):
            raise ValueError("cycles must be a list of CycleConfig objects")

        expanded = []
        for config in cycles:
            if not isinstance(config, CycleConfig):
                raise ValueError("Each element in cycles must be a CycleConfig instance")

            for _ in range(config.n_cycles):
                expanded.append(CycleConfig(
                    strategy=config.strategy,
                    n_cycles=1,
                    batch_fraction=config.batch_fraction,
                    pruning_strategy=config.pruning_strategy,
                    pruning_params=config.pruning_params,
                    acquisition_params=config.acquisition_params
                ))

        return expanded

    schedule = []

    schedule.append(CycleConfig(
        strategy=initial_strategy,
        n_cycles=1,
        batch_fraction=initial_batch_fraction,
        **cycle_kwargs
    ))

    for _ in range(n_cycles - 1):
        schedule.append(CycleConfig(
            strategy=strategy,
            n_cycles=1,
            batch_fraction=batch_fraction,
            **cycle_kwargs
        ))

    return schedule

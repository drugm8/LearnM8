"""Utilities for handling active learning cycle specifications."""

from typing import List, Tuple


def parse_cycle_spec(cycle_spec: str) -> List[Tuple[str, float]]:
    """Parse cycle specification string into cycles list.
    
    Args:
        cycle_spec: String like "random:0.01 greedy:0.005*5 diverse:0.01"
        
    Returns:
        List of (strategy, batch_fraction) tuples
        
    Examples:
        >>> parse_cycle_spec("random:0.01 greedy:0.005*5")
        [('random', 0.01), ('greedy', 0.005), ('greedy', 0.005), ('greedy', 0.005), ('greedy', 0.005), ('greedy', 0.005)]
    """
    cycles_list = []
    for part in cycle_spec.split():
        if '*' in part:
            strategy_batch, count = part.split('*')
            strategy, fraction = strategy_batch.split(':')
            for _ in range(int(count)):
                cycles_list.append((strategy, float(fraction)))
        else:
            strategy, fraction = part.split(':')
            cycles_list.append((strategy, float(fraction)))
    return cycles_list


def summarize_cycle_spec(cycle_spec: str) -> str:
    """Create concise summary of cycle specification for naming.
    
    Args:
        cycle_spec: String like "random:0.01 greedy:0.005*5 diverse:0.01"
        
    Returns:
        Abbreviated summary like "r1_g5_d1"
    """
    strategy_abbrev = {
        'random': 'r', 'greedy': 'g',
        'ucb': 'u', 'ei': 'e', 'pi': 'p', 'thompson': 't',
        'entropy': 'h', 'bitbirch': 'b', 'simulated_annealing': 's'
    }
    
    parts = []
    for part in cycle_spec.split():
        if '*' in part:
            strategy_batch, count = part.split('*')
            strategy = strategy_batch.split(':')[0]
            abbrev = strategy_abbrev.get(strategy, strategy[0])
            parts.append(f"{abbrev}{count}")
        else:
            strategy = part.split(':')[0]
            abbrev = strategy_abbrev.get(strategy, strategy[0])
            parts.append(f"{abbrev}1")
    
    return '_'.join(parts)


def validate_cycle_spec(cycle_spec: str) -> tuple[bool, str]:
    """Validate cycle specification format.
    
    Args:
        cycle_spec: Cycle specification string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        cycles = parse_cycle_spec(cycle_spec)
        if not cycles:
            return False, "Empty cycle specification"
        
        for strategy, fraction in cycles:
            if not strategy:
                return False, "Empty strategy name"
            if not 0 < fraction <= 1:
                return False, f"Invalid batch fraction {fraction}, must be between 0 and 1"
        
        return True, ""
    except Exception as e:
        return False, f"Invalid cycle specification: {str(e)}"
"""Random selection strategy for active learning."""

import pandas as pd


def select_random(compounds: pd.DataFrame, n_select: int, random_state: int = None) -> pd.DataFrame:
    """
    Randomly select compounds.
    
    Args:
        compounds: DataFrame with compound information
        n_select: Number of compounds to select
        random_state: Random seed for reproducibility
        
    Returns:
        Randomly selected compounds DataFrame
    """
    selected = compounds.sample(n=min(n_select, len(compounds)), random_state=random_state)
    return selected
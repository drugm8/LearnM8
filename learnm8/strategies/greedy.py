"""Greedy selection strategy for active learning."""

import pandas as pd


def select_greedy(compounds: pd.DataFrame, n_select: int, score_direction: str = 'higher') -> pd.DataFrame:
    """
    Select compounds greedily based on predicted scores.
    
    Args:
        compounds: DataFrame with 'prediction' column
        n_select: Number of compounds to select
        score_direction: 'higher' for maximization, 'lower' for minimization
        
    Returns:
        Selected compounds DataFrame
    """
    if 'prediction' not in compounds.columns:
        raise ValueError("Compounds must have 'prediction' column")
    
    # Sort by prediction score
    ascending = (score_direction == 'lower')
    sorted_compounds = compounds.sort_values('prediction', ascending=ascending)
    
    # Select top compounds
    selected = sorted_compounds.head(n_select)
    
    return selected
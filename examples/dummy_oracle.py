"""
Example oracle function for LearnM8 run mode.

Oracle functions should:
1. Take a list of compound IDs as input
2. Return a pandas DataFrame with 'ID' column and target property columns
3. Optionally include 'Activity' column (binary 0/1) for enrichment calculations

This dummy oracle simulates molecular property prediction with random values.
"""

import pandas as pd
import numpy as np
from typing import List


def oracle_function(compound_ids: List[str]) -> pd.DataFrame:
    """
    Example oracle function that returns simulated binding affinity data.
    
    Args:
        compound_ids: List of compound IDs to evaluate
        
    Returns:
        DataFrame with columns: ['ID', 'binding_affinity', 'Activity']
    """
    np.random.seed(42)  # For reproducible results
    
    results = []
    for compound_id in compound_ids:
        # Simulate binding affinity (lower is better, range: -12 to -6)
        binding_affinity = np.random.normal(-9.0, 2.0)
        
        # Simulate binary activity (1 for active, 0 for inactive)
        # Make compounds with better binding affinity more likely to be active
        activity_prob = 1 / (1 + np.exp(binding_affinity + 8))  # Sigmoid
        activity = 1 if np.random.random() < activity_prob else 0
        
        results.append({
            'ID': compound_id,
            'binding_affinity': binding_affinity,
            'Activity': activity
        })
    
    return pd.DataFrame(results)


# Alternative oracle function with different property
def simple_oracle(compound_ids: List[str]) -> pd.DataFrame:
    """
    Simplified oracle that only returns the target property.
    
    Args:
        compound_ids: List of compound IDs to evaluate
        
    Returns:
        DataFrame with columns: ['ID', 'score']
    """
    np.random.seed(123)
    
    results = []
    for compound_id in compound_ids:
        # Simulate a simple score (higher is better, range: 0-10)
        score = np.random.uniform(0, 10)
        
        results.append({
            'ID': compound_id,
            'score': score
        })
    
    return pd.DataFrame(results)
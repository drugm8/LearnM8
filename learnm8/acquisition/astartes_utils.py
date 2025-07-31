import numpy as np
import pandas as pd
from typing import List, Optional
import logging

from learnm8.core.data_manager import DataManager

logger = logging.getLogger(__name__)


def compute_tanimoto_distance_matrix(fingerprints: np.ndarray) -> np.ndarray:
    """Compute pairwise Tanimoto distance matrix from binary fingerprints.
    
    Args:
        fingerprints: Binary fingerprint matrix of shape (n_compounds, n_bits)
        
    Returns:
        Square distance matrix of shape (n_compounds, n_compounds)
        
    Note:
        Tanimoto distance = 1 - Tanimoto similarity
        Tanimoto similarity = intersection / union
    """
    if len(fingerprints.shape) != 2:
        raise ValueError(f"Expected 2D fingerprint matrix, got shape {fingerprints.shape}")
    
    n_compounds = fingerprints.shape[0]
    
    if n_compounds > 5000:
        logger.warning(f"Computing distance matrix for {n_compounds} compounds "
                      f"requires {n_compounds**2 * 8 / 1e9:.1f}GB memory")
    
    # Compute intersection matrix: dot product for binary fingerprints
    intersection = np.dot(fingerprints, fingerprints.T)
    
    # Compute union matrix: |A| + |B| - |A ∩ B|
    fingerprint_counts = np.sum(fingerprints, axis=1)
    union = (fingerprint_counts[:, np.newaxis] + 
             fingerprint_counts[np.newaxis, :] - 
             intersection)
    
    # Avoid division by zero
    union = np.maximum(union, 1e-8)
    
    # Tanimoto similarity = intersection / union
    similarity = intersection / union
    
    # Tanimoto distance = 1 - similarity
    distance = 1.0 - similarity
    
    return distance


def get_molecular_features(compounds: pd.DataFrame, 
                          data_manager: DataManager,
                          featurizer_type: str = 'morgan') -> np.ndarray:
    """Extract molecular features using DataManager.
    
    Args:
        compounds: DataFrame with 'ID' and 'SMILES' columns
        data_manager: DataManager instance for feature extraction
        featurizer_type: Type of molecular features ('morgan', 'maccs', 'ecfp6')
        
    Returns:
        Feature matrix of shape (n_compounds, n_features)
    """
    if 'ID' not in compounds.columns:
        raise ValueError("Compounds DataFrame must have 'ID' column")
    if 'SMILES' not in compounds.columns:
        raise ValueError("Compounds DataFrame must have 'SMILES' column")
    
    compound_ids = compounds['ID'].tolist()
    smiles_list = compounds['SMILES'].tolist()
    
    features = data_manager.get_features(
        compound_ids=compound_ids,
        smiles_list=smiles_list,
        featurizer_type=featurizer_type
    )
    
    return features


def validate_acquisition_input(compounds: pd.DataFrame, n_select: int) -> None:
    """Validate inputs for acquisition functions.
    
    Args:
        compounds: Compounds DataFrame
        n_select: Number of compounds to select
        
    Raises:
        ValueError: If inputs are invalid
    """
    if compounds.empty:
        raise ValueError("Compounds DataFrame is empty")
    
    if n_select <= 0:
        raise ValueError(f"n_select must be positive, got {n_select}")
    
    if n_select > len(compounds):
        logger.warning(f"n_select ({n_select}) exceeds available compounds ({len(compounds)}), "
                      f"will select all {len(compounds)} available compounds")
    
    required_columns = ['ID', 'SMILES']
    missing_columns = [col for col in required_columns if col not in compounds.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def fast_kennard_stone(distance_matrix: np.ndarray) -> np.ndarray:
    """Fast implementation of Kennard-Stone algorithm.
    
    This is extracted from the astartes library with their optimized numpy operations.
    
    Args:
        distance_matrix: Square distance matrix of shape (n_samples, n_samples)
        
    Returns:
        Array of indices in order of Kennard-Stone selection
    """
    n_samples = len(distance_matrix)
    
    if n_samples < 2:
        return np.arange(n_samples)
    
    # When searching for max distance, disregard self-distances
    distance_copy = distance_matrix.copy()
    np.fill_diagonal(distance_copy, -np.inf)
    
    # Get the row/col of maximum distance (most distant pair)
    max_idx = np.nanargmax(distance_copy)
    max_coords = np.unravel_index(max_idx, distance_copy.shape)
    
    # Track indices in order of selection
    selected = np.empty(n_samples, dtype=int)
    selected[0] = max_coords[0]
    selected[1] = max_coords[1]
    
    # Initialize minimum distances to the two selected samples
    min_distances = np.min(distance_matrix[:, max_coords], axis=1)
    
    # Iteratively select point with maximum minimum distance
    for i in range(2, n_samples):
        selected[i] = np.argmax(min_distances)
        # Update minimum distances using new selected point
        min_distances = np.minimum(min_distances, distance_matrix[:, selected[i]])
    
    return selected
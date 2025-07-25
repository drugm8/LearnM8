"""Diversity-based selection strategy using clustering."""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina


def _calculate_tanimoto_distances(fingerprints):
    """Calculate pairwise Tanimoto distances between fingerprints."""
    distances = []
    n_fps = len(fingerprints)
    
    for i in range(1, n_fps):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        distances.extend([1 - sim for sim in similarities])
    
    return distances


def select_diverse(compounds: pd.DataFrame, n_select: int, 
                  max_compounds: int = 35000, random_state: int = None) -> pd.DataFrame:
    """
    Select diverse compounds using Butina clustering.
    
    Args:
        compounds: DataFrame with 'SMILES' column
        n_select: Number of compounds to select
        max_compounds: Maximum compounds to consider for clustering (memory constraint)
        random_state: Random seed for reproducibility
        
    Returns:
        Selected diverse compounds DataFrame
    """
    working_compounds = compounds.copy()
    
    # Sample if too many compounds (memory constraint)
    if len(working_compounds) > max_compounds:
        working_compounds = working_compounds.sample(n=max_compounds, random_state=random_state)
    
    # Generate fingerprints
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    fingerprints = []
    
    for smiles in working_compounds['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = rdkit_gen.GetFingerprint(mol)
        fingerprints.append(fp)
    
    # Calculate distances and cluster
    distances = _calculate_tanimoto_distances(fingerprints)
    clusters = Butina.ClusterData(distances, len(fingerprints), 0.4, isDistData=True)
    
    # Sort clusters by size (largest first)
    sorted_clusters = sorted(clusters, key=len, reverse=True)
    
    # Select compounds: cluster centers first, then members
    selected_indices = []
    
    # Add cluster centers
    for cluster in sorted_clusters:
        if len(selected_indices) < n_select:
            selected_indices.append(cluster[0])
    
    # Add additional compounds from larger clusters
    cluster_idx = 0
    while len(selected_indices) < n_select and cluster_idx < len(sorted_clusters):
        cluster = sorted_clusters[cluster_idx]
        
        # Determine how many to take from this cluster
        if len(cluster) > 10:
            n_from_cluster = min(10, n_select - len(selected_indices))
        else:
            n_from_cluster = min(len(cluster) // 2 + 1, n_select - len(selected_indices))
        
        # Add compounds (skip first as it's already added as center)
        for i in range(1, min(n_from_cluster + 1, len(cluster))):
            if len(selected_indices) < n_select:
                selected_indices.append(cluster[i])
        
        cluster_idx += 1
    
    # Return selected compounds
    selected = working_compounds.iloc[selected_indices]
    return selected
"""Sphere exclusion acquisition for distance-based molecular clustering.

This module implements the sphere exclusion algorithm for selecting diverse compounds
by creating non-overlapping spherical clusters in molecular feature space. The 
implementation is extracted from the astartes library for integration with LearnM8.
"""

import logging
from typing import Optional, TYPE_CHECKING
import pandas as pd
import numpy as np

from learnm8.acquisition.base import AcquisitionFunction
from learnm8.acquisition.astartes_utils import (
    compute_tanimoto_distance_matrix,
    get_molecular_features,
    validate_acquisition_input
)

if TYPE_CHECKING:
    from learnm8.core.data_manager import DataManager

logger = logging.getLogger(__name__)

# O(n²) complexity protection constant
MAX_COMPOUNDS = 10000


def sphere_exclusion_clustering(distance_matrix: np.ndarray, 
                               distance_cutoff: float = 0.25,
                               random_state: Optional[int] = None) -> np.ndarray:
    """Apply sphere exclusion clustering to distance matrix.
    
    Creates non-overlapping spherical clusters by iteratively selecting cluster
    centers and excluding nearby points within the distance cutoff.
    
    Implementation follows the astartes library approach:
    1. Normalize distance matrix to [0,1] range
    2. Randomly shuffle sample processing order
    3. For each unassigned sample, create cluster with all unassigned points within cutoff
    
    Args:
        distance_matrix: Square distance matrix
        distance_cutoff: Maximum distance for points to be in same cluster
        random_state: Random seed for sample order shuffling
        
    Returns:
        Array of cluster labels for each sample
    """
    n_samples = distance_matrix.shape[0]
    
    # Normalize distance matrix to [0, 1] range (following astartes approach)
    dist_min = distance_matrix.min()
    dist_max = distance_matrix.max()
    if dist_max > dist_min:
        normalized_distances = (distance_matrix - dist_min) / (dist_max - dist_min)
    else:
        # All distances are the same - set to zero
        normalized_distances = np.zeros_like(distance_matrix)
    
    # Create random order for processing samples (following astartes)
    if random_state is not None:
        np.random.seed(random_state)
    sample_order = np.random.permutation(n_samples)
    
    # Initialize tracking structures
    already_assigned = set()
    cluster_labels = np.full(n_samples, -1, dtype=int)
    cluster_idx = 0
    
    # Process samples in random order
    for sample_idx in sample_order:
        if sample_idx in already_assigned:
            # Skip already assigned samples
            continue
        
        # Get distances from this sample to all others
        distances_from_sample = normalized_distances[sample_idx, :]
        
        # Find all samples within distance cutoff (using <= for inclusive boundary)
        candidate_indices = set(np.flatnonzero(distances_from_sample <= distance_cutoff))
        
        # Only consider unassigned candidates
        unassigned_candidates = candidate_indices.difference(already_assigned)
        
        # Assign all unassigned candidates to this cluster
        for idx in unassigned_candidates:
            cluster_labels[idx] = cluster_idx
        
        # Update tracking
        already_assigned.update(unassigned_candidates)
        cluster_idx += 1
    
    return cluster_labels


class SphereExclusionAcquisition(AcquisitionFunction):
    """Sphere exclusion acquisition for tunable molecular diversity.
    
    The sphere exclusion algorithm creates diversity through distance-based clustering:
    1. Computes pairwise distances between molecular fingerprints
    2. Normalizes distances to [0,1] range
    3. Creates non-overlapping spherical clusters with configurable radius
    4. Selects representatives from different clusters for diversity
    
    The distance_cutoff parameter controls diversity level:
    - Smaller values (0.1) create more clusters = higher diversity
    - Larger values (0.5) create fewer clusters = lower diversity
    
    Args:
        data_manager: DataManager instance for feature extraction and caching
        distance_cutoff: Maximum distance for clustering (0.0-1.0, default: 0.25)
        featurizer_type: Type of molecular features ('morgan', 'maccs', 'ecfp6')
        random_state: Random seed for reproducible clustering
    """
    
    def __init__(self,
                 data_manager: Optional['DataManager'] = None,
                 distance_cutoff: float = 0.25,
                 featurizer_type: str = 'morgan',
                 random_state: Optional[int] = 42,
                 **kwargs):
        """Initialize Sphere Exclusion acquisition function.
        
        Args:
            data_manager: DataManager instance for feature extraction and caching
            distance_cutoff: Maximum distance for clustering (0.0-1.0, default: 0.25)
            featurizer_type: Type of molecular features ('morgan', 'maccs', 'ecfp6')
            random_state: Random seed for reproducible clustering
            **kwargs: Additional parameters for compatibility
        """
        super().__init__(data_manager=data_manager, **kwargs)
        
        if data_manager is None:
            raise ValueError("SphereExclusionAcquisition requires a DataManager for feature extraction")
        
        if not 0.0 <= distance_cutoff <= 1.0:
            raise ValueError(f"distance_cutoff must be between 0.0 and 1.0, got {distance_cutoff}")
        
        self.distance_cutoff = distance_cutoff
        self.featurizer_type = featurizer_type
        self.random_state = random_state
        
        # Set numpy random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select diverse compounds using sphere exclusion clustering.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction' columns
            n_select: Number of compounds to select
            
        Returns:
            DataFrame with selected compounds from different clusters
            
        Raises:
            ValueError: If inputs are invalid or feature extraction fails
            RuntimeError: If clustering fails
        """
        # Validate inputs
        self.validate_input(compounds, n_select)
        validate_acquisition_input(compounds, n_select)
        
        # Check dataset size for O(n²) complexity protection
        if len(compounds) > MAX_COMPOUNDS:
            raise ValueError(f"Too many compounds ({len(compounds)}) for {self.__class__.__name__}. "
                           f"Maximum allowed: {MAX_COMPOUNDS}. "
                           f"Consider using a different acquisition method or reducing dataset size.")
        
        logger.info(f"Starting sphere exclusion selection of {n_select} compounds "
                   f"from {len(compounds)} candidates using {self.featurizer_type} features "
                   f"with distance cutoff {self.distance_cutoff}")
        
        # Handle edge cases
        if n_select >= len(compounds):
            logger.info("Selecting all available compounds")
            return compounds.copy()
        
        try:
            # Extract molecular features
            features = get_molecular_features(
                compounds=compounds,
                data_manager=self.data_manager,
                featurizer_type=self.featurizer_type
            )
            
            logger.info(f"Extracted {features.shape[1]} molecular features "
                       f"for {features.shape[0]} compounds")
            
            # Compute distance matrix
            logger.info("Computing pairwise distance matrix...")
            distance_matrix = compute_tanimoto_distance_matrix(features)
            
            # Apply sphere exclusion clustering
            logger.info(f"Running sphere exclusion clustering with cutoff {self.distance_cutoff}...")
            cluster_labels = sphere_exclusion_clustering(
                distance_matrix=distance_matrix,
                distance_cutoff=self.distance_cutoff,
                random_state=self.random_state
            )
            
            # Get cluster information
            unique_clusters = np.unique(cluster_labels)
            n_clusters = len(unique_clusters)
            
            logger.info(f"Created {n_clusters} clusters from {len(compounds)} compounds")
            
            # Select representatives from clusters
            selected_indices = self._select_cluster_representatives(
                compounds=compounds,
                cluster_labels=cluster_labels,
                n_select=n_select
            )
            
            # Build result DataFrame
            selected_compounds = compounds.iloc[selected_indices].copy()
            
            # Add acquisition metadata
            selected_compounds['acquisition_score'] = np.arange(len(selected_indices), 0, -1)
            selected_compounds['cluster_id'] = cluster_labels[selected_indices]
            
            logger.info(f"Successfully selected {len(selected_compounds)} compounds "
                       f"from {n_clusters} clusters")
            
            return selected_compounds
            
        except Exception as e:
            logger.error(f"Sphere exclusion selection failed: {str(e)}")
            raise RuntimeError(f"Sphere exclusion acquisition failed: {str(e)}") from e
    
    def _select_cluster_representatives(self,
                                       compounds: pd.DataFrame,
                                       cluster_labels: np.ndarray,
                                       n_select: int) -> np.ndarray:
        """Select representative compounds from clusters.
        
        Strategy: 
        1. Select one compound per cluster (round-robin)
        2. For additional selections, prefer highest prediction scores within clusters
        
        Args:
            compounds: Original compounds DataFrame
            cluster_labels: Cluster assignment for each compound
            n_select: Number of compounds to select
            
        Returns:
            Array of selected compound indices
        """
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        selected_indices = []
        
        # First round: select one from each cluster
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(selected_indices) >= n_select:
                break
            
            # Select highest prediction score in cluster
            cluster_predictions = compounds.iloc[cluster_indices]['prediction'].values
            best_in_cluster = cluster_indices[np.argmax(cluster_predictions)]
            selected_indices.append(best_in_cluster)
        
        # Additional rounds: continue round-robin with remaining compounds
        while len(selected_indices) < n_select:
            added_this_round = False
            
            for cluster_id in unique_clusters:
                if len(selected_indices) >= n_select:
                    break
                
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                # Find unselected compounds in this cluster
                unselected_in_cluster = [idx for idx in cluster_indices 
                                       if idx not in selected_indices]
                
                if unselected_in_cluster:
                    # Select best remaining compound in cluster
                    remaining_predictions = compounds.iloc[unselected_in_cluster]['prediction'].values
                    best_remaining = unselected_in_cluster[np.argmax(remaining_predictions)]
                    selected_indices.append(best_remaining)
                    added_this_round = True
            
            # If no compounds were added, we've exhausted all clusters
            if not added_this_round:
                break
        
        return np.array(selected_indices[:n_select])
    
    def requires_uncertainty(self) -> bool:
        """Sphere exclusion doesn't require uncertainty estimates."""
        return False
    
    def get_name(self) -> str:
        """Return descriptive name for this acquisition function."""
        return f"SphereExclusion({self.featurizer_type}, cutoff={self.distance_cutoff})"


def create_sphere_exclusion_acquisition(data_manager: 'DataManager',
                                        distance_cutoff: float = 0.25,
                                        featurizer_type: str = 'morgan',
                                        random_state: Optional[int] = 42) -> SphereExclusionAcquisition:
    """Factory function for creating SphereExclusionAcquisition instances.
    
    Args:
        data_manager: DataManager instance for feature extraction and caching
        distance_cutoff: Maximum distance for clustering (0.0-1.0)
        featurizer_type: Type of molecular features ('morgan', 'maccs', 'ecfp6')
        random_state: Random seed for reproducible clustering
        
    Returns:
        Configured SphereExclusionAcquisition instance
    """
    return SphereExclusionAcquisition(
        data_manager=data_manager,
        distance_cutoff=distance_cutoff,
        featurizer_type=featurizer_type,
        random_state=random_state
    )
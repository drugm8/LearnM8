"""Simplified base class for clustering-based acquisition functions.

This module provides a single, straightforward base class for acquisition
strategies that use dimensionality reduction followed by clustering.
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List

from .base import AcquisitionFunction
from ..core.data_manager import DataManager

logger = logging.getLogger(__name__)


class ClusteringAcquisition(AcquisitionFunction, ABC):
    """Simple base class for clustering-based acquisition.
    
    This class provides common functionality for acquisition strategies that:
    1. Extract molecular features
    2. Apply dimensionality reduction (UMAP, t-SNE)
    3. Perform clustering in the reduced space
    4. Select compounds evenly across clusters
    """
    
    def __init__(self, 
                 data_manager: DataManager,
                 featurizer_type: str = 'morgan',
                 n_components: int = 2,
                 random_state: int = 42,
                 **reduction_params):
        """Initialize clustering acquisition.
        
        Args:
            data_manager: DataManager instance for feature extraction and caching
            featurizer_type: Type of molecular features ('morgan', 'ecfp6', 'maccs', 'descriptors')
            n_components: Number of embedding dimensions (typically 2)
            random_state: Random seed for reproducibility
            **reduction_params: Additional parameters for dimensionality reduction
        """
        self.data_manager = data_manager
        self.featurizer_type = featurizer_type
        self.n_components = n_components
        self.random_state = random_state
        self.reduction_params = reduction_params
        np.random.seed(random_state)
    
    @property
    @abstractmethod
    def reduction_method(self) -> str:
        """Return the name of the dimensionality reduction method."""
        pass
    
    @abstractmethod
    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform clustering on embeddings.
        
        Args:
            embeddings: 2D embedding matrix
            
        Returns:
            Array of cluster labels
        """
        pass
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select compounds using dimensionality reduction and clustering.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES' columns and predictions
            n_select: Number of compounds to select
            
        Returns:
            DataFrame subset with selected compounds
        """
        # Validate input
        self.validate_input(compounds, n_select)
        
        if n_select >= len(compounds):
            return compounds.copy()
        
        logger.info(f"Selecting {n_select} compounds using {self.reduction_method} + clustering")
        
        # Get embeddings from DataManager (with caching)
        embeddings = self.data_manager.get_embeddings(
            compound_ids=compounds['ID'].tolist(),
            smiles_list=compounds['SMILES'].tolist(),
            featurizer_type=self.featurizer_type,
            reduction_method=self.reduction_method,
            n_components=self.n_components,
            **self.reduction_params
        )
        
        # Perform clustering
        cluster_labels = self._perform_clustering(embeddings)
        
        # Select compounds evenly across clusters
        selected_indices = self._select_evenly_from_clusters(
            compounds, cluster_labels, n_select
        )
        
        return compounds.iloc[selected_indices].copy()
    
    def _select_evenly_from_clusters(self, 
                                    compounds: pd.DataFrame,
                                    cluster_labels: np.ndarray,
                                    n_select: int) -> List[int]:
        """Select compounds evenly across clusters using simple random selection.
        
        Args:
            compounds: Input compounds DataFrame
            cluster_labels: Cluster assignment for each compound
            n_select: Number of compounds to select
            
        Returns:
            List of selected compound indices
        """
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        # Handle case where we have no clusters
        if n_clusters == 0:
            logger.warning("No clusters found, falling back to random selection")
            return np.random.choice(len(compounds), n_select, replace=False).tolist()
        
        # Calculate even distribution across clusters
        base_per_cluster = n_select // n_clusters
        remainder = n_select % n_clusters
        
        selected_indices = []
        
        # Sort clusters by size for consistent behavior
        cluster_sizes = [(cluster, np.sum(cluster_labels == cluster)) 
                        for cluster in unique_clusters]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)  # Largest first
        
        for i, (cluster, cluster_size) in enumerate(cluster_sizes):
            # Determine how many to select from this cluster
            n_from_cluster = base_per_cluster
            if i < remainder:  # Distribute remainder to first clusters
                n_from_cluster += 1
            
            # Don't select more than available in cluster
            n_from_cluster = min(n_from_cluster, cluster_size)
            
            if n_from_cluster > 0:
                cluster_mask = cluster_labels == cluster
                cluster_indices = np.where(cluster_mask)[0]
                
                # Simple random selection within cluster
                if n_from_cluster >= len(cluster_indices):
                    # Take all compounds from this cluster
                    selected_indices.extend(cluster_indices.tolist())
                else:
                    # Randomly sample from cluster
                    cluster_selection = np.random.choice(
                        cluster_indices, n_from_cluster, replace=False
                    )
                    selected_indices.extend(cluster_selection.tolist())
        
        # Handle case where we still need more compounds
        if len(selected_indices) < n_select:
            remaining_indices = set(range(len(compounds))) - set(selected_indices)
            remaining_needed = n_select - len(selected_indices)
            
            if remaining_indices and remaining_needed > 0:
                additional = np.random.choice(
                    list(remaining_indices), 
                    min(remaining_needed, len(remaining_indices)), 
                    replace=False
                )
                selected_indices.extend(additional.tolist())
        
        # Trim to exact number if we somehow selected too many
        if len(selected_indices) > n_select:
            selected_indices = selected_indices[:n_select]
        
        logger.info(f"Selected {len(selected_indices)} compounds from {n_clusters} clusters")
        
        return selected_indices
    
    def get_name(self) -> str:
        """Return descriptive name for this acquisition function."""
        return f"{self.reduction_method.upper()}+Clustering({self.featurizer_type})"
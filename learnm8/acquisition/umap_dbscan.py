"""UMAP-based acquisition functions for molecular diversity.

This module implements simplified UMAP dimensionality reduction followed by
clustering for molecular diversity selection.
"""

import logging
import numpy as np
from typing import Optional

from .clustering import ClusteringAcquisition
from ..core.data_manager import DataManager

logger = logging.getLogger(__name__)


class UMAPDBSCANAcquisition(ClusteringAcquisition):
    """UMAP+DBSCAN acquisition for molecular diversity.
    
    Uses Uniform Manifold Approximation and Projection (UMAP) for non-linear
    dimensionality reduction followed by DBSCAN clustering.
    """
    
    def __init__(self,
                 data_manager: DataManager,
                 featurizer_type: str = 'morgan',
                 n_components: int = 2,
                 n_neighbors: int = 15,
                 min_dist: float = 0.1,
                 metric: str = 'euclidean',
                 eps: float = 0.5,
                 min_samples: int = 5,
                 random_state: int = 42,
                 **umap_params):
        """Initialize UMAP+DBSCAN acquisition function.
        
        Args:
            data_manager: DataManager instance for feature extraction and caching
            featurizer_type: Type of molecular features ('morgan', 'ecfp6', 'maccs', 'descriptors')
            n_components: Number of UMAP components (typically 2)
            n_neighbors: UMAP n_neighbors parameter (controls local vs global structure)
            min_dist: UMAP min_dist parameter (controls tightness of embedding)
            metric: Distance metric for UMAP ('euclidean', 'cosine', 'manhattan', etc.)
            eps: DBSCAN eps parameter (neighborhood radius)
            min_samples: DBSCAN min_samples parameter
            random_state: Random seed for reproducibility
            **umap_params: Additional parameters for UMAP
        """
        # Set UMAP-specific defaults
        umap_defaults = {
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric,
        }
        umap_defaults.update(umap_params)
        
        super().__init__(
            data_manager=data_manager,
            featurizer_type=featurizer_type,
            n_components=n_components,
            random_state=random_state,
            **umap_defaults
        )
        
        self.eps = eps
        self.min_samples = min_samples
        
        logger.info(f"Initialized UMAP+DBSCAN acquisition: "
                   f"featurizer={featurizer_type}, components={n_components}, "
                   f"n_neighbors={n_neighbors}, min_dist={min_dist}, "
                   f"eps={eps}, min_samples={min_samples}")
    
    @property
    def reduction_method(self) -> str:
        """Return the name of the dimensionality reduction method."""
        return 'umap'
    
    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering on UMAP embeddings.
        
        Args:
            embeddings: 2D UMAP embedding matrix
            
        Returns:
            Array of cluster labels (-1 for noise points)
        """
        from sklearn.cluster import DBSCAN
        
        if len(embeddings) < 2:
            return np.zeros(len(embeddings))
        
        logger.info(f"DBSCAN parameters: eps={self.eps}, min_samples={self.min_samples}")
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.cluster_labels = dbscan.fit_predict(embeddings)
        
        unique_clusters = np.unique(self.cluster_labels[self.cluster_labels != -1])
        n_noise = np.sum(self.cluster_labels == -1)
        
        logger.info(f"DBSCAN clustering: {len(unique_clusters)} clusters, "
                   f"{n_noise} noise points from {len(embeddings)} compounds")
        
        # Handle noise points by assigning them to individual clusters
        if n_noise > 0:
            noise_mask = self.cluster_labels == -1
            noise_indices = np.where(noise_mask)[0]
            
            # Assign each noise point to its own cluster
            max_cluster = self.cluster_labels.max() if len(unique_clusters) > 0 else -1
            for i, noise_idx in enumerate(noise_indices):
                self.cluster_labels[noise_idx] = max_cluster + 1 + i

            logger.info(f"Assigned {n_noise} noise points to individual clusters")
        
        return self.cluster_labels
    
    def get_name(self) -> str:
        """Return descriptive name for this acquisition function."""
        return f"UMAP+DBSCAN({self.featurizer_type})"


class UMAPKMeansAcquisition(ClusteringAcquisition):
    """UMAP+K-Means acquisition for molecular diversity.
    
    Uses UMAP followed by K-Means clustering. K-Means provides more stable
    clustering results than DBSCAN and doesn't produce noise points.
    """
    
    def __init__(self,
                 data_manager: DataManager,
                 featurizer_type: str = 'morgan',
                 n_components: int = 2,
                 n_neighbors: int = 15,
                 min_dist: float = 0.1,
                 metric: str = 'euclidean',
                 n_clusters: Optional[int] = None,
                 random_state: int = 42,
                 **umap_params):
        """Initialize UMAP+K-Means acquisition function.
        
        Args:
            data_manager: DataManager instance for feature extraction and caching
            featurizer_type: Type of molecular features ('morgan', 'ecfp6', 'maccs', 'descriptors')
            n_components: Number of UMAP components (typically 2)
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            metric: Distance metric for UMAP
            n_clusters: Number of K-Means clusters. If None, estimated from data.
            random_state: Random seed for reproducibility
            **umap_params: Additional parameters for UMAP
        """
        # Set UMAP-specific defaults
        umap_defaults = {
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric,
        }
        umap_defaults.update(umap_params)
        
        super().__init__(
            data_manager=data_manager,
            featurizer_type=featurizer_type,
            n_components=n_components,
            random_state=random_state,
            **umap_defaults
        )
        
        self.n_clusters = n_clusters
        
        logger.info(f"Initialized UMAP+K-Means acquisition: "
                   f"featurizer={featurizer_type}, components={n_components}, "
                   f"n_neighbors={n_neighbors}, min_dist={min_dist}, "
                   f"n_clusters={n_clusters}")
    
    @property
    def reduction_method(self) -> str:
        """Return the name of the dimensionality reduction method."""
        return 'umap'
    
    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform K-Means clustering on UMAP embeddings.
        
        Args:
            embeddings: 2D UMAP embedding matrix
            
        Returns:
            Array of cluster labels
        """
        from sklearn.cluster import KMeans
        
        if len(embeddings) < 2:
            return np.zeros(len(embeddings))
        
        # Determine number of clusters
        if self.n_clusters is None:
            # Simple heuristic: sqrt(n/2) but at least 3 and at most n/3
            n_clusters = min(max(3, int(np.sqrt(len(embeddings) / 2))), len(embeddings) // 3)
        else:
            n_clusters = min(self.n_clusters, len(embeddings))
        
        logger.info(f"K-Means parameters: n_clusters={n_clusters}")
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        self.cluster_labels = kmeans.fit_predict(embeddings)

        unique_clusters = np.unique(self.cluster_labels)
        logger.info(f"K-Means clustering: {len(unique_clusters)} clusters from {len(embeddings)} compounds")

        return self.cluster_labels
    
    def get_name(self) -> str:
        """Return descriptive name for this acquisition function."""
        return f"UMAP+K-Means({self.featurizer_type})"
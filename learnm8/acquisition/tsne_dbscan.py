"""t-SNE-based acquisition functions for molecular diversity.

This module implements simplified t-SNE dimensionality reduction followed by
clustering for molecular diversity selection.
"""

import logging
import numpy as np
import warnings
from typing import Optional

from .clustering import ClusteringAcquisition
from ..core.data_manager import DataManager

logger = logging.getLogger(__name__)


class TSNEDBSCANAcquisition(ClusteringAcquisition):
    """t-SNE+DBSCAN acquisition for molecular diversity.
    
    Uses t-Distributed Stochastic Neighbor Embedding (t-SNE) for non-linear
    dimensionality reduction followed by DBSCAN clustering. Best suited for
    smaller datasets (< 10,000 compounds).
    """
    
    def __init__(self,
                 data_manager: DataManager,
                 featurizer_type: str = 'morgan',
                 n_components: int = 2,
                 perplexity: Optional[float] = None,
                 early_exaggeration: float = 12.0,
                 learning_rate: str = 'auto',
                 max_iter: int = 1000,
                 eps: float = 0.5,
                 min_samples: int = 5,
                 max_compounds_warning: int = 10000,
                 max_compounds_error: int = 50000,
                 random_state: int = 42,
                 **tsne_params):
        """Initialize t-SNE+DBSCAN acquisition function.
        
        Args:
            data_manager: DataManager instance for feature extraction and caching
            featurizer_type: Type of molecular features ('morgan', 'ecfp6', 'maccs', 'descriptors')
            n_components: Number of t-SNE components (typically 2)
            perplexity: t-SNE perplexity parameter. If None, auto-determined from dataset size.
            early_exaggeration: Controls how tight natural clusters are in the original space
            learning_rate: Learning rate for t-SNE optimization ('auto' for automatic)
            max_iter: Maximum number of iterations for optimization
            eps: DBSCAN eps parameter (neighborhood radius)
            min_samples: DBSCAN min_samples parameter
            max_compounds_warning: Issue warning if dataset exceeds this size
            max_compounds_error: Raise error if dataset exceeds this size
            random_state: Random seed for reproducibility
            **tsne_params: Additional parameters for t-SNE
        """
        # Set t-SNE-specific defaults
        tsne_defaults = {
            'early_exaggeration': early_exaggeration,
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'method': 'barnes_hut',  # Faster approximation for larger datasets
            'angle': 0.5,  # Trade-off between speed and accuracy
        }
        
        # Add perplexity if specified
        if perplexity is not None:
            tsne_defaults['perplexity'] = perplexity
        
        tsne_defaults.update(tsne_params)
        
        super().__init__(
            data_manager=data_manager,
            featurizer_type=featurizer_type,
            n_components=n_components,
            random_state=random_state,
            **tsne_defaults
        )
        
        self.perplexity = perplexity
        self.eps = eps
        self.min_samples = min_samples
        self.max_compounds_warning = max_compounds_warning
        self.max_compounds_error = max_compounds_error
        
        logger.info(f"Initialized t-SNE+DBSCAN acquisition: "
                   f"featurizer={featurizer_type}, components={n_components}, "
                   f"perplexity={perplexity}, max_iter={max_iter}, "
                   f"eps={eps}, min_samples={min_samples}")
    
    @property
    def reduction_method(self) -> str:
        """Return the name of the dimensionality reduction method."""
        return 'tsne'
    
    def select(self, compounds, n_select):
        """Select compounds using t-SNE+DBSCAN clustering with size warnings."""
        # Validate input and check size constraints
        self.validate_input(compounds, n_select)
        
        n_compounds = len(compounds)
        
        if n_compounds > self.max_compounds_error:
            raise ValueError(f"Dataset too large for t-SNE ({n_compounds} compounds). "
                           f"t-SNE is not recommended for datasets larger than {self.max_compounds_error} compounds. "
                           f"Consider using UMAP instead.")
        elif n_compounds > self.max_compounds_warning:
            warnings.warn(f"Large dataset for t-SNE ({n_compounds} compounds). "
                         f"t-SNE may be slow for datasets larger than {self.max_compounds_warning} compounds. "
                         f"Consider using UMAP for better performance.",
                         UserWarning)
        
        # Auto-tune perplexity if not specified
        if self.perplexity is None:
            auto_perplexity = self._auto_tune_perplexity(n_compounds)
            logger.info(f"Auto-tuned perplexity: {auto_perplexity}")
            # Update reduction_params with auto-tuned perplexity
            self.reduction_params['perplexity'] = auto_perplexity
        
        return super().select(compounds, n_select)
    
    def _auto_tune_perplexity(self, n_compounds: int) -> float:
        """Auto-tune perplexity based on dataset size."""
        if n_compounds <= 50:
            return min(5.0, n_compounds - 1)
        elif n_compounds <= 500:
            return min(30.0, n_compounds / 4)
        elif n_compounds <= 5000:
            return min(50.0, n_compounds / 10)
        else:
            return min(100.0, n_compounds / 20)
    
    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering on t-SNE embeddings.
        
        Args:
            embeddings: 2D t-SNE embedding matrix
            
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
        return f"t-SNE+DBSCAN({self.featurizer_type})"


class TSNEKMeansAcquisition(ClusteringAcquisition):
    """t-SNE+K-Means acquisition for molecular diversity.
    
    Uses t-SNE followed by K-Means clustering. K-Means may be more stable
    for t-SNE embeddings where cluster density varies.
    """
    
    def __init__(self,
                 data_manager: DataManager,
                 featurizer_type: str = 'morgan',
                 n_components: int = 2,
                 perplexity: Optional[float] = None,
                 early_exaggeration: float = 12.0,
                 learning_rate: str = 'auto',
                 max_iter: int = 1000,
                 n_clusters: Optional[int] = None,
                 max_compounds_warning: int = 10000,
                 max_compounds_error: int = 50000,
                 random_state: int = 42,
                 **tsne_params):
        """Initialize t-SNE+K-Means acquisition function.
        
        Args:
            data_manager: DataManager instance for feature extraction and caching
            featurizer_type: Type of molecular features ('morgan', 'ecfp6', 'maccs', 'descriptors')
            n_components: Number of t-SNE components (typically 2)
            perplexity: t-SNE perplexity parameter. If None, auto-determined.
            early_exaggeration: Controls tightness of natural clusters
            learning_rate: Learning rate for t-SNE optimization
            max_iter: Maximum number of iterations
            n_clusters: Number of K-Means clusters. If None, estimated from data.
            max_compounds_warning: Issue warning if dataset exceeds this size
            max_compounds_error: Raise error if dataset exceeds this size
            random_state: Random seed for reproducibility
            **tsne_params: Additional parameters for t-SNE
        """
        # Set t-SNE-specific defaults
        tsne_defaults = {
            'early_exaggeration': early_exaggeration,
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'method': 'barnes_hut',
            'angle': 0.5,
        }
        
        if perplexity is not None:
            tsne_defaults['perplexity'] = perplexity
        
        tsne_defaults.update(tsne_params)
        
        super().__init__(
            data_manager=data_manager,
            featurizer_type=featurizer_type,
            n_components=n_components,
            random_state=random_state,
            **tsne_defaults
        )
        
        self.perplexity = perplexity
        self.n_clusters = n_clusters
        self.max_compounds_warning = max_compounds_warning
        self.max_compounds_error = max_compounds_error
        
        logger.info(f"Initialized t-SNE+K-Means acquisition: "
                   f"featurizer={featurizer_type}, components={n_components}, "
                   f"perplexity={perplexity}, n_clusters={n_clusters}")
    
    @property
    def reduction_method(self) -> str:
        """Return the name of the dimensionality reduction method."""
        return 'tsne'
    
    def select(self, compounds, n_select):
        """Select compounds using t-SNE+K-Means clustering with size warnings."""
        # Validate input and check size constraints
        self.validate_input(compounds, n_select)
        
        n_compounds = len(compounds)
        
        if n_compounds > self.max_compounds_error:
            raise ValueError(f"Dataset too large for t-SNE ({n_compounds} compounds)")
        elif n_compounds > self.max_compounds_warning:
            warnings.warn(f"Large dataset for t-SNE ({n_compounds} compounds)", UserWarning)
        
        # Auto-tune perplexity if not specified
        if self.perplexity is None:
            auto_perplexity = self._auto_tune_perplexity(n_compounds)
            logger.info(f"Auto-tuned perplexity: {auto_perplexity}")
            self.reduction_params['perplexity'] = auto_perplexity
        
        return super().select(compounds, n_select)
    
    def _auto_tune_perplexity(self, n_compounds: int) -> float:
        """Auto-tune perplexity based on dataset size."""
        if n_compounds <= 50:
            return min(5.0, n_compounds - 1)
        elif n_compounds <= 500:
            return min(30.0, n_compounds / 4)
        elif n_compounds <= 5000:
            return min(50.0, n_compounds / 10)
        else:
            return min(100.0, n_compounds / 20)
    
    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform K-Means clustering on t-SNE embeddings.
        
        Args:
            embeddings: 2D t-SNE embedding matrix
            
        Returns:
            Array of cluster labels
        """
        from sklearn.cluster import KMeans
        
        if len(embeddings) < 2:
            return np.zeros(len(embeddings))
        
        # Determine number of clusters
        if self.n_clusters is None:
            # For t-SNE visualizations, typically want more clusters than UMAP
            n_clusters = min(max(3, int(np.sqrt(len(embeddings) / 3))), len(embeddings) // 3)
        else:
            n_clusters = min(self.n_clusters, len(embeddings))
        
        logger.info(f"K-Means parameters: n_clusters={n_clusters}")
			
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        self.cluster_labels = kmeans.fit_predict(embeddings)

        unique_clusters = np.unique(self.cluster_labels)
        logger.info(f"K-Means clustering: {len(unique_clusters)} clusters from {len(embeddings)} compounds")

        return self.cluster_labels

    def get_name(self) -> str:
        """Return descriptive name for this acquisition function."""
        return f"t-SNE+K-Means({self.featurizer_type})"
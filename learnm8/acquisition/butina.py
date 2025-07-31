"""Butina clustering acquisition for molecular diversity.

This module implements Butina clustering algorithm for diverse molecular selection.
The algorithm creates clusters based on molecular similarity thresholds and selects
representative compounds from each cluster to ensure chemical diversity.

Reference: Butina, D. JCICS 39, 747-750 (1999)
"""

import logging
from typing import Optional, List, Tuple, Dict
import pandas as pd
import numpy as np
from collections import defaultdict

from learnm8.acquisition.base import AcquisitionFunction

# RDKit imports with graceful failure handling
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.ML.Cluster import Butina
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None
    DataStructs = None
    Butina = None

logger = logging.getLogger(__name__)


class ButinaClusteringAcquisition(AcquisitionFunction):
    """Acquisition function using Butina clustering for diverse molecular selection.
    
    Butina clustering is a deterministic algorithm that creates non-hierarchical clusters
    based on molecular similarity. It guarantees that every molecule in a cluster is
    within a specified similarity threshold to the cluster centroid.
    
    The acquisition strategy selects cluster centroids first (molecules with the most
    neighbors), then fills remaining slots from the largest clusters to maximize
    chemical diversity.
    """
    
    def __init__(self, 
                 threshold: float = 0.4,
                 featurizer_type: str = 'morgan',
                 fp_radius: int = 2,
                 fp_size: int = 1024,
                 max_compounds: int = 2000,
                 random_state: Optional[int] = None):
        """Initialize Butina clustering acquisition.
        
        Args:
            threshold: Distance threshold for clustering (0.2-0.5).
                      Lower values create tighter clusters.
                      Default 0.4 corresponds to 60% similarity cutoff.
            featurizer_type: Type of molecular fingerprint ('morgan', 'maccs')
            fp_radius: Morgan fingerprint radius (default 2)
            fp_size: Fingerprint bit size (default 1024)
            max_compounds: Maximum compounds to process (memory protection)
            random_state: Random seed for reproducible selection within clusters
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for Butina clustering acquisition. "
                            "Please install RDKit: conda install -c conda-forge rdkit")
        
        if not 0.1 <= threshold <= 0.8:
            raise ValueError(f"Threshold {threshold} not in valid range [0.1, 0.8]")
        
        self.threshold = threshold
        self.featurizer_type = featurizer_type
        self.fp_radius = fp_radius
        self.fp_size = fp_size
        self.max_compounds = max_compounds
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def get_name(self) -> str:
        """Return descriptive name for logging and identification."""
        return f"butina_t{self.threshold}"
    
    def _generate_fingerprints(self, smiles_list: List[str]) -> List[np.ndarray]:
        """Generate molecular fingerprints from SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of fingerprint bit vectors
            
        Raises:
            ValueError: If any SMILES is invalid
        """
        fingerprints = []
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
            
            if self.featurizer_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_size
                )
            elif self.featurizer_type == 'maccs':
                fp = AllChem.GetMACCSKeysFingerprint(mol)
            else:
                raise ValueError(f"Unsupported featurizer type: {self.featurizer_type}")
            
            fingerprints.append(fp)
        
        return fingerprints
    
    def _calculate_distance_matrix(self, fingerprints: List) -> List[float]:
        """Calculate Tanimoto distance matrix for Butina clustering.
        
        Args:
            fingerprints: List of RDKit fingerprint objects
            
        Returns:
            Distance matrix as flat list (lower triangle format for Butina)
        """
        n_mols = len(fingerprints)
        distance_matrix = []
        
        for i in range(n_mols):
            for j in range(i):
                # Tanimoto distance = 1 - Tanimoto similarity
                similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                distance = 1.0 - similarity
                distance_matrix.append(distance)
        
        return distance_matrix
    
    def _perform_butina_clustering(self, distance_matrix: List[float], 
                                  n_compounds: int) -> List[List[int]]:
        """Perform Butina clustering using RDKit implementation.
        
        Args:
            distance_matrix: Flat distance matrix (lower triangle)
            n_compounds: Number of compounds
            
        Returns:
            List of clusters, each cluster is a list of compound indices
        """
        clusters = Butina.ClusterData(
            distance_matrix, 
            n_compounds, 
            self.threshold, 
            isDistData=True
        )
        
        # Sort clusters by size (largest first) for consistent selection
        clusters = sorted(clusters, key=len, reverse=True)
        
        return clusters
    
    def _select_from_clusters(self, compounds: pd.DataFrame, 
                            clusters: List[List[int]], 
                            n_select: int) -> pd.DataFrame:
        """Select diverse compounds from Butina clusters.
        
        Selection strategy:
        1. Select cluster centroids first (index 0 in each cluster)
        2. If more compounds needed, select from largest remaining clusters
        3. Use prediction scores to break ties within clusters
        
        Args:
            compounds: DataFrame with compound data
            clusters: List of clusters from Butina algorithm
            n_select: Number of compounds to select
            
        Returns:
            Selected compounds with acquisition scores
        """
        selected_indices = []
        cluster_sizes = []
        
        # Phase 1: Select cluster centroids (first molecule in each cluster)
        for cluster in clusters:
            if len(selected_indices) >= n_select:
                break
            
            centroid_idx = cluster[0]  # First molecule is the centroid
            selected_indices.append(centroid_idx)
            cluster_sizes.append(len(cluster))
        
        # Phase 2: If more compounds needed, select from remaining cluster members
        if len(selected_indices) < n_select:
            # Create pool of remaining compounds, prioritizing larger clusters
            remaining_pool = []
            
            for cluster in clusters:
                # Skip centroid (already selected), add remaining cluster members
                for member_idx in cluster[1:]:
                    remaining_pool.append((member_idx, len(cluster)))
            
            # Sort by cluster size (desc) then by prediction score (desc)
            if remaining_pool:
                predictions = compounds['prediction'].values
                remaining_pool.sort(
                    key=lambda x: (x[1], predictions[x[0]]), 
                    reverse=True
                )
                
                # Select additional compounds
                needed = n_select - len(selected_indices)
                for compound_idx, cluster_size in remaining_pool[:needed]:
                    selected_indices.append(compound_idx)
                    cluster_sizes.append(cluster_size)
        
        # Create result DataFrame
        selected = compounds.iloc[selected_indices[:n_select]].copy()
        
        # Add acquisition scores (cluster size as diversity metric)
        selected['acquisition_score'] = cluster_sizes[:len(selected)]
        
        return selected
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select diverse compounds using Butina clustering.
        
        Args:
            compounds: DataFrame with columns ['ID', 'SMILES', 'prediction']
            n_select: Number of compounds to select
            
        Returns:
            DataFrame with selected compounds and acquisition scores
            
        Raises:
            ValueError: If input validation fails or clustering fails
            RuntimeError: If clustering produces no valid clusters
        """
        # Input validation
        self.validate_input(compounds, n_select)
        
        # Check dataset size for memory protection
        if len(compounds) > self.max_compounds:
            logger.warning(f"Dataset size {len(compounds)} exceeds maximum {self.max_compounds}. "
                          f"Consider using a different clustering method or reducing dataset size.")
            raise ValueError(f"Dataset too large for Butina clustering: {len(compounds)} > {self.max_compounds}")
        
        # Handle edge case: select all compounds
        if n_select >= len(compounds):
            result = compounds.copy()
            result['acquisition_score'] = 1.0  # All singletons
            return result
        
        try:
            # Generate molecular fingerprints
            smiles_list = compounds['SMILES'].tolist()
            fingerprints = self._generate_fingerprints(smiles_list)
            
            # Calculate distance matrix
            distance_matrix = self._calculate_distance_matrix(fingerprints)
            
            # Perform Butina clustering
            clusters = self._perform_butina_clustering(distance_matrix, len(compounds))
            
            if not clusters:
                raise RuntimeError("Butina clustering produced no clusters")
            
            logger.info(f"Butina clustering created {len(clusters)} clusters "
                       f"(largest: {len(clusters[0])}, threshold: {self.threshold})")
            
            # Select diverse compounds from clusters
            selected = self._select_from_clusters(compounds, clusters, n_select)
            
            return selected
            
        except Exception as e:
            logger.error(f"Butina clustering failed: {str(e)}")
            raise
"""BitBIRCH-based acquisition function for molecular diversity.

This module implements BitBIRCH clustering for molecular fingerprints, providing
native molecular diversity selection optimized for large-scale molecular libraries.
"""

import logging
import numpy as np
import pandas as pd
from typing import List

from .base import AcquisitionFunction
from ..core.data_manager import DataManager

logger = logging.getLogger(__name__)


class BitBIRCHAcquisition(AcquisitionFunction):
	"""BitBIRCH-based acquisition for molecular diversity.
	
	Uses the BitBIRCH algorithm specifically designed for molecular fingerprints
	with native Tanimoto similarity support. Provides exceptional scalability
	for large molecular libraries (1M+ compounds).
	
	Selects compounds evenly across clusters without utility scoring for
	straightforward diversity sampling.
	"""
	
	def __init__(self,
				 data_manager: DataManager,
				 featurizer_type: str = 'morgan',
				 threshold: float = 0.5,
				 branching_factor: int = 50,
				 random_state: int = 42):
		"""Initialize BitBIRCH acquisition function.
		
		Args:
			data_manager: DataManager instance for feature extraction
			featurizer_type: Type of molecular features ('morgan', 'ecfp6', 'maccs')
			threshold: BitBIRCH threshold parameter (molecular similarity threshold)
			branching_factor: Maximum number of subclusters in a node
			random_state: Random seed for reproducibility
		"""
		self.data_manager = data_manager
		self.featurizer_type = featurizer_type
		self.threshold = threshold
		self.branching_factor = branching_factor
		self.random_state = random_state
		
		if not (0.0 <= threshold <= 1.0):
			raise ValueError("threshold must be between 0.0 and 1.0")

		if branching_factor <= 2:
			raise ValueError("branching_factor must be greater than 2")
		
		if featurizer_type not in ['morgan', 'ecfp6', 'maccs']:
			logger.warning(f"BitBIRCH is optimized for binary fingerprints. "
						 f"'{featurizer_type}' may not work optimally.")
		
		# Check BitBIRCH availability
		self._bitbirch_available = self._check_bitbirch_availability()
		
		# Storage for clustering results (for validation/visualization)
		self.cluster_labels_ = None
		self.n_clusters_ = None
		self.cluster_mol_ids_ = None
		
	def _check_bitbirch_availability(self) -> bool:
		"""Check if BitBIRCH package is available."""
		try:
			import importlib.util
			spec = importlib.util.find_spec("bitbirch.bitbirch")
			return spec is not None
		except ImportError:
			logger.error("BitBIRCH package not found. Install with: "
						"pip install git+https://github.com/mqcomplab/bitbirch.git")
			return False
	
	def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
		"""Select compounds using BitBIRCH clustering.
		
		Args:
			compounds: DataFrame with 'ID', 'SMILES' columns and predictions
			n_select: Number of compounds to select
			
		Returns:
			DataFrame subset with selected compounds
			
		Raises:
			ImportError: If BitBIRCH package is not available
			ValueError: If required columns are missing or n_select is invalid
		"""
		# Validate input
		self.validate_input(compounds, n_select)
		
		if not self._bitbirch_available:
			raise ImportError("BitBIRCH package is required but not available. "
							"Install with: pip install git+https://github.com/mqcomplab/bitbirch.git")
		
		if n_select >= len(compounds):
			return compounds.copy()
		
		logger.info(f"Selecting {n_select} compounds using BitBIRCH clustering")
		
		# Get molecular fingerprints (binary features required for BitBIRCH)
		fingerprints = self.data_manager.get_features(
			compound_ids=compounds['ID'].tolist(),
			smiles_list=compounds['SMILES'].tolist(),
			featurizer_type=self.featurizer_type
		)
		
		# Ensure fingerprints are binary and int64 format (BitBIRCH requirement)
		fingerprints = self._prepare_fingerprints(fingerprints)
		
		cluster_mol_ids = self._bitbirch_clustering(fingerprints)
		
		# Store clustering results for validation/visualization
		self.cluster_mol_ids_ = cluster_mol_ids
		self.n_clusters_ = len(cluster_mol_ids) if cluster_mol_ids else 0
		self.cluster_labels_ = self._convert_clusters_to_labels(cluster_mol_ids, len(compounds))
		
		# Select representatives evenly from clusters
		selected_indices = self._select_from_bitbirch_clusters(
			compounds, cluster_mol_ids, n_select
		)
		
		return compounds.iloc[selected_indices].copy()
	
	def _prepare_fingerprints(self, fingerprints: np.ndarray) -> np.ndarray:
		"""Prepare fingerprints for BitBIRCH (binary, int64 format).
		
		Args:
			fingerprints: Raw fingerprint array
			
		Returns:
			Prepared fingerprint array
		"""
		# Convert to binary if not already
		if fingerprints.dtype not in [np.int64, np.float64, np.int32]:
			fingerprints = fingerprints.astype(np.int64)
		
		# Ensure binary values
		if not np.all(np.isin(fingerprints, [0, 1])):
			logger.warning("Non-binary fingerprint values detected. Converting to binary.")
			fingerprints = (fingerprints > 0).astype(np.int64)
		
		# Convert to int64 (BitBIRCH requirement)
		fingerprints = fingerprints.astype(np.int64)
		
		logger.info(f"Prepared {len(fingerprints)} fingerprints for BitBIRCH "
				   f"(shape: {fingerprints.shape}, binary: {np.all(np.isin(fingerprints, [0, 1]))})")
		
		return fingerprints
	
	def _bitbirch_clustering(self, fingerprints: np.ndarray) -> List[List[int]]:
		"""Perform basic BitBIRCH clustering.
		
		Args:
			fingerprints: Binary fingerprint array
			
		Returns:
			List of clusters, where each cluster is a list of molecule indices
		"""
		import bitbirch.bitbirch as bb
		
		# Set merging strategy to diameter
		bb.set_merge('diameter')
		
		# Initialize BitBIRCH
		brc = bb.BitBirch(
			threshold=self.threshold,
			branching_factor=self.branching_factor
		)
		
		# Fit to fingerprints
		brc.fit(fingerprints)
		
		# Get cluster assignments
		mol_ids = brc.get_cluster_mol_ids()
		
		logger.info(f"BitBIRCH clustering: {len(mol_ids)} clusters from {len(fingerprints)} compounds")
		
		return mol_ids
	
	def _convert_clusters_to_labels(self, cluster_mol_ids: List[List[int]], n_compounds: int) -> np.ndarray:
		"""Convert cluster assignments to label array for visualization.
		
		Args:
			cluster_mol_ids: List of clusters from BitBIRCH
			n_compounds: Total number of compounds
			
		Returns:
			Array of cluster labels (-1 for noise, 0+ for cluster IDs)
		"""
		labels = np.full(n_compounds, -1, dtype=int)  # Initialize with -1 (noise)
		
		for cluster_id, mol_indices in enumerate(cluster_mol_ids):
			for mol_idx in mol_indices:
				if 0 <= mol_idx < n_compounds:
					labels[mol_idx] = cluster_id
		
		return labels
	
	def _select_from_bitbirch_clusters(self,
									  compounds: pd.DataFrame,
									  cluster_mol_ids: List[List[int]],
									  n_select: int) -> List[int]:
		"""Select compounds evenly from BitBIRCH clusters.
		
		Args:
			compounds: Input compounds DataFrame
			cluster_mol_ids: List of clusters from BitBIRCH
			n_select: Number of compounds to select
			
		Returns:
			List of selected compound indices
		"""
		if not cluster_mol_ids:
			logger.warning("No clusters found, falling back to random selection")
			np.random.seed(self.random_state)
			return np.random.choice(len(compounds), n_select, replace=False).tolist()
		
		# Sort clusters by size (descending) for consistent ordering
		sorted_clusters = sorted(cluster_mol_ids, key=len, reverse=True)
		n_clusters = len(sorted_clusters)
		
		logger.info(f"Selecting {n_select} compounds evenly from {n_clusters} clusters")
		
		selected_indices = []
		np.random.seed(self.random_state)
		
		if n_select <= n_clusters:
			# Select one compound from the largest clusters
			for i in range(n_select):
				cluster = sorted_clusters[i]
				# Randomly select from cluster
				selected_idx = np.random.choice(cluster)
				selected_indices.append(selected_idx)
		
		else:
			# Distribute selections evenly across clusters
			base_per_cluster = n_select // n_clusters
			remainder = n_select % n_clusters
			
			for i, cluster in enumerate(sorted_clusters):
				# Determine how many to select from this cluster
				n_from_cluster = base_per_cluster
				if i < remainder:
					n_from_cluster += 1
				
				# Don't select more than available in cluster
				n_from_cluster = min(n_from_cluster, len(cluster))
				
				if n_from_cluster > 0:
					# Randomly select compounds from cluster
					cluster_selections = np.random.choice(
						cluster, n_from_cluster, replace=False
					).tolist()
					selected_indices.extend(cluster_selections)
		
		# Fill any remaining slots with random compounds from any cluster
		if len(selected_indices) < n_select:
			all_remaining = []
			for cluster in sorted_clusters:
				for idx in cluster:
					if idx not in selected_indices:
						all_remaining.append(idx)
			
			remaining_needed = n_select - len(selected_indices)
			if all_remaining and remaining_needed > 0:
				additional = np.random.choice(
					all_remaining, 
					min(remaining_needed, len(all_remaining)), 
					replace=False
				).tolist()
				selected_indices.extend(additional)
		
		return selected_indices[:n_select]
	
	def get_name(self) -> str:
		"""Return descriptive name for this acquisition function."""
		return f"BitBIRCH({self.featurizer_type})"
	
	def requires_uncertainty(self) -> bool:
		"""BitBIRCH does not require uncertainty estimates."""
		return False
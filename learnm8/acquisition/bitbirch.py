"""BitBIRCH-based acquisition function for molecular diversity.

This module implements BitBIRCH clustering for molecular fingerprints, providing
native molecular diversity selection optimized for large-scale molecular libraries.
"""

import logging
import numpy as np
import polars as pl

import bitbirch.bitbirch as bb

from .base import AcquisitionFunction

logger = logging.getLogger(__name__)


class BitBIRCHAcquisition(AcquisitionFunction):
	"""BitBIRCH-based acquisition for molecular diversity.

	Uses the BitBIRCH algorithm specifically designed for molecular fingerprints
	with native Tanimoto similarity support. Provides exceptional scalability
	for large molecular libraries (1M+ compounds).

	Selects compounds evenly across clusters without utility scoring for
	straightforward diversity sampling.

	Features must be pre-computed and passed during initialization.
	"""

	def __init__(self,
				 features: np.ndarray,
				 compound_ids: list[str],
				 featurizer: str = 'morgan',
				 threshold: float = 0.5,
				 branching_factor: int = 50,
				 random_state: int = 42):
		"""Initialize BitBIRCH acquisition function.

		Args:
			features: Pre-computed molecular fingerprints (n_compounds, n_features)
			compound_ids: List of compound IDs corresponding to feature rows
			featurizer: Type of molecular features ('morgan', 'ecfp6', 'maccs')
			threshold: BitBIRCH threshold parameter (molecular similarity threshold)
			branching_factor: Maximum number of subclusters in a node
			random_state: Random seed for reproducibility
		"""
		super().__init__(score_direction='higher')
		self.features = features
		self.compound_ids = compound_ids
		self._id_to_idx = {cid: idx for idx, cid in enumerate(compound_ids)}
		self.featurizer = featurizer
		self.threshold = threshold
		self.branching_factor = branching_factor
		self.random_state = random_state
		
		if not (0.0 <= threshold <= 1.0):
			raise ValueError("threshold must be between 0.0 and 1.0")

		if branching_factor <= 2:
			raise ValueError("branching_factor must be greater than 2")
		
		if featurizer not in ['morgan', 'ecfp6', 'maccs']:
			logger.warning(f"BitBIRCH is optimized for binary fingerprints. "
						 f"'{featurizer}' may not work optimally.")

		# Storage for clustering results (for validation/visualization)
		self.cluster_labels_ = None
		self.n_clusters_ = None
		self.cluster_mol_ids_ = None
	
	def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
		"""Select compounds using BitBIRCH clustering.

		Args:
			compounds: DataFrame with 'ID', 'SMILES' columns and predictions
			n_select: Number of compounds to select

		Returns:
			DataFrame subset with selected compounds

		Raises:
			ValueError: If required columns are missing or n_select is invalid
		"""
		# Validate input
		self.validate_input(compounds, n_select)

		if n_select >= len(compounds):
			return compounds.clone()

		logger.info(f"Selecting {n_select} compounds using BitBIRCH clustering")

		requested_ids = compounds.get_column('ID').to_list()
		valid_ids = [cid for cid in requested_ids if cid in self._id_to_idx]

		if len(valid_ids) == 0:
			logger.warning("No matching features for provided IDs. Returning empty selection.")
			return compounds.head(0).clone()

		if len(valid_ids) < len(requested_ids):
			missing_ids = set(requested_ids) - set(valid_ids)
			logger.warning(f"Only {len(valid_ids)}/{len(requested_ids)} compounds have valid features. "
						  f"Missing IDs: {list(missing_ids)[:5]}{'...' if len(missing_ids) > 5 else ''}")
			compounds = compounds.filter(pl.col('ID').is_in(valid_ids))

		if n_select > len(compounds):
			logger.warning(
				f"n_select ({n_select}) exceeds available compounds with features ({len(compounds)}); selecting all available"
			)
			n_select = len(compounds)

		indices = [self._id_to_idx[cid] for cid in valid_ids]
		fingerprints = self.features[indices]

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

		# Get IDs for selected indices and filter
		all_ids = compounds.get_column('ID').to_numpy()
		selected_ids = all_ids[selected_indices]

		return compounds.filter(pl.col('ID').is_in(selected_ids.tolist()))
	
	def _prepare_fingerprints(self, fingerprints: np.ndarray) -> np.ndarray:
		"""Prepare fingerprints for BitBIRCH (binary, int64 format).

		Args:
			fingerprints: Raw fingerprint array

		Returns:
			Prepared fingerprint array
		"""
		if not np.all(np.isin(fingerprints, [0, 1])):
			logger.warning("Non-binary fingerprint values detected. Converting to binary.")
			fingerprints = (fingerprints > 0).astype(np.int64)
		else:
			fingerprints = fingerprints.astype(np.int64)
		logger.info(f"Prepared {len(fingerprints)} fingerprints for BitBIRCH (shape: {fingerprints.shape}, binary: {np.all(np.isin(fingerprints, [0, 1]))})")
		return fingerprints
	
	def _bitbirch_clustering(self, fingerprints: np.ndarray) -> list[list[int]]:
		"""Perform basic BitBIRCH clustering.

		Args:
			fingerprints: Binary fingerprint array

		Returns:
			List of clusters, where each cluster is a list of molecule indices
		"""
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
	
	def _convert_clusters_to_labels(self, cluster_mol_ids: list[list[int]], n_compounds: int) -> np.ndarray:
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
									  compounds: pl.DataFrame,
									  cluster_mol_ids: list[list[int]],
									  n_select: int) -> list[int]:
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
		return f"BitBIRCH({self.featurizer})"
	
	def requires_uncertainty(self) -> bool:
		"""BitBIRCH does not require uncertainty estimates."""
		return False
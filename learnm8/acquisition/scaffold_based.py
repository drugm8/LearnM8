"""Scaffold-based acquisition for chemical structure diversity.

This module implements scaffold-based molecular selection using Bemis-Murcko scaffolds
to ensure diversity in chemical structural frameworks. The implementation is extracted
from the astartes library for integration with LearnM8.
"""

import logging
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from collections import defaultdict

from learnm8.acquisition.base import AcquisitionFunction
from learnm8.acquisition.astartes_utils import validate_acquisition_input

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# RDKit imports with graceful failure handling
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    MurckoScaffold = None

logger = logging.getLogger(__name__)


def generate_scaffolds(smiles_list: List[str], 
					  include_chirality: bool = False,
					  n_jobs: int = -1) -> List[Optional[str]]:
	"""Generate Bemis-Murcko scaffolds from SMILES strings.
	
	Args:
		smiles_list: List of SMILES strings
		include_chirality: Whether to consider chirality in scaffold generation
		n_jobs: Number of parallel jobs (-1 for all cores)
		
	Returns:
		List of scaffold SMILES (None for invalid molecules)
		
	Raises:
		ImportError: If RDKit is not available
	"""
	if not RDKIT_AVAILABLE:
		raise ImportError("RDKit is required for scaffold-based acquisition. "
						 "Please install RDKit: conda install -c conda-forge rdkit")
	
	
	def _generate_single_scaffold(smiles: str) -> Optional[str]:
		"""Generate scaffold for a single SMILES string."""
		try:
			mol = Chem.MolFromSmiles(smiles)
			if mol is None:
				return None
			
			scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
			if scaffold_mol is None:
				return None
			
			return Chem.MolToSmiles(scaffold_mol, isomericSmiles=include_chirality)
			
		except Exception:
			return None
	
	# Determine number of workers
	n_cores = mp.cpu_count() if n_jobs == -1 else min(n_jobs, mp.cpu_count())
	
	# Use threading for small datasets, multiprocessing for large ones
	if len(smiles_list) < 100:
		scaffolds = [_generate_single_scaffold(smiles) for smiles in smiles_list]
	else:
		with ProcessPoolExecutor(max_workers=n_cores) as executor:
			scaffolds = list(executor.map(_generate_single_scaffold, smiles_list))
	
	# Log warnings for failed scaffolds
	failed_count = scaffolds.count(None)
	if failed_count > 0:
		logger.warning(f"Failed to generate scaffolds for {failed_count}/{len(smiles_list)} molecules")
	
	return scaffolds


def group_by_scaffold(compounds: pd.DataFrame,
                     scaffolds: List[Optional[str]]) -> Dict[str, List[int]]:
    """Group compound indices by scaffold.
    
    Args:
        compounds: DataFrame of compounds
        scaffolds: List of scaffold SMILES (parallel to compounds)
        
    Returns:
        Dictionary mapping scaffold SMILES to lists of compound indices
    """
    scaffold_groups = defaultdict(list)
    
    for idx, scaffold in enumerate(scaffolds):
        if scaffold is not None:
            scaffold_groups[scaffold].append(idx)
        else:
            # Create unique group for compounds without valid scaffolds
            scaffold_groups[f"invalid_{idx}"].append(idx)
    
    return dict(scaffold_groups)


class ScaffoldAcquisition(AcquisitionFunction):
    """Scaffold-based acquisition for chemical structural diversity.
    
    The scaffold-based algorithm ensures diversity in chemical frameworks by:
    1. Generating Bemis-Murcko scaffolds from molecular structures
    2. Grouping compounds by shared structural scaffolds
    3. Selecting representatives from different scaffold families
    4. Prioritizing prediction scores within scaffold groups
    
    This provides chemically-informed diversity that complements fingerprint-based
    methods by focusing on core structural frameworks rather than detailed features.
    
    Args:
        include_chirality: Whether to consider chirality in scaffold generation
        random_state: Random seed for reproducible selection
    """
    
    def __init__(self,
                 include_chirality: bool = False,
                 random_state: Optional[int] = 42):
        self.include_chirality = include_chirality
        self.random_state = random_state
        
        # Set numpy random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select diverse compounds using scaffold-based clustering.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction' columns
            n_select: Number of compounds to select
            
        Returns:
            DataFrame with selected compounds from different scaffold families
            
        Raises:
            ValueError: If inputs are invalid
            ImportError: If RDKit is not available
            RuntimeError: If scaffold generation fails
        """
        # Validate inputs
        self.validate_input(compounds, n_select)
        validate_acquisition_input(compounds, n_select)
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for scaffold-based acquisition")
        
        logger.info(f"Starting scaffold-based selection of {n_select} compounds "
                   f"from {len(compounds)} candidates "
                   f"(include_chirality={self.include_chirality})")
        
        # Handle edge cases
        if n_select >= len(compounds):
            logger.info("Selecting all available compounds")
            return compounds.copy()
        
        try:
            # Generate scaffolds from SMILES
            smiles_list = compounds['SMILES'].tolist()
            logger.info("Generating molecular scaffolds...")
            
            scaffolds = generate_scaffolds(
                smiles_list=smiles_list,
                include_chirality=self.include_chirality
            )
            
            # Group compounds by scaffold
            scaffold_groups = group_by_scaffold(compounds, scaffolds)
            
            # Log scaffold statistics
            valid_scaffolds = [s for s in scaffolds if s is not None]
            unique_scaffolds = len(set(valid_scaffolds))
            invalid_count = scaffolds.count(None)
            
            logger.info(f"Found {unique_scaffolds} unique scaffolds from "
                       f"{len(compounds)} compounds ({invalid_count} invalid)")
            
            # Select representatives from scaffold groups
            selected_indices = self._select_scaffold_representatives(
                compounds=compounds,
                scaffold_groups=scaffold_groups,
                n_select=n_select
            )
            
            # Build result DataFrame
            selected_compounds = compounds.iloc[selected_indices].copy()
            
            # Add acquisition metadata
            selected_compounds['acquisition_score'] = np.arange(len(selected_indices), 0, -1)
            
            # Add scaffold information
            selected_scaffolds = [scaffolds[i] for i in selected_indices]
            selected_compounds['scaffold'] = selected_scaffolds
            
            logger.info(f"Successfully selected {len(selected_compounds)} compounds "
                       f"from {len(scaffold_groups)} scaffold families")
            
            return selected_compounds
            
        except Exception as e:
            logger.error(f"Scaffold-based selection failed: {str(e)}")
            raise RuntimeError(f"Scaffold-based acquisition failed: {str(e)}") from e
    
    def _select_scaffold_representatives(self,
                                        compounds: pd.DataFrame,
                                        scaffold_groups: Dict[str, List[int]],
                                        n_select: int) -> List[int]:
        """Select representative compounds from scaffold groups.
        
        Strategy:
        1. Sort scaffold groups by size (largest first) 
        2. Round-robin selection from groups
        3. Within each group, prefer highest prediction scores
        
        Args:
            compounds: Original compounds DataFrame
            scaffold_groups: Mapping from scaffold to compound indices
            n_select: Number of compounds to select
            
        Returns:
            List of selected compound indices
        """
        # Sort groups by size (largest first) for better coverage
        sorted_groups = sorted(scaffold_groups.items(), 
                              key=lambda x: len(x[1]), reverse=True)
        
        selected_indices = []
        group_positions = {scaffold: 0 for scaffold, _ in sorted_groups}
        
        # Round-robin selection from groups
        while len(selected_indices) < n_select and sorted_groups:
            added_this_round = False
            
            for scaffold, indices in sorted_groups:
                if len(selected_indices) >= n_select:
                    break
                
                pos = group_positions[scaffold]
                if pos < len(indices):
                    # Sort group indices by prediction score (descending)
                    group_predictions = compounds.iloc[indices]['prediction'].values
                    sorted_group_indices = [indices[i] for i in np.argsort(group_predictions)[::-1]]
                    
                    # Select next compound from this group
                    selected_indices.append(sorted_group_indices[pos])
                    group_positions[scaffold] += 1
                    added_this_round = True
            
            # If no compounds were added, all groups are exhausted
            if not added_this_round:
                break
        
        return selected_indices[:n_select]
    
    def requires_uncertainty(self) -> bool:
        """Scaffold-based acquisition doesn't require uncertainty estimates."""
        return False
    
    def get_name(self) -> str:
        """Return descriptive name for this acquisition function."""
        chiral_suffix = "+chiral" if self.include_chirality else ""
        return f"Scaffold{chiral_suffix}"


def create_scaffold_acquisition(include_chirality: bool = False,
                               random_state: Optional[int] = 42) -> ScaffoldAcquisition:
    """Factory function for creating ScaffoldAcquisition instances.
    
    Args:
        include_chirality: Whether to consider chirality in scaffold generation
        random_state: Random seed for reproducible selection
        
    Returns:
        Configured ScaffoldAcquisition instance
    """
    return ScaffoldAcquisition(
        include_chirality=include_chirality,
        random_state=random_state
    )
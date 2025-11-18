"""
Molecular featurization utilities for fingerprints and descriptors.
Provides both direct computation and file-based caching of molecular representations.
"""

import os
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import pickle
import logging
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, AllChem
from joblib import Parallel, delayed


def smiles_to_morgan_fingerprint(smiles: str) -> np.ndarray:
    """
    Convert a single SMILES string to Morgan fingerprint.
    
    Args:
        smiles: SMILES string representation of molecule
        
    Returns:
        Morgan fingerprint as numpy array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = morgan_gen.GetFingerprint(mol)
    arr = np.zeros(len(fp), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def smiles_to_morgan_feature_fingerprint(smiles: str) -> np.ndarray:
    """
    Convert a single SMILES string to Morgan feature fingerprint.

    Uses feature-based Morgan fingerprints which encode pharmacophore features
    rather than exact atom types.

    Args:
        smiles: SMILES string representation of molecule

    Returns:
        Morgan feature fingerprint as numpy array (2048-bit)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fp = AllChem.GetHashedMorganFingerprint(mol, radius=2, nBits=2048, useFeatures=True)
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr



def smiles_to_maccs_fingerprint(smiles: str) -> np.ndarray:
    """
    Convert a single SMILES string to MACCS fingerprint.
    
    Args:
        smiles: SMILES string representation of molecule
        
    Returns:
        MACCS fingerprint as numpy array (167 bits)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    arr = np.zeros(len(maccs_fp), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(maccs_fp, arr)
    return arr


def smiles_to_ecfp6_fingerprint(smiles: str) -> np.ndarray:
    """
    Convert a single SMILES string to ECFP6 fingerprint.
    
    Args:
        smiles: SMILES string representation of molecule
        
    Returns:
        ECFP6 fingerprint as numpy array (2048 bits)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    ecfp6_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fp = ecfp6_gen.GetFingerprint(mol)
    arr = np.zeros(len(fp), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_fingerprints(smiles_list: list[str], featurizer_type: str = "morgan", n_jobs: int = -1) -> np.ndarray:
    """
    Convert list of SMILES to fingerprints in parallel.
    
    Args:
        smiles_list: List of SMILES strings
        featurizer_type: Type of fingerprint ("morgan", "maccs", "ecfp6")
        n_jobs: Number of parallel jobs (-1 uses all CPUs, capped at 32)
        
    Returns:
        Array of shape (n_compounds, n_features) with fingerprints
    """
    if n_jobs == -1:
        n_jobs = min(os.cpu_count() or 1, 32)
    
    if featurizer_type == "morgan":
        fingerprint_func = smiles_to_morgan_fingerprint
    elif featurizer_type == "maccs":
        fingerprint_func = smiles_to_maccs_fingerprint
    elif featurizer_type == "ecfp6":
        fingerprint_func = smiles_to_ecfp6_fingerprint
    else:
        raise ValueError(f"Unknown featurizer type: {featurizer_type}")
    
    fingerprints = Parallel(n_jobs=n_jobs)(
        delayed(fingerprint_func)(smiles) 
        for smiles in smiles_list
    )
    
    return np.array(fingerprints)


def _compute_mordred_descriptors(smiles_list: List[str]) -> pl.DataFrame:
    """Compute Mordred descriptors for a list of SMILES.

    Note:
        The Mordred library only provides a pandas API (Calculator.pandas()).
        We convert the pandas DataFrame to Polars immediately after computation
        for consistency with the rest of the LearnM8 codebase. This conversion
        overhead is acceptable as descriptor computation is cached.
    """
    from mordred import Calculator, descriptors
    import pandas as pd  # Required by Mordred library

    # Create calculator
    calc = Calculator(descriptors, ignore_3D=True)

    # Convert SMILES to molecules, keeping track of valid indices
    mols = []
    valid_indices = []

    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                valid_indices.append(i)
        except Exception:
            # Skip invalid molecules
            continue

    if not mols:
        # No valid molecules, return empty dataframe with correct structure
        # Calculate descriptors for a dummy molecule to get column structure
        dummy_mol = Chem.MolFromSmiles('C')
        dummy_df = calc.pandas([dummy_mol])
        empty_df = pd.DataFrame(columns=dummy_df.columns)
        # Add rows filled with zeros for all input molecules
        for _ in range(len(smiles_list)):
            empty_df.loc[len(empty_df)] = 0
        empty_df = empty_df.astype(np.float32)
        return pl.from_pandas(empty_df)  # Convert to Polars for consistency

    # Calculate descriptors for valid molecules (Mordred returns pandas)
    descriptors_df = calc.pandas(mols)

    # Fill NaN values for molecules that couldn't be processed
    descriptors_df = descriptors_df.fillna(0)

    # Cast to float32
    descriptors_df = descriptors_df.astype(np.float32)

    # Create final dataframe with all input molecules
    if len(valid_indices) == len(smiles_list):
        # All molecules were valid - convert to Polars
        return pl.from_pandas(descriptors_df)
    else:
        # Some molecules were invalid, need to insert zero rows
        final_df = pd.DataFrame(columns=descriptors_df.columns)
        valid_idx = 0

        for i in range(len(smiles_list)):
            if i in valid_indices:
                # Use computed descriptors
                final_df.loc[len(final_df)] = descriptors_df.iloc[valid_idx]
                valid_idx += 1
            else:
                # Fill with zeros for invalid molecules
                final_df.loc[len(final_df)] = 0

        final_df = final_df.astype(np.float32)
        return pl.from_pandas(final_df)  # Convert to Polars for consistency


def _get_representation_file(results_dir: Path, featurizer_type: str) -> Path:
    """Get the file path for storing representations."""
    cache_dir = results_dir / ".representations"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{featurizer_type}.pkl"


def _compute_and_store_representations(
    compounds: pl.DataFrame,
    featurizer_type: str,
    results_dir: Path,
    logger: Optional[logging.Logger] = None
) -> None:
    """Compute and store representations to file."""
    if logger:
        logger.info(f"Computing {featurizer_type} representations for {len(compounds)} compounds")

    representations = {}

    if featurizer_type in ["morgan", "maccs", "ecfp6"]:
        # Compute fingerprints
        fingerprints = smiles_to_fingerprints(compounds.get_column('SMILES').to_list(), featurizer_type)
        for idx, compound_id in enumerate(compounds.get_column('ID').to_list()):
            representations[compound_id] = fingerprints[idx]

    elif featurizer_type == "descriptors":
        # Compute Mordred descriptors
        descriptors_df = _compute_mordred_descriptors(compounds.get_column('SMILES').to_list())
        for idx, compound_id in enumerate(compounds.get_column('ID').to_list()):
            representations[compound_id] = descriptors_df.row(idx)

    else:
        raise ValueError(f"Unknown featurizer type: {featurizer_type}. Supported: morgan, maccs, ecfp6, descriptors")

    # Store to file
    representation_file = _get_representation_file(results_dir, featurizer_type)
    with open(representation_file, 'wb') as f:
        pickle.dump(representations, f)

    if logger:
        logger.info(f"Stored {featurizer_type} representations to {representation_file}")


def _load_representations(results_dir: Path, featurizer_type: str) -> Dict:
    """Load representations from file."""
    representation_file = _get_representation_file(results_dir, featurizer_type)
    
    if not representation_file.exists():
        return {}
    
    with open(representation_file, 'rb') as f:
        return pickle.load(f)


def get_fingerprints(compound_ids: List[str], results_dir: Path, featurizer_type: str = "morgan") -> np.ndarray:
    """Get fingerprints for specified compound IDs."""
    representations = _load_representations(results_dir, featurizer_type)
    
    fingerprints = []
    for compound_id in compound_ids:
        if compound_id in representations:
            fingerprints.append(representations[compound_id])
        else:
            raise KeyError(f"Fingerprint not found for compound {compound_id}")
    
    return np.array(fingerprints)


def get_descriptors(compound_ids: List[str], results_dir: Path) -> np.ndarray:
    """Get Mordred descriptors for specified compound IDs."""
    representations = _load_representations(results_dir, "descriptors")
    
    descriptors = []
    for compound_id in compound_ids:
        if compound_id in representations:
            descriptors.append(representations[compound_id])
        else:
            raise KeyError(f"Descriptors not found for compound {compound_id}")
    
    return np.array(descriptors)


def precompute_representations(
    compounds: pl.DataFrame,
    featurizer_type: str,
    results_dir: Path,
    logger: Optional[logging.Logger] = None
) -> None:
    """Precompute and store representations for all compounds."""
    representation_file = _get_representation_file(results_dir, featurizer_type)

    if representation_file.exists():
        if logger:
            logger.info(f"Representations already exist at {representation_file}")
        return

    _compute_and_store_representations(compounds, featurizer_type, results_dir, logger)
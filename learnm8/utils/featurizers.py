"""
Molecular featurization utilities for fingerprints and descriptors.
Provides both direct computation and file-based caching of molecular representations.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import pickle
import logging
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
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
    return np.array(fp)


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
    return np.array(maccs_fp)


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
    return np.array(fp)


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


def _compute_mordred_descriptors(smiles_list: List[str]) -> pd.DataFrame:
    """Compute Mordred descriptors for a list of SMILES."""
    from mordred import Calculator, descriptors
    
    # Create calculator
    calc = Calculator(descriptors, ignore_3D=True)
    
    # Convert SMILES to molecules
    mols = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mols.append(None)
            else:
                mols.append(mol)
        except Exception:
            mols.append(None)
    
    # Calculate descriptors
    descriptors_df = calc.pandas(mols)
    
    # Fill NaN values for invalid molecules
    descriptors_df = descriptors_df.fillna(0)
    
    return descriptors_df


def _get_representation_file(results_dir: Path, featurizer_type: str) -> Path:
    """Get the file path for storing representations."""
    cache_dir = results_dir / ".representations"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{featurizer_type}.pkl"


def _compute_and_store_representations(
    compounds: pd.DataFrame, 
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
        fingerprints = smiles_to_fingerprints(compounds['SMILES'].tolist(), featurizer_type)
        for idx, compound_id in enumerate(compounds['ID']):
            representations[compound_id] = fingerprints[idx]
    
    elif featurizer_type == "descriptors":
        # Compute Mordred descriptors
        descriptors_df = _compute_mordred_descriptors(compounds['SMILES'].tolist())
        for idx, compound_id in enumerate(compounds['ID']):
            representations[compound_id] = descriptors_df.iloc[idx].values
    
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
    compounds: pd.DataFrame,
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
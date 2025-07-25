"""Chemistry utilities for molecular fingerprint generation."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from joblib import Parallel, delayed
import os


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


def smiles_to_fingerprints(smiles_list: list[str], n_jobs: int = -1) -> np.ndarray:
    """
    Convert list of SMILES to Morgan fingerprints in parallel.
    
    Args:
        smiles_list: List of SMILES strings
        n_jobs: Number of parallel jobs (-1 uses all CPUs, capped at 32)
        
    Returns:
        Array of shape (n_compounds, n_features) with fingerprints
    """
    if n_jobs == -1:
        n_jobs = min(os.cpu_count() or 1, 32)
    
    fingerprints = Parallel(n_jobs=n_jobs)(
        delayed(smiles_to_morgan_fingerprint)(smiles) 
        for smiles in smiles_list
    )
    
    return np.array(fingerprints)
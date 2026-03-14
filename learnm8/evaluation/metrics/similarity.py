"""Molecular similarity and diversity metrics for active learning evaluation."""

import numpy as np
import polars as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def _generate_fingerprints(smiles_list: list) -> list:
    """
    Generate Morgan fingerprints for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of Morgan fingerprint objects
    """
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fingerprints.append(None)  # Keep index alignment
        else:
            fp = morgan_gen.GetFingerprint(mol)
            fingerprints.append(fp)

    return fingerprints


def _calculate_average_tanimoto_similarity(fingerprints1: list, fingerprints2: list) -> float:
    """
    Calculate average Tanimoto similarity between two sets of fingerprints.

    Args:
        fingerprints1: First set of fingerprints
        fingerprints2: Second set of fingerprints

    Returns:
        Average Tanimoto similarity (0-1)
    """
    similarities = []

    # Filter out None fingerprints
    valid_fp1 = [fp for fp in fingerprints1 if fp is not None]
    valid_fp2 = [fp for fp in fingerprints2 if fp is not None]

    if len(valid_fp1) == 0 or len(valid_fp2) == 0:
        return 0.0

    for fp1 in valid_fp1:
        for fp2 in valid_fp2:
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            similarities.append(sim)

    return np.mean(similarities) if similarities else 0.0


def _calculate_intra_batch_diversity(fingerprints: list) -> float:
    """
    Calculate average pairwise Tanimoto distance within a batch.

    Args:
        fingerprints: List of fingerprints

    Returns:
        Average pairwise distance (0-1)
    """
    # Filter out None fingerprints
    valid_fps = [fp for fp in fingerprints if fp is not None]

    if len(valid_fps) < 2:
        return 0.0

    distances = []

    for i in range(len(valid_fps)):
        for j in range(i + 1, len(valid_fps)):
            sim = DataStructs.TanimotoSimilarity(valid_fps[i], valid_fps[j])
            distance = 1.0 - sim
            distances.append(distance)

    return np.mean(distances) if distances else 0.0


def calculate_molecular_similarity_metrics(newly_selected_df: pl.DataFrame,
                                         previously_selected_df: pl.DataFrame = None) -> dict:
    """
    Calculate molecular similarity metrics for the acquisition step.

    Args:
        newly_selected_df: DataFrame with newly selected compounds (must have 'SMILES' column)
        previously_selected_df: DataFrame with previously selected compounds (optional)

    Returns:
        Dictionary with molecular similarity metrics
    """
    results = {
        'intra_batch_diversity': 0.0,
        'inter_cycle_similarity': 0.0,
        'batch_novelty_score': 0.0
    }

    # Check if SMILES column exists
    if 'SMILES' not in newly_selected_df.columns:
        return results

    new_smiles = newly_selected_df.get_column('SMILES').to_list()
    new_fingerprints = _generate_fingerprints(new_smiles)

    # Calculate intra-batch diversity
    results['intra_batch_diversity'] = _calculate_intra_batch_diversity(new_fingerprints)

    # Calculate inter-cycle similarity and novelty if previous compounds exist
    if previously_selected_df is not None and 'SMILES' in previously_selected_df.columns and len(previously_selected_df) > 0:
        prev_smiles = previously_selected_df.get_column('SMILES').to_list()
        prev_fingerprints = _generate_fingerprints(prev_smiles)

        # Inter-cycle similarity (how similar new compounds are to previous ones)
        results['inter_cycle_similarity'] = _calculate_average_tanimoto_similarity(new_fingerprints, prev_fingerprints)

        # Batch novelty score (fraction of compounds with distance > 0.4 to all previous)
        valid_new_fps = [fp for fp in new_fingerprints if fp is not None]
        valid_prev_fps = [fp for fp in prev_fingerprints if fp is not None]

        if len(valid_new_fps) > 0 and len(valid_prev_fps) > 0:
            novel_count = 0
            for new_fp in valid_new_fps:
                # Check if this compound is novel (max similarity < 0.6, i.e., min distance > 0.4)
                max_similarity = 0.0
                for prev_fp in valid_prev_fps:
                    sim = DataStructs.TanimotoSimilarity(new_fp, prev_fp)
                    max_similarity = max(max_similarity, sim)

                if max_similarity < 0.6:  # Distance > 0.4
                    novel_count += 1

            results['batch_novelty_score'] = novel_count / len(valid_new_fps)

    # Round results for cleaner output
    for key in results:
        results[key] = round(results[key], 3)

    return results

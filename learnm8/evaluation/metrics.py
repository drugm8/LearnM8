"""Evaluation metrics for active learning performance."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MSE value
    """
    return mean_squared_error(y_true, y_pred)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R² value
    """
    return r2_score(y_true, y_pred)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value as percentage
    """
    # Handle potential division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = mean_absolute_percentage_error(y_true, y_pred)
        # Convert to percentage and handle inf/nan values
        mape_percentage = mape * 100
        if np.isnan(mape_percentage) or np.isinf(mape_percentage):
            return 0.0
        return mape_percentage


def calculate_spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Spearman rank correlation coefficient.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Spearman correlation coefficient
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    
    try:
        correlation, _ = spearmanr(y_true, y_pred)
        # Handle nan values
        if np.isnan(correlation):
            return 0.0
        return correlation
    except Exception:
        return 0.0


def calculate_average_score(scores: np.ndarray) -> float:
    """
    Calculate average score of compounds.
    
    Args:
        scores: Array of scores
        
    Returns:
        Average score
    """
    return np.mean(scores)


def calculate_enrichment_factor(scores: np.ndarray, labels: np.ndarray, 
                               percentile: float, score_direction: str = 'higher') -> float:
    """
    Calculate Enrichment Factor for virtual screening.
    
    EF = (n_actives_selected / n_selected) / (n_actives_total / n_total)
    
    Args:
        scores: Screening scores
        labels: Binary labels (0/1)
        percentile: Percentile threshold for selection (e.g., 1.0 for top 1%)
        score_direction: 'higher' or 'lower' for score interpretation
        
    Returns:
        Enrichment factor value
    """
    if not 0 < percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    
    # Sort by scores according to direction
    ascending = (score_direction == 'lower')
    sorted_indices = np.argsort(scores) if ascending else np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]
    
    # Calculate selection size
    n_total = len(labels)
    n_selected = max(1, int(n_total * percentile / 100))
    
    # Count actives
    n_actives_total = np.sum(labels)
    n_actives_selected = np.sum(sorted_labels[:n_selected])
    
    # Avoid division by zero
    if n_actives_total == 0 or n_selected == 0:
        return 0.0
    
    # Calculate enrichment factor
    ef = (n_total * n_actives_selected) / (n_actives_total * n_selected)
    
    return round(ef, 3)


def calculate_top_k_overlap(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
                           k: int, target_column: str, score_direction: str = 'higher') -> float:
    """
    Calculate percentage overlap between top-k predicted and true compounds.
    
    Args:
        predictions_df: DataFrame with 'ID' and 'prediction' columns
        ground_truth_df: DataFrame with 'ID' and target_column
        k: Number of top compounds to compare
        target_column: Column name in ground truth
        score_direction: 'higher' or 'lower' for score interpretation
        
    Returns:
        Percentage overlap (0-100)
    """
    # Merge predictions with ground truth
    merged = pd.merge(predictions_df, ground_truth_df[['ID', target_column]], on='ID')
    
    if len(merged) < k:
        from learnm8.utils.logging import get_logger, log_warning
        logger = get_logger()
        log_warning(logger, f"Only {len(merged)} compounds available, using all for top-k calculation")
        k = len(merged)
    
    # Sort by scores
    ascending = (score_direction == 'lower')
    
    # Get top k by predictions
    top_k_predicted = set(merged.nlargest(k, 'prediction', keep='first')['ID'].values) \
                      if not ascending else \
                      set(merged.nsmallest(k, 'prediction', keep='first')['ID'].values)
    
    # Get top k by ground truth
    top_k_true = set(merged.nlargest(k, target_column, keep='first')['ID'].values) \
                 if not ascending else \
                 set(merged.nsmallest(k, target_column, keep='first')['ID'].values)
    
    # Calculate overlap
    overlap_count = len(top_k_predicted & top_k_true)
    overlap_percentage = (overlap_count / k) * 100
    
    return round(overlap_percentage, 2)


def calculate_multiple_top_k_overlaps(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
                                    target_column: str, score_direction: str = 'higher') -> dict:
    """
    Calculate multiple top-K overlaps for different K values.
    
    Args:
        predictions_df: DataFrame with 'ID' and 'prediction' columns
        ground_truth_df: DataFrame with 'ID' and target_column
        target_column: Column name in ground truth
        score_direction: 'higher' or 'lower' for score interpretation
        
    Returns:
        Dictionary with keys: top_100_overlap, top_1000_overlap, top_0_1_percent_overlap, top_1_percent_overlap
    """
    # Merge predictions with ground truth
    merged = pd.merge(predictions_df, ground_truth_df[['ID', target_column]], on='ID')
    n_total = len(merged)
    
    if n_total == 0:
        return {
            'top_100_overlap': 0.0,
            'top_1000_overlap': 0.0,
            'top_0_1_percent_overlap': 0.0,
            'top_1_percent_overlap': 0.0
        }
    
    # Define K values: fixed numbers and percentages
    k_values = {
        'top_100_overlap': min(100, n_total),
        'top_1000_overlap': min(1000, n_total),
        'top_0_1_percent_overlap': max(1, int(n_total * 0.001)),  # 0.1%
        'top_1_percent_overlap': max(1, int(n_total * 0.01))      # 1%
    }
    
    results = {}
    ascending = (score_direction == 'lower')
    
    for key, k in k_values.items():
        if k > n_total:
            k = n_total
            
        # Get top k by predictions
        top_k_predicted = set(merged.nlargest(k, 'prediction', keep='first')['ID'].values) \
                          if not ascending else \
                          set(merged.nsmallest(k, 'prediction', keep='first')['ID'].values)
        
        # Get top k by ground truth
        top_k_true = set(merged.nlargest(k, target_column, keep='first')['ID'].values) \
                     if not ascending else \
                     set(merged.nsmallest(k, target_column, keep='first')['ID'].values)
        
        # Calculate overlap
        overlap_count = len(top_k_predicted & top_k_true)
        overlap_percentage = (overlap_count / k) * 100
        results[key] = round(overlap_percentage, 2)
    
    return results


def calculate_multiple_enrichment_factors(scores: np.ndarray, labels: np.ndarray, 
                                        score_direction: str = 'higher') -> dict:
    """
    Calculate multiple enrichment factors at fixed percentiles: 5%, 1%, 0.5%, 0.1%.
    
    Args:
        scores: Screening scores
        labels: Binary labels (0/1)
        score_direction: 'higher' or 'lower' for score interpretation
        
    Returns:
        Dictionary with keys: ef_5, ef_1, ef_0_5, ef_0_1
    """
    percentiles = [5.0, 1.0, 0.5, 0.1]
    
    results = {}
    for p in percentiles:
        try:
            ef_value = calculate_enrichment_factor(scores, labels, p, score_direction)
            # Create key name (e.g., 5.0 -> ef_5, 0.5 -> ef_0_5)
            key = f"ef_{str(p).replace('.', '_')}"
            results[key] = ef_value
        except Exception as e:
            # Handle edge cases gracefully
            key = f"ef_{str(p).replace('.', '_')}"
            results[key] = 0.0
    
    return results


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


def calculate_molecular_similarity_metrics(newly_selected_df: pd.DataFrame, 
                                         previously_selected_df: pd.DataFrame = None) -> dict:
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
    
    new_smiles = newly_selected_df['SMILES'].tolist()
    new_fingerprints = _generate_fingerprints(new_smiles)
    
    # Calculate intra-batch diversity
    results['intra_batch_diversity'] = _calculate_intra_batch_diversity(new_fingerprints)
    
    # Calculate inter-cycle similarity and novelty if previous compounds exist
    if previously_selected_df is not None and 'SMILES' in previously_selected_df.columns and len(previously_selected_df) > 0:
        prev_smiles = previously_selected_df['SMILES'].tolist()
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


def calculate_ground_truth_enrichment_factors(ground_truth_df: pd.DataFrame, 
                                            target_column: str, 
                                            score_direction: str = 'higher') -> dict:
    """
    Calculate ground truth enrichment factors using true target scores vs Activity labels.
    These remain constant across all cycles since they're based on ground truth data.
    
    Args:
        ground_truth_df: DataFrame with ground truth values including Activity column
        target_column: Column name for target scores to use as screening scores
        score_direction: 'higher' or 'lower' for score interpretation
        
    Returns:
        Dictionary with keys: ground_truth_ef_5, ground_truth_ef_1, ground_truth_ef_0_5, ground_truth_ef_0_1
    """
    # Check if Activity column exists for binary labels
    if 'Activity' not in ground_truth_df.columns:
        # Return empty dict with None values if no Activity column
        return {
            'ground_truth_ef_5': None,
            'ground_truth_ef_1': None, 
            'ground_truth_ef_0_5': None,
            'ground_truth_ef_0_1': None
        }
    
    # Remove any NaN values for calculation
    valid_data = ground_truth_df.dropna(subset=[target_column, 'Activity'])
    
    if len(valid_data) == 0:
        return {
            'ground_truth_ef_5': None,
            'ground_truth_ef_1': None,
            'ground_truth_ef_0_5': None, 
            'ground_truth_ef_0_1': None
        }
    
    # Use target scores as screening scores and Activity as binary labels
    scores = valid_data[target_column].values
    labels = valid_data['Activity'].values
    
    # Calculate enrichment factors at fixed percentiles
    percentiles = [5.0, 1.0, 0.5, 0.1]
    
    results = {}
    for p in percentiles:
        try:
            ef_value = calculate_enrichment_factor(scores, labels, p, score_direction)
            # Create key name with ground_truth prefix
            key = f"ground_truth_ef_{str(p).replace('.', '_')}"
            results[key] = ef_value
        except Exception as e:
            # Handle edge cases gracefully
            key = f"ground_truth_ef_{str(p).replace('.', '_')}"
            results[key] = None
    
    return results
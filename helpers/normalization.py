# Example usage

import numpy as np
import pandas as pd

RESCORING_FUNCTIONS = {
    "AAScore": {"column_name": "AAScore", "best_value": "min", "score_range": (100, -100)},
    "AD4": {"column_name": "AD4", "best_value": "min", "score_range": (1000, -100)},
    "CHEMPLP": {"column_name": "CHEMPLP", "best_value": "min", "score_range": (200, -200)},
    "ConvexPLR": {"column_name": "ConvexPLR", "best_value": "max", "score_range": (-10, 10)},
    "GNINA-Affinity": {"column_name": "GNINA-Affinity", "best_value": "min", "score_range": (100, -100)},
    "CNN-Score": {"column_name": "CNN-Score", "best_value": "max", "score_range": (0, 1)},
    "CNN-Affinity": {"column_name": "CNN-Affinity", "best_value": "max", "score_range": (0, 20)},
    "GenScore-scoring": {"column_name": "GenScore-scoring", "best_value": "max", "score_range": (0, 100)},
    "GenScore-docking": {"column_name": "GenScore-docking", "best_value": "max", "score_range": (0, 100)},
    "GenScore-balanced": {"column_name": "GenScore-balanced", "best_value": "max", "score_range": (0, 100)},
    "KORP-PL": {"column_name": "KORP-PL", "best_value": "min", "score_range": (200, -500)},
    "LinF9": {"column_name": "LinF9", "best_value": "min", "score_range": (50, -50)},
    "NNScore": {"column_name": "NNScore", "best_value": "max", "score_range": (0, 20)},
    "PLECScore": {"column_name": "PLECScore", "best_value": "max", "score_range": (0, 20)},
    "PLP": {"column_name": "PLP", "best_value": "min", "score_range": (200, -200)},
    "RFScoreVS": {"column_name": "RFScoreVS", "best_value": "max", "score_range": (5, 10)},
    "RTMScore": {"column_name": "RTMScore", "best_value": "max", "score_range": (0, 100)},
    "SCORCH": {"column_name": "SCORCH", "best_value": "max", "score_range": (0, 1)},
    "Vinardo": {"column_name": "Vinardo", "best_value": "min", "score_range": (200, -20)}
}

scoring_data = pd.DataFrame() #!! this is empty thats why errors are produced

scoring_function = "KORP-PL"

scores = scoring_data[scoring_function]

scoring_info = RESCORING_FUNCTIONS[scoring_function]

def normalize_scores(scores: np.ndarray, scoring_info: dict) -> np.ndarray:
    """
    Normalize scores to [0,1] range where 1 is always best.
    
    Parameters
    ----------
    scores : np.ndarray
        Raw scores to normalize
    scoring_info : Dict
        Scoring function information from RESCORING_FUNCTIONS
        
    Returns:
    -------
    np.ndarray
        Normalized scores in [0,1] range
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.full_like(scores, 0.5)
        
    normalized = (scores - min_score) / (max_score - min_score)
    if scoring_info["best_value"] == "min":
        normalized = 1 - normalized
    return normalized

#"gedockte daten aufklambüsern, dann for i in column normailze und dann wieder zusammenfügen"

normalized_scores = normalize_scores(scores, scoring_info)
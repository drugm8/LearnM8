"""Modular evaluation metrics for LearnM8 active learning.

This package organizes metrics into logical categories:
- performance: ML model performance metrics
- enrichment: Virtual screening enrichment metrics
- similarity: Diversity metrics (the inverse of pairwise similarity);
  filename retained for git-history continuity.
- diagnostics: Cycle diagnostic metrics — selected_percentile and
  prediction_entropy (feature 014).
"""

from .diagnostics import (
    compute_prediction_entropy,
    compute_selected_percentile,
)
from .enrichment import (
    calculate_average_score_ratio,
    calculate_batch_average_score_ratio,
    calculate_batch_enrichment_factor,
    calculate_batch_hit_rate,
    calculate_cumulative_enrichment_factor,
    # Existing functions (kept for compatibility)
    calculate_enrichment_factor,
    calculate_ground_truth_enrichment_factors,
    calculate_multiple_enrichment_factors,
    # Discovery metrics (Category A)
    calculate_multiple_top_k_discovery_rates,
    calculate_multiple_top_k_overlaps,
    calculate_multiple_unlabeled_enrichment_factors,
    # Ranking metrics (Category B) - unlabeled only
    calculate_multiple_unlabeled_top_k_overlaps,
    calculate_top_k_overlap,
    calculate_unlabeled_ranking_correlation,
)
from .performance import (
    calculate_average_score,
    calculate_mape,
    calculate_spearman_correlation,
)
from .similarity import (
    DIVERSITY_KEYS,
    RunCache,
    compute_diversity_metrics,
)

__all__ = [
    'DIVERSITY_KEYS',
    'RunCache',
    'calculate_average_score',
    'calculate_average_score_ratio',
    'calculate_batch_average_score_ratio',
    'calculate_batch_enrichment_factor',
    'calculate_batch_hit_rate',
    'calculate_cumulative_enrichment_factor',
    'calculate_enrichment_factor',
    'calculate_ground_truth_enrichment_factors',
    'calculate_mape',
    'calculate_multiple_enrichment_factors',
    'calculate_multiple_top_k_discovery_rates',
    'calculate_multiple_top_k_overlaps',
    'calculate_multiple_unlabeled_enrichment_factors',
    'calculate_multiple_unlabeled_top_k_overlaps',
    'calculate_spearman_correlation',
    'calculate_top_k_overlap',
    'calculate_unlabeled_ranking_correlation',
    'compute_diversity_metrics',
    'compute_prediction_entropy',
    'compute_selected_percentile',
]

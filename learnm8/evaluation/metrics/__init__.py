"""Modular evaluation metrics for LearnM8 active learning.

This package organizes metrics into logical categories:
- performance: ML model performance metrics
- enrichment: Virtual screening enrichment metrics
- similarity: Molecular similarity and diversity metrics
"""

# Import all metric functions for easy access
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
from .similarity import calculate_molecular_similarity_metrics

__all__ = [
    # Performance metrics
    'calculate_mape',
    'calculate_spearman_correlation',
    'calculate_average_score',

    # Discovery metrics
    'calculate_multiple_top_k_discovery_rates',
    'calculate_cumulative_enrichment_factor',
    'calculate_batch_hit_rate',
    'calculate_batch_enrichment_factor',
    'calculate_average_score_ratio',
    'calculate_batch_average_score_ratio',

    # Ranking metrics
    'calculate_multiple_unlabeled_top_k_overlaps',
    'calculate_multiple_unlabeled_enrichment_factors',
    'calculate_unlabeled_ranking_correlation',

    # Existing enrichment metrics
    'calculate_enrichment_factor',
    'calculate_multiple_enrichment_factors',
    'calculate_top_k_overlap',
    'calculate_multiple_top_k_overlaps',
    'calculate_ground_truth_enrichment_factors',

    # Similarity metrics
    'calculate_molecular_similarity_metrics'
]

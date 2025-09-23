"""Modular evaluation metrics for LearnM8 active learning.

This package organizes metrics into logical categories:
- performance: ML model performance metrics
- enrichment: Virtual screening enrichment metrics
- similarity: Molecular similarity and diversity metrics
"""

# Import all metric functions for easy access
from .performance import (
    calculate_mape,
    calculate_spearman_correlation,
    calculate_average_score
)

from .enrichment import (
    calculate_enrichment_factor,
    calculate_multiple_enrichment_factors,
    calculate_top_k_overlap,
    calculate_multiple_top_k_overlaps,
    calculate_ground_truth_enrichment_factors
)

from .similarity import (
    calculate_molecular_similarity_metrics
)

__all__ = [
    # Performance metrics
    'calculate_mape',
    'calculate_spearman_correlation',
    'calculate_average_score',

    # Enrichment metrics
    'calculate_enrichment_factor',
    'calculate_multiple_enrichment_factors',
    'calculate_top_k_overlap',
    'calculate_multiple_top_k_overlaps',
    'calculate_ground_truth_enrichment_factors',

    # Similarity metrics
    'calculate_molecular_similarity_metrics'
]
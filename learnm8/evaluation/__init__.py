"""Comprehensive adaptive evaluation for LearnM8 active learning.

This module provides a streamlined evaluation system that intelligently adapts
metrics based on mode (benchmark vs run) and data availability while including
all essential model performance metrics and molecular similarity analysis.
"""

# Import main evaluation functions from core module
from .core import evaluate_cycle, export_metrics_csv, format_progress_output

# Import specialized metric functions from modular metrics package
from .metrics import (
    DIVERSITY_KEYS,
    RunCache,
    calculate_average_score,
    calculate_enrichment_factor,
    calculate_multiple_enrichment_factors,
    calculate_multiple_top_k_overlaps,
    calculate_spearman_correlation,
    calculate_top_k_overlap,
    compute_diversity_metrics,
)

# Public API exports
__all__ = [
    'DIVERSITY_KEYS',
    'RunCache',
    'calculate_average_score',
    'calculate_enrichment_factor',
    'calculate_multiple_enrichment_factors',
    'calculate_multiple_top_k_overlaps',
    'calculate_spearman_correlation',
    'calculate_top_k_overlap',
    'compute_diversity_metrics',
    # Main evaluation functions
    'evaluate_cycle',
    'export_metrics_csv',
    'format_progress_output'
]

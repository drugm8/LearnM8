"""Comprehensive adaptive evaluation for LearnM8 active learning.

This module provides a streamlined evaluation system that intelligently adapts
metrics based on mode (benchmark vs run) and data availability while including
all essential model performance metrics and molecular similarity analysis.
"""

# Import main evaluation functions from core module
from .core import (
    evaluate_cycle,
    format_progress_output,
    export_metrics_csv
)

# Import specialized metric functions from metrics module
from .metrics import (
    calculate_spearman_correlation,
    calculate_average_score,
    calculate_top_k_overlap,
    calculate_multiple_top_k_overlaps,
    calculate_enrichment_factor,
    calculate_multiple_enrichment_factors,
    calculate_molecular_similarity_metrics,
    calculate_mape
)

# Public API exports for backward compatibility
__all__ = [
    # Main evaluation functions
    'evaluate_cycle',
    'format_progress_output', 
    'export_metrics_csv',
    
    # Specialized metric functions (sklearn basics not exported)
    'calculate_spearman_correlation',
    'calculate_average_score',
    'calculate_top_k_overlap',
    'calculate_multiple_top_k_overlaps',
    'calculate_enrichment_factor',
    'calculate_multiple_enrichment_factors',
    'calculate_molecular_similarity_metrics',
    'calculate_mape'
]
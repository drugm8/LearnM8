"""Tests for LearnM8 enrichment metrics.

Tests for virtual screening enrichment metrics using real molecular data.
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from learnm8.evaluation.metrics.enrichment import (
    calculate_top_k_overlap,
    calculate_enrichment_factor
)


class TestEnrichmentMetrics:
    """Test enrichment and virtual screening metrics."""

    def test_top_k_overlap(self):
        """Test top-K overlap calculation."""
        # Test data - create predictions DataFrame
        predictions_df = pd.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5'],
            'prediction': [9.5, 8.5, 7.5, 6.5, 5.5]  # mol_1, mol_2, mol_3 should be top 3
        })

        ground_truth_data = pd.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_6', 'mol_7', 'mol_8'],
            'Activity': [10, 9, 8, 7, 6, 5]  # mol_1, mol_2, mol_3 are top 3
        })

        # Test k=3
        overlap = calculate_top_k_overlap(predictions_df, ground_truth_data, k=3, target_column='Activity')
        assert overlap == 100.0  # All top 3 are in selected (100% overlap)

        # Test k=5 - but we only have 3 compounds that match, so 100% overlap on those 3
        # (The function only considers compounds that exist in both datasets)
        overlap = calculate_top_k_overlap(predictions_df, ground_truth_data, k=5, target_column='Activity')
        assert overlap == 100.0  # All 3 available compounds overlap (100%)

    def test_enrichment_factor(self):
        """Test enrichment factor calculation."""
        # Create test data with scores and binary labels
        scores = np.array([10, 9, 8, 7, 6, 5, 4, 3])
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # Top 4 are active

        # Calculate enrichment factor at 50% (top 4 compounds)
        # All 4 selected compounds are active (4/4 = 100%)
        # Random selection would give 50% (4/8 actives total)
        # EF = 100% / 50% = 2.0
        enrichment = calculate_enrichment_factor(scores, labels, percentile=50.0)
        assert_allclose(enrichment, 2.0, rtol=1e-2)

        # Test at 25% (top 2 compounds)
        # 2 out of 2 selected are active (100%)
        # Random would give 50%
        # EF = 100% / 50% = 2.0
        enrichment = calculate_enrichment_factor(scores, labels, percentile=25.0)
        assert_allclose(enrichment, 2.0, rtol=1e-2)
"""Tests for LearnM8 similarity metrics.

Tests for molecular similarity and diversity metrics using real molecular data.
Note: These tests require RDKit and may be slow due to fingerprint calculations.
"""

import pytest
import polars as pl

from learnm8.evaluation.metrics.similarity import (
    calculate_molecular_similarity_metrics
)


class TestSimilarityMetrics:
    """Test molecular similarity and diversity metrics."""

    def test_molecular_similarity_metrics_basic(self):
        """Test basic molecular similarity metrics calculation."""
        # Create test data with real SMILES
        newly_selected = pl.DataFrame({
            'ID': ['mol_1', 'mol_2'],
            'SMILES': ['CCO', 'CCC']  # Ethanol and propane
        })

        # Test without previous compounds
        result = calculate_molecular_similarity_metrics(newly_selected)

        # Should return structure with expected keys
        expected_keys = {'intra_batch_diversity', 'inter_cycle_similarity', 'batch_novelty_score'}
        assert set(result.keys()) == expected_keys

        # With only 2 compounds, diversity should be calculated
        assert 0.0 <= result['intra_batch_diversity'] <= 1.0
        assert result['inter_cycle_similarity'] == 0.0  # No previous compounds
        assert result['batch_novelty_score'] == 0.0  # No previous compounds

    def test_molecular_similarity_with_previous(self):
        """Test molecular similarity with previous compound data."""
        newly_selected = pl.DataFrame({
            'ID': ['mol_3', 'mol_4'],
            'SMILES': ['CCCO', 'CCCC']  # Propanol and butane
        })

        previously_selected = pl.DataFrame({
            'ID': ['mol_1', 'mol_2'],
            'SMILES': ['CCO', 'CCC']  # Ethanol and propane
        })

        result = calculate_molecular_similarity_metrics(newly_selected, previously_selected)

        # Should include inter-cycle and novelty calculations
        assert 0.0 <= result['intra_batch_diversity'] <= 1.0
        assert 0.0 <= result['inter_cycle_similarity'] <= 1.0
        assert 0.0 <= result['batch_novelty_score'] <= 1.0

    def test_molecular_similarity_no_smiles(self):
        """Test handling when SMILES column is missing."""
        compounds_no_smiles = pl.DataFrame({
            'ID': ['mol_1', 'mol_2'],
            'other_column': ['A', 'B']
        })

        result = calculate_molecular_similarity_metrics(compounds_no_smiles)

        # Should return zero values when no SMILES
        assert result['intra_batch_diversity'] == 0.0
        assert result['inter_cycle_similarity'] == 0.0
        assert result['batch_novelty_score'] == 0.0
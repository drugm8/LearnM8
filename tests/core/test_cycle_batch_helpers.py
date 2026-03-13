"""Unit tests for batch prediction helper functions in cycle.py."""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from learnm8.core.cycle import (
    _calculate_optimal_batch_size,
    _chunk_dataframe,
    _predict_chunk
)


@pytest.mark.unit
class TestCalculateOptimalBatchSize:
    """Test _calculate_optimal_batch_size() helper function."""

    def test_small_dataset_returns_full_size(self):
        """Small datasets (≤100k) should return n_compounds."""
        batch_size = _calculate_optimal_batch_size(50000, 'morgan', 4.0)
        assert batch_size == 50000

    def test_large_dataset_returns_calculated_size(self):
        """Large datasets (>100k) should return calculated batch size."""
        batch_size = _calculate_optimal_batch_size(500000, 'morgan', 4.0)
        assert 1000 <= batch_size <= 50000

    def test_different_featurizers(self):
        """Different featurizers should affect batch size calculation."""
        batch_maccs = _calculate_optimal_batch_size(500000, 'maccs', 0.1)
        batch_morgan = _calculate_optimal_batch_size(500000, 'morgan', 0.1)
        batch_descriptors = _calculate_optimal_batch_size(500000, 'descriptors', 0.1)

        assert batch_maccs > batch_descriptors
        assert batch_descriptors > batch_morgan

    def test_unknown_featurizer_uses_default(self):
        """Unknown featurizer should use default feature size."""
        batch_size = _calculate_optimal_batch_size(500000, 'unknown', 4.0)
        assert 1000 <= batch_size <= 50000

    def test_minimum_batch_size_enforced(self):
        """Batch size should never be less than 1000."""
        batch_size = _calculate_optimal_batch_size(100000, 'descriptors', 0.1)
        assert batch_size >= 1000

    def test_maximum_batch_size_enforced(self):
        """Batch size should never exceed 50000."""
        batch_size = _calculate_optimal_batch_size(100000, 'maccs', 100.0)
        assert batch_size <= 50000


@pytest.mark.unit
class TestChunkDataFrame:
    """Test _chunk_dataframe() helper function."""

    def test_exact_division(self):
        """DataFrame with size divisible by chunk_size."""
        df = pl.DataFrame({'ID': range(100), 'SMILES': ['C'] * 100})
        chunks = list(_chunk_dataframe(df, 25))

        assert len(chunks) == 4
        assert all(len(chunk) == 25 for chunk in chunks)

    def test_uneven_division(self):
        """DataFrame with size not divisible by chunk_size."""
        df = pl.DataFrame({'ID': range(100), 'SMILES': ['C'] * 100})
        chunks = list(_chunk_dataframe(df, 30))

        assert len(chunks) == 4
        assert len(chunks[0]) == 30
        assert len(chunks[1]) == 30
        assert len(chunks[2]) == 30
        assert len(chunks[3]) == 10

    def test_single_chunk(self):
        """Chunk size larger than DataFrame should yield single chunk."""
        df = pl.DataFrame({'ID': range(50), 'SMILES': ['C'] * 50})
        chunks = list(_chunk_dataframe(df, 100))

        assert len(chunks) == 1
        assert len(chunks[0]) == 50

    def test_empty_dataframe(self):
        """Empty DataFrame should yield no chunks."""
        df = pl.DataFrame({'ID': [], 'SMILES': []})
        chunks = list(_chunk_dataframe(df, 10))

        assert len(chunks) == 0


@pytest.mark.unit
class TestPredictChunk:
    """Test _predict_chunk() helper function."""

    def test_feature_based_learner(self, tmp_path, sample_compounds):
        """Feature-based learner requires featurizer and works correctly."""
        chunk_df = sample_compounds.head(5)

        mock_learner = Mock()
        mock_learner.requires_smiles.return_value = False
        mock_learner.get_name.return_value = 'MockLearner'
        mock_learner.predict.return_value = (
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            None
        )

        with patch('learnm8.core.cycle.extract_features') as mock_extract:
            mock_extract.return_value = np.random.rand(5, 2048)

            predictions, uncertainties = _predict_chunk(
                chunk_df, mock_learner, 'morgan', tmp_path, show_progress=False
            )

            assert predictions.shape == (5,)
            assert uncertainties is None
            mock_extract.assert_called_once()
            mock_learner.predict.assert_called_once()

    def test_feature_based_learner_requires_featurizer(self, tmp_path, sample_compounds):
        """Feature-based learner should raise error if featurizer is None."""
        chunk_df = sample_compounds.head(5)

        mock_learner = Mock()
        mock_learner.requires_smiles.return_value = False
        mock_learner.get_name.return_value = 'MockLearner'

        with pytest.raises(ValueError, match="featurizer is required"):
            _predict_chunk(chunk_df, mock_learner, None, tmp_path, show_progress=False)

    def test_smiles_aware_learner_without_features(self, tmp_path, sample_compounds):
        """SMILES-aware learner works without features (pure graph-based)."""
        chunk_df = sample_compounds.head(5)

        mock_learner = Mock()
        mock_learner.requires_smiles.return_value = True
        mock_learner.predict.return_value = (
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        )

        predictions, uncertainties = _predict_chunk(
            chunk_df, mock_learner, None, tmp_path, show_progress=False
        )

        assert predictions.shape == (5,)
        assert uncertainties.shape == (5,)
        mock_learner.predict.assert_called_once()
        call_kwargs = mock_learner.predict.call_args[1]
        assert call_kwargs['features'] is None
        assert len(call_kwargs['smiles']) == 5

    def test_smiles_aware_learner_with_features(self, tmp_path, sample_compounds):
        """SMILES-aware learner works with features (hybrid mode)."""
        chunk_df = sample_compounds.head(5)

        mock_learner = Mock()
        mock_learner.requires_smiles.return_value = True
        mock_learner.predict.return_value = (
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            None
        )

        with patch('learnm8.core.cycle.extract_features') as mock_extract:
            mock_extract.return_value = np.random.rand(5, 2048)

            predictions, uncertainties = _predict_chunk(
                chunk_df, mock_learner, 'morgan', tmp_path, show_progress=False
            )

            assert predictions.shape == (5,)
            assert uncertainties is None
            mock_extract.assert_called_once()
            mock_learner.predict.assert_called_once()
            call_kwargs = mock_learner.predict.call_args[1]
            assert call_kwargs['features'] is not None
            assert len(call_kwargs['smiles']) == 5

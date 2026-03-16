import numpy as np
import polars as pl
import pytest

from learnm8.acquisition import RandomAcquisition


@pytest.mark.unit
class TestRandomAcquisition:
    """Test RandomAcquisition functionality."""

    def test_random_selection_returns_requested_valid_compounds(self, small_real_compounds):
        """Test basic random acquisition."""
        compounds = small_real_compounds.clone()
        acq = RandomAcquisition(random_state=42)
        selected = acq.select(compounds, n_select=8)

        assert len(selected) == 8
        assert 'acquisition_score' in selected.columns

        # All compounds should have been valid selections
        compound_ids = compounds.get_column('ID').to_list()
        selected_ids = selected.get_column('ID').to_list()
        assert all(id in compound_ids for id in selected_ids)

    def test_random_reproducibility(self, small_real_compounds):
        """Test random acquisition reproducibility with seed."""
        compounds = small_real_compounds.clone()
        acq1 = RandomAcquisition(random_state=42)
        acq2 = RandomAcquisition(random_state=42)

        selected1 = acq1.select(compounds, n_select=10)
        selected2 = acq2.select(compounds, n_select=10)

        # Should select identical compounds with same seed
        assert selected1.get_column('ID').to_list() == selected2.get_column('ID').to_list()

    def test_random_different_seeds(self, small_real_compounds):
        """Test that different seeds produce different selections."""
        compounds = small_real_compounds.clone()
        acq1 = RandomAcquisition(random_state=42)
        acq2 = RandomAcquisition(random_state=123)

        selected1 = acq1.select(compounds, n_select=15)
        selected2 = acq2.select(compounds, n_select=15)

        # Should select different compounds with different seeds
        assert selected1.get_column('ID').to_list() != selected2.get_column('ID').to_list()

    def test_different_random_seeds_cover_overlapping_but_nonidentical_regions(self, medium_real_compounds):
        """Test random selection covers dataset reasonably."""
        compounds = medium_real_compounds.clone()

        if len(compounds) < 50:
            pytest.skip("Insufficient compounds for coverage test")

        # Use different random states for different selections
        acq1 = RandomAcquisition(random_state=42)
        acq2 = RandomAcquisition(random_state=123)

        # Multiple selections should cover different parts of dataset
        selected1 = acq1.select(compounds, n_select=20)
        selected2 = acq2.select(compounds, n_select=20)

        # Should have some overlap but not complete overlap
        ids1 = set(selected1.get_column('ID').to_list())
        ids2 = set(selected2.get_column('ID').to_list())
        overlap = len(ids1 & ids2)
        assert 0 < overlap < 20  # Some but not complete overlap

    def test_random_returns_all_compounds_when_requested_batch_exceeds_pool(self, small_real_compounds):
        compounds = small_real_compounds.head(4).clone()

        selected = RandomAcquisition(random_state=42).select(compounds, n_select=10)

        assert len(selected) == 4
        assert set(selected.get_column('ID').to_list()) == set(compounds.get_column('ID').to_list())

    def test_random_rejects_empty_pool(self, empty_compounds):
        with pytest.raises(ValueError, match='compounds DataFrame is empty'):
            RandomAcquisition(random_state=42).select(empty_compounds, n_select=1)

    def test_random_rejects_non_positive_batch_size(self, small_real_compounds):
        with pytest.raises(ValueError, match='n_select must be positive'):
            RandomAcquisition(random_state=42).select(small_real_compounds.head(4).clone(), n_select=0)

    def test_random_get_name_includes_seed(self):
        assert RandomAcquisition(random_state=7).get_name() == 'Random(seed=7)'


@pytest.mark.unit
class TestBasicAcquisitionIntegration:
    """Integration tests for basic acquisition functions."""

    def test_acquisition_with_real_molecular_workflow(self, medium_real_compounds):
        """Test acquisition functions in realistic molecular workflow."""
        compounds = medium_real_compounds.clone()

        if len(compounds) < 20:
            pytest.skip("Insufficient compounds for workflow test")

        # Simulate predictions from different models
        np.random.seed(42)
        compounds = compounds.with_columns(
            (pl.col('Activity') + pl.Series('noise', np.random.normal(0, 2, len(compounds)))).alias('prediction')
        )

        random_acq = RandomAcquisition(random_state=42)

        n_select = 10
        random_selected = random_acq.select(compounds, n_select=n_select)

        assert len(random_selected) == n_select

        # Selection should still return valid compound IDs for a real Polars workflow.
        all_ids = set(compounds.get_column('ID').to_list())
        assert set(random_selected.get_column('ID').to_list()).issubset(all_ids)

    def test_acquisition_with_diverse_molecular_targets(self, diverse_real_compounds):
        """Test acquisition with multi-target molecular data."""
        compounds = diverse_real_compounds.clone()

        if len(compounds) == 0:
            pytest.skip("No diverse molecular data available")

        # Add target-aware predictions
        compounds = compounds.with_columns(
            (pl.col('Activity') + pl.Series('noise', np.random.normal(0, 1, len(compounds)))).alias('prediction')
        )

        selected = RandomAcquisition(random_state=42).select(compounds, n_select=min(15, len(compounds)))

        assert len(selected) <= len(compounds)
        if 'Target' in compounds.columns:
            selected_targets = set(selected.get_column('Target').to_list())
            assert set(selected.get_column('Target').to_list()).issubset(
                set(compounds.get_column('Target').to_list())
            )
            assert len(selected_targets) > 0

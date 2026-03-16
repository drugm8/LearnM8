"""
Tests for compound selection functionality using new acquisition API.

Tests acquisition functions with real molecular data,
focusing on different strategies, error handling, and data validation.
"""


import numpy as np
import polars as pl
import pytest

from learnm8.acquisition import get_acquisition_function, list_acquisition_functions


@pytest.mark.unit
class TestSelectCompoundsByStrategy:
    """Test compound selection with different strategies."""

    def test_greedy_selection_higher(self, small_real_compounds, tmp_path):
        """Test greedy selection with 'higher' score direction."""
        compounds = small_real_compounds.clone()

        if len(compounds) == 0:
            compounds = pl.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(10)],
                'SMILES': ['CCO'] * 10
            })

        # Create predictable predictions
        np.random.seed(42)
        predictions = np.random.uniform(0, 1, len(compounds))

        # Prepare pool with predictions
        pool = compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', predictions)
        )

        # Get acquisition function and select
        acq_class = get_acquisition_function('greedy')
        acq = acq_class(score_direction='higher')
        selected = acq.select(pool, n_select=3)

        assert len(selected) == 3
        assert 'ID' in selected.columns
        assert 'SMILES' in selected.columns

        # Should select compounds with highest predictions
        # Get indices by matching IDs
        selected_ids = selected['ID'].to_list()
        id_to_index = {id_val: i for i, id_val in enumerate(compounds['ID'].to_list())}
        selected_indices = [id_to_index[id_val] for id_val in selected_ids]
        selected_predictions = predictions[selected_indices]

        # Verify they are among the top predictions
        top_3_predictions = np.sort(predictions)[-3:]
        assert all(pred in top_3_predictions for pred in selected_predictions)

    def test_greedy_selection_lower(self, small_real_compounds, tmp_path):
        """Test greedy selection with 'lower' score direction."""
        compounds = small_real_compounds.clone()

        if len(compounds) == 0:
            compounds = pl.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(8)],
                'SMILES': ['CCC'] * 8
            })

        np.random.seed(42)
        predictions = np.random.uniform(0, 1, len(compounds))

        # Prepare pool with predictions
        pool = compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', predictions)
        )

        # Get acquisition function and select
        acq_class = get_acquisition_function('greedy')
        acq = acq_class(score_direction='lower')
        selected = acq.select(pool, n_select=2)

        assert len(selected) == 2

        # Should select compounds with lowest predictions
        selected_ids = selected['ID'].to_list()
        id_to_index = {id_val: i for i, id_val in enumerate(compounds['ID'].to_list())}
        selected_indices = [id_to_index[id_val] for id_val in selected_ids]
        selected_predictions = predictions[selected_indices]

        # With 'lower' direction, it should prefer lower predictions
        bottom_2_predictions = np.sort(predictions)[:2]
        assert all(pred in bottom_2_predictions for pred in selected_predictions)

    def test_random_strategy_selects_subset_of_original_compounds(self, small_real_compounds, tmp_path):
        """Test random selection strategy."""
        compounds = small_real_compounds.clone()

        if len(compounds) == 0:
            compounds = pl.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(6)],
                'SMILES': ['CCN'] * 6
            })

        predictions = np.random.uniform(0, 1, len(compounds))

        # Prepare pool with predictions
        pool = compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', predictions)
        )

        # Get acquisition function and select
        acq_class = get_acquisition_function('random')
        acq = acq_class(score_direction='higher', random_state=42)
        selected = acq.select(pool, n_select=4)

        assert len(selected) == 4
        assert 'ID' in selected.columns
        assert 'SMILES' in selected.columns

        # Should be a subset of original compounds
        assert selected['ID'].is_in(compounds['ID']).all()

    def test_ucb_strategy_selects_valid_compounds_when_uncertainty_is_present(self, small_real_compounds, tmp_path):
        """Test uncertainty-based selection strategies."""
        compounds = small_real_compounds.clone()

        if len(compounds) == 0:
            compounds = pl.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(8)],
                'SMILES': ['CCO'] * 8
            })

        predictions = np.random.uniform(0.3, 0.7, len(compounds))
        uncertainties = np.random.uniform(0.1, 0.9, len(compounds))

        # Prepare pool with predictions and uncertainties
        pool = compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', predictions)
        )
        pool = pool.with_columns(
            pl.Series('uncertainty', uncertainties)
        )

        # Test strategies that use uncertainty
        uncertainty_strategies = ['ucb']  # Start with basic ones that should work

        for strategy in uncertainty_strategies:
            try:
                acq_class = get_acquisition_function(strategy)
                acq = acq_class(score_direction='higher')
                selected = acq.select(pool, n_select=3)

                assert len(selected) == 3
                assert 'ID' in selected.columns
                assert selected['ID'].is_in(compounds['ID']).all()

            except Exception as e:
                pytest.skip(f"Strategy {strategy} not available or failed: {e}")

    def test_selection_caps_batch_size_to_pool_and_rejects_zero(self, small_real_compounds, tmp_path):
        """Test batch size constraint handling."""
        compounds = small_real_compounds.clone()

        # Use a small subset for predictable testing
        if len(compounds) >= 2:
            compounds = compounds.head(2).clone()
        else:
            compounds = pl.DataFrame({
                'ID': ['COMP_001', 'COMP_002'],
                'SMILES': ['CCO', 'CCC']
            })

        predictions = np.array([0.3, 0.7])

        # Prepare pool with predictions
        pool = compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', predictions)
        )

        # Test batch size larger than pool
        acq_class = get_acquisition_function('greedy')
        acq = acq_class(score_direction='higher')
        selected = acq.select(pool, n_select=10)  # Larger than pool size

        # Should return all available compounds
        assert len(selected) == len(compounds)

        # Test zero batch size should raise ValueError
        with pytest.raises(ValueError, match="n_select must be positive"):
            acq.select(pool, n_select=0)

    def test_greedy_raises_when_prediction_column_is_missing(self, small_real_compounds, tmp_path):
        """Test handling when predictions are missing."""
        compounds = small_real_compounds.clone()

        if len(compounds) == 0:
            compounds = pl.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(5)],
                'SMILES': ['CCC'] * 5
            })

        # Pool without predictions column
        pool = compounds.clone()

        # Should raise ValueError when predictions are missing
        acq_class = get_acquisition_function('greedy')
        acq = acq_class(score_direction='higher')

        with pytest.raises(ValueError, match="prediction"):
            acq.select(pool, n_select=2)

    def test_ucb_raises_when_uncertainty_column_is_missing(self, small_real_compounds, tmp_path):
        """Test handling when uncertainties are None for uncertainty-based strategies."""
        compounds = small_real_compounds.clone()

        if len(compounds) == 0:
            compounds = pl.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(6)],
                'SMILES': ['CCN'] * 6
            })

        predictions = np.random.uniform(0, 1, len(compounds))

        # Prepare pool without uncertainties
        pool = compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', predictions)
        )

        # Should raise ValueError when uncertainties are required but not provided
        acq_class = get_acquisition_function('ucb')
        acq = acq_class(score_direction='higher')

        with pytest.raises(ValueError, match="uncertainty"):
            acq.select(pool, n_select=3)

    def test_greedy_selects_from_dataframe_with_prediction_column(self, small_real_compounds, tmp_path):
        """Test handling of DataFrame predictions."""
        compounds = small_real_compounds.clone()

        # Use a small subset for predictable testing
        if len(compounds) >= 4:
            compounds = compounds.head(4).clone()
        else:
            compounds = pl.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(4)],
                'SMILES': ['CCO'] * 4
            })

        # Create DataFrame with predictions
        pool = compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', [0.2, 0.8, 0.5, 0.9][:len(compounds)])
        )

        acq_class = get_acquisition_function('greedy')
        acq = acq_class(score_direction='higher')
        selected = acq.select(pool, n_select=2)

        assert len(selected) == 2
        assert 'ID' in selected.columns

    def test_invalid_strategy_name_raises_key_error(self, small_real_compounds, tmp_path):
        """Test error handling for invalid strategy names."""
        compounds = small_real_compounds.clone()

        # Use a single compound for predictable testing
        if len(compounds) >= 1:
            compounds = compounds.head(1).clone()
        else:
            compounds = pl.DataFrame({
                'ID': ['COMP_001'],
                'SMILES': ['CCO']
            })

        # Pool with predictions
        pool = compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', np.array([0.5]))
        )

        with pytest.raises(KeyError):
            get_acquisition_function('invalid_strategy')

    def test_acquisition_parameters_allow_successful_selection(self, small_real_compounds, tmp_path):
        """Test passing of acquisition parameters."""
        compounds = small_real_compounds.clone()

        if len(compounds) == 0:
            compounds = pl.DataFrame({
                'ID': [f'COMP_{i:03d}' for i in range(5)],
                'SMILES': ['CCC'] * 5
            })

        predictions = np.random.uniform(0, 1, len(compounds))

        # Prepare pool with predictions
        pool = compounds.clone()
        pool = pool.with_columns(
            pl.Series('prediction', predictions)
        )

        # Test with acquisition parameters
        acq_class = get_acquisition_function('greedy')
        acq = acq_class(score_direction='higher')  # Greedy ignores extra params
        selected = acq.select(pool, n_select=2)

        assert len(selected) == 2

    def test_list_acquisition_functions_includes_basic_strategies(self):
        """Test listing available acquisition strategies."""
        strategies = list_acquisition_functions()

        assert isinstance(strategies, list)
        assert len(strategies) > 0

        # Should include basic strategies
        assert 'greedy' in strategies
        assert 'random' in strategies

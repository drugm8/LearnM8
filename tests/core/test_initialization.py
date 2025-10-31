import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from learnm8.core.initialization import initialize_master_dataframe_empty
from learnm8.core.data_structures import STATUS_LABELED, STATUS_UNLABELED


class TestInitializeMasterDataframeEmpty:

    def test_basic_initialization(self, sample_compounds):
        master_df = initialize_master_dataframe_empty(
            valid_compounds=sample_compounds,
            target_col='Activity'
        )

        assert len(master_df) == len(sample_compounds)
        assert 'ID' in master_df.columns
        assert 'SMILES' in master_df.columns
        assert 'status' in master_df.columns
        assert 'labeled_cycle' in master_df.columns
        assert 'selected_cycle' in master_df.columns
        assert 'pruned_cycle' in master_df.columns
        assert 'Activity' in master_df.columns

    def test_uses_actual_target_column_name(self, sample_compounds):
        master_df = initialize_master_dataframe_empty(
            valid_compounds=sample_compounds,
            target_col='pIC50'
        )

        assert 'pIC50' in master_df.columns
        assert 'target_value' not in master_df.columns

    def test_all_compounds_unlabeled(self, sample_compounds):
        master_df = initialize_master_dataframe_empty(
            valid_compounds=sample_compounds,
            target_col='Activity'
        )

        assert (master_df['status'] == STATUS_UNLABELED).all()
        assert pd.isna(master_df['labeled_cycle']).all()
        assert pd.isna(master_df['selected_cycle']).all()
        assert pd.isna(master_df['pruned_cycle']).all()
        assert pd.isna(master_df['Activity']).all()

    def test_missing_required_columns_raises_error(self):
        invalid_df = pd.DataFrame({'compound_id': ['C1', 'C2'], 'structure': ['CCO', 'CCC']})

        with pytest.raises(ValueError, match="must contain 'ID' and 'SMILES' columns"):
            initialize_master_dataframe_empty(
                valid_compounds=invalid_df,
                target_col='Activity'
            )

    def test_preserves_compound_order(self, sample_compounds):
        master_df = initialize_master_dataframe_empty(
            valid_compounds=sample_compounds,
            target_col='Activity'
        )

        assert list(master_df['ID']) == list(sample_compounds['ID'])
        assert list(master_df['SMILES']) == list(sample_compounds['SMILES'])

    def test_initialize_empty_compound_pool(self):
        empty_pool = pd.DataFrame({'ID': [], 'SMILES': []})

        master_df = initialize_master_dataframe_empty(
            valid_compounds=empty_pool,
            target_col='Activity'
        )

        assert len(master_df) == 0
        assert all(col in master_df.columns for col in ['ID', 'SMILES', 'status', 'Activity'])

    def test_duplicate_compound_ids(self):
        duplicate_pool = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_001'],
            'SMILES': ['CCO', 'CCC', 'CCCC']
        })

        master_df = initialize_master_dataframe_empty(
            valid_compounds=duplicate_pool,
            target_col='Activity'
        )

        assert len(master_df) == 3
        assert master_df['ID'].tolist() == ['COMP_001', 'COMP_002', 'COMP_001']


class TestSelectInitialBatch:
    """Tests for select_initial_batch function."""

    def test_random_initialization(self, sample_compounds, mock_oracle, tmp_path):
        """Test random initialization strategy."""
        from learnm8.core.initialization import select_initial_batch

        master_df = initialize_master_dataframe_empty(sample_compounds, 'Activity')

        updated_df, metrics = select_initial_batch(
            compounds_df=master_df,
            oracle=mock_oracle,
            target_col='Activity',
            strategy='random',
            batch_fraction=0.1,
            featurizer_type='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(sample_compounds),
            random_state=42
        )

        expected_batch = int(np.ceil(100 * 0.1))
        labeled = updated_df[updated_df['status'] == STATUS_LABELED]
        assert len(labeled) == expected_batch

        assert (labeled['labeled_cycle'] == 0).all()

        assert (labeled['selected_cycle'] == 0).all()

        assert labeled['Activity'].notna().all()

        unlabeled = updated_df[updated_df['status'] == STATUS_UNLABELED]
        assert len(unlabeled) == 90

        # Verify metrics structure
        assert metrics['cycle'] == 0
        assert metrics['strategy'] == 'random'
        assert metrics['batch_size'] == expected_batch
        assert metrics['has_uncertainty'] is False

    def test_batch_size_calculation(self, sample_compounds, mock_oracle, tmp_path):
        """Test batch size calculation with different fractions."""
        from learnm8.core.initialization import select_initial_batch

        test_cases = [
            (0.01, 1),
            (0.05, 5),
            (0.10, 10),
            (0.23, 23),
        ]

        for fraction, expected in test_cases:
            master_df = initialize_master_dataframe_empty(sample_compounds, 'Activity')

            updated_df, metrics = select_initial_batch(
                compounds_df=master_df,
                oracle=mock_oracle,
                target_col='Activity',
                strategy='random',
                batch_fraction=fraction,
                featurizer_type='morgan',
                cache_dir=tmp_path,
                original_pool_size=len(sample_compounds),
                random_state=42
            )

            labeled = updated_df[updated_df['status'] == STATUS_LABELED]
            assert len(labeled) == expected, f"Fraction {fraction} should select {expected} compounds"

    def test_oracle_measurement_integration(self, sample_compounds, tmp_path):
        """Test integration with real oracle measurement."""
        from learnm8.core.initialization import select_initial_batch
        from learnm8.oracles import CSVOracle

        oracle_df = sample_compounds.copy()
        oracle_df['Activity'] = np.random.uniform(0, 1, len(oracle_df))
        oracle_path = tmp_path / 'oracle.csv'
        oracle_df.to_csv(oracle_path, index=False)
        oracle = CSVOracle(str(oracle_path))

        master_df = initialize_master_dataframe_empty(sample_compounds, 'Activity')

        updated_df, metrics = select_initial_batch(
            compounds_df=master_df,
            oracle=oracle,
            target_col='Activity',
            strategy='random',
            batch_fraction=0.1,
            featurizer_type='morgan',
            cache_dir=tmp_path,
            original_pool_size=len(sample_compounds),
            random_state=42
        )

        labeled = updated_df[updated_df['status'] == STATUS_LABELED]

        for _, row in labeled.iterrows():
            compound_id = row['ID']
            expected_activity = oracle_df[oracle_df['ID'] == compound_id]['Activity'].values[0]
            assert np.isclose(row['Activity'], expected_activity)

    def test_unsupported_strategy_fallback(self, sample_compounds, mock_oracle, tmp_path):
        """Test that unsupported strategy falls back to random with warning."""
        from learnm8.core.initialization import select_initial_batch
        import warnings

        master_df = initialize_master_dataframe_empty(sample_compounds, 'Activity')

        with warnings.catch_warnings(record=True):
            updated_df, metrics = select_initial_batch(
                compounds_df=master_df,
                oracle=mock_oracle,
                target_col='Activity',
                strategy='ucb',
                batch_fraction=0.1,
                featurizer_type='morgan',
                cache_dir=tmp_path,
                original_pool_size=len(sample_compounds),
                random_state=42
            )

        labeled = updated_df[updated_df['status'] == STATUS_LABELED]
        assert len(labeled) == 10

    def test_empty_pool_raises_error(self, mock_oracle, tmp_path):
        """Test that empty compound pool raises error."""
        from learnm8.core.initialization import select_initial_batch

        empty_df = pd.DataFrame(columns=['ID', 'SMILES', 'status', 'labeled_cycle', 'Activity'])
        empty_df['status'] = pd.Categorical(empty_df['status'], categories=['unlabeled', 'labeled', 'pruned'])

        with pytest.raises(ValueError, match="Calculated batch size is 0"):
            select_initial_batch(
                compounds_df=empty_df,
                oracle=mock_oracle,
                target_col='Activity',
                strategy='random',
                batch_fraction=0.1,
                featurizer_type='morgan',
                cache_dir=tmp_path,
                original_pool_size=0,
                random_state=42
            )

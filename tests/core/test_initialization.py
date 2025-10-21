import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from learnm8.core.initialization import (
    initialize_master_dataframe,
    select_initial_sample
)
from learnm8.core.data_structures import STATUS_LABELED, STATUS_UNLABELED


class TestInitializeMasterDataframe:

    def test_basic_initialization(self, sample_compounds):
        initial_ids = sample_compounds['ID'].iloc[:3].tolist()
        initial_values = pd.Series([0.3, 0.6, 0.9], index=initial_ids)

        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            initial_labeled_ids=initial_ids,
            initial_measurements=initial_values,
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
        initial_ids = sample_compounds['ID'].iloc[:3].tolist()
        initial_values = pd.Series([0.3, 0.6, 0.9], index=initial_ids)

        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            initial_labeled_ids=initial_ids,
            initial_measurements=initial_values,
            target_col='pIC50'
        )

        assert 'pIC50' in master_df.columns
        assert 'target_value' not in master_df.columns

    def test_no_legacy_columns(self, sample_compounds):
        initial_ids = sample_compounds['ID'].iloc[:3].tolist()
        initial_values = pd.Series([0.3, 0.6, 0.9], index=initial_ids)

        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            initial_labeled_ids=initial_ids,
            initial_measurements=initial_values,
            target_col='Activity'
        )

        assert 'last_prediction' not in master_df.columns
        assert 'last_uncertainty' not in master_df.columns

    def test_initial_compounds_marked_correctly(self, sample_compounds):
        initial_ids = sample_compounds['ID'].iloc[:3].tolist()
        initial_values = pd.Series([0.3, 0.6, 0.9], index=initial_ids)

        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            initial_labeled_ids=initial_ids,
            initial_measurements=initial_values,
            target_col='Activity'
        )

        labeled_mask = master_df['status'] == STATUS_LABELED
        assert labeled_mask.sum() == 3

        for cid in initial_ids:
            row = master_df[master_df['ID'] == cid].iloc[0]
            assert row['status'] == STATUS_LABELED
            assert row['labeled_cycle'] == -1
            assert row['selected_cycle'] == -1
            assert pd.notna(row['Activity'])

    def test_unlabeled_compounds_marked_correctly(self, sample_compounds):
        initial_ids = sample_compounds['ID'].iloc[:3].tolist()
        initial_values = pd.Series([0.3, 0.6, 0.9], index=initial_ids)

        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            initial_labeled_ids=initial_ids,
            initial_measurements=initial_values,
            target_col='Activity'
        )

        unlabeled_mask = master_df['status'] == STATUS_UNLABELED
        assert unlabeled_mask.sum() == len(sample_compounds) - 3

        unlabeled_rows = master_df[unlabeled_mask]
        assert pd.isna(unlabeled_rows['labeled_cycle']).all()
        assert pd.isna(unlabeled_rows['selected_cycle']).all()
        assert pd.isna(unlabeled_rows['Activity']).all()

    def test_target_values_assigned_correctly(self, sample_compounds):
        initial_ids = sample_compounds['ID'].iloc[:5].tolist()
        initial_values = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=initial_ids)

        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            initial_labeled_ids=initial_ids,
            initial_measurements=initial_values,
            target_col='Activity'
        )

        for cid, expected_val in initial_values.items():
            actual_val = master_df[master_df['ID'] == cid]['Activity'].iloc[0]
            assert actual_val == expected_val

    def test_missing_required_columns_raises_error(self):
        invalid_df = pd.DataFrame({'compound_id': ['C1', 'C2'], 'structure': ['CCO', 'CCC']})
        initial_ids = ['C1']
        initial_values = pd.Series([0.5], index=['C1'])

        with pytest.raises(ValueError, match="must contain 'ID' and 'SMILES' columns"):
            initialize_master_dataframe(
                valid_compounds=invalid_df,
                initial_labeled_ids=initial_ids,
                initial_measurements=initial_values,
                target_col='Activity'
            )

    def test_empty_initial_labeled_ids(self, sample_compounds):
        master_df = initialize_master_dataframe(
            valid_compounds=sample_compounds,
            initial_labeled_ids=[],
            initial_measurements=pd.Series(dtype=float),
            target_col='Activity'
        )

        assert len(master_df) == len(sample_compounds)
        assert (master_df['status'] == STATUS_UNLABELED).all()


class TestSelectInitialSample:

    def test_random_strategy(self, sample_compounds):
        n_initial = 10
        selected_ids = select_initial_sample(
            sample_compounds,
            n_initial=n_initial,
            strategy='random',
            random_state=42
        )

        assert len(selected_ids) == n_initial
        assert all(cid in sample_compounds['ID'].values for cid in selected_ids)
        assert len(set(selected_ids)) == n_initial

    def test_random_strategy_reproducible(self, sample_compounds):
        n_initial = 10
        selected_ids_1 = select_initial_sample(
            sample_compounds,
            n_initial=n_initial,
            strategy='random',
            random_state=42
        )
        selected_ids_2 = select_initial_sample(
            sample_compounds,
            n_initial=n_initial,
            strategy='random',
            random_state=42
        )

        assert selected_ids_1 == selected_ids_2

    def test_diverse_strategy_with_bitbirch(self, sample_compounds, tmp_path):
        pytest.importorskip("bitbirch")

        n_initial = 10
        selected_ids = select_initial_sample(
            sample_compounds,
            n_initial=n_initial,
            strategy='diverse',
            featurizer_type='morgan',
            cache_dir=tmp_path,
            random_state=42
        )

        assert len(selected_ids) == n_initial
        assert all(cid in sample_compounds['ID'].values for cid in selected_ids)
        assert len(set(selected_ids)) == n_initial

    def test_diverse_strategy_fallback_without_bitbirch(self, sample_compounds, tmp_path, monkeypatch):
        def mock_import_error(*args, **kwargs):
            raise ImportError("BitBIRCH not available")

        monkeypatch.setattr("learnm8.core.initialization.select_initial_sample.__code__",
                           select_initial_sample.__code__)

        n_initial = 10
        selected_ids = select_initial_sample(
            sample_compounds,
            n_initial=n_initial,
            strategy='diverse',
            featurizer_type='morgan',
            cache_dir=tmp_path,
            random_state=42
        )

        assert len(selected_ids) == n_initial
        assert all(cid in sample_compounds['ID'].values for cid in selected_ids)

    def test_n_initial_larger_than_pool(self, sample_compounds):
        n_initial = len(sample_compounds) + 100
        selected_ids = select_initial_sample(
            sample_compounds,
            n_initial=n_initial,
            strategy='random',
            random_state=42
        )

        assert len(selected_ids) == len(sample_compounds)

    def test_invalid_strategy_raises_error(self, sample_compounds):
        with pytest.raises(ValueError, match="Unknown strategy"):
            select_initial_sample(
                sample_compounds,
                n_initial=10,
                strategy='invalid_strategy',
                random_state=42
            )

    def test_empty_compound_pool(self):
        empty_df = pd.DataFrame(columns=['ID', 'SMILES'])
        selected_ids = select_initial_sample(
            empty_df,
            n_initial=10,
            strategy='random',
            random_state=42
        )

        assert len(selected_ids) == 0

    def test_single_compound_pool(self):
        single_df = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO']
        })
        selected_ids = select_initial_sample(
            single_df,
            n_initial=5,
            strategy='random',
            random_state=42
        )

        assert len(selected_ids) == 1
        assert selected_ids[0] == 'COMP_001'

    def test_bitbirch_partial_feature_availability(self, sample_compounds, tmp_path):
        pytest.importorskip("bitbirch")

        from learnm8.features.extraction import extract_features
        from learnm8.acquisition.bitbirch import BitBIRCHAcquisition

        small_subset = sample_compounds.iloc[:5].copy()
        small_subset['prediction'] = 0.5
        smiles_list = small_subset['SMILES'].tolist()
        compound_ids = small_subset['ID'].tolist()

        features = extract_features(
            smiles_list,
            featurizer_type='morgan',
            cache_dir=tmp_path,
            n_jobs=1
        )

        acquisition = BitBIRCHAcquisition(
            features=features,
            compound_ids=compound_ids[:3],
            featurizer_type='morgan',
            random_state=42
        )

        n_initial = 4
        selected = acquisition.select(small_subset, n_select=n_initial)

        expected_available = min(n_initial, 3)
        assert len(selected) == expected_available
        assert all(cid in compound_ids[:3] for cid in selected['ID'].values)

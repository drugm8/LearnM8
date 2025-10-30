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

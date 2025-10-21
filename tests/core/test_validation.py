import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from learnm8.core.validation import validate_compound_pool, ValidationResult


class TestValidateCompoundPool:

    def test_all_valid_compounds(self, sample_compounds, tmp_path):
        result = validate_compound_pool(
            sample_compounds,
            featurizer_type='morgan',
            cache_dir=tmp_path
        )

        assert isinstance(result, ValidationResult)
        assert len(result.valid_compounds) == len(sample_compounds)
        assert len(result.invalid_compounds) == 0
        assert len(result.validation_errors) == 0
        assert result.success_rate == 1.0

    def test_all_invalid_compounds(self, tmp_path):
        invalid_df = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['INVALID', 'ALSO_INVALID', 'NOT_A_SMILES']
        })

        result = validate_compound_pool(
            invalid_df,
            featurizer_type='morgan',
            cache_dir=tmp_path
        )

        assert len(result.valid_compounds) == 0
        assert len(result.invalid_compounds) == 3
        assert len(result.validation_errors) == 3
        assert result.success_rate == 0.0

    def test_mixed_valid_invalid(self, tmp_path):
        mixed_df = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004'],
            'SMILES': ['CCO', 'INVALID', 'CCC', 'NOT_SMILES']
        })

        result = validate_compound_pool(
            mixed_df,
            featurizer_type='morgan',
            cache_dir=tmp_path
        )

        assert len(result.valid_compounds) == 2
        assert len(result.invalid_compounds) == 2
        assert len(result.validation_errors) == 2
        assert result.success_rate == 0.5

        valid_ids = set(result.valid_compounds['ID'])
        assert 'COMP_001' in valid_ids
        assert 'COMP_003' in valid_ids

        invalid_ids = set(result.invalid_compounds['ID'])
        assert 'COMP_002' in invalid_ids
        assert 'COMP_004' in invalid_ids

    def test_empty_compound_pool(self, tmp_path):
        empty_df = pd.DataFrame(columns=['ID', 'SMILES'])

        result = validate_compound_pool(
            empty_df,
            featurizer_type='morgan',
            cache_dir=tmp_path
        )

        assert len(result.valid_compounds) == 0
        assert len(result.invalid_compounds) == 0
        assert len(result.validation_errors) == 0
        assert result.success_rate == 0.0

    def test_missing_id_column(self, tmp_path):
        missing_id_df = pd.DataFrame({
            'compound_id': ['C1', 'C2'],
            'SMILES': ['CCO', 'CCC']
        })

        result = validate_compound_pool(
            missing_id_df,
            featurizer_type='morgan',
            cache_dir=tmp_path
        )

        assert len(result.valid_compounds) == 0
        assert len(result.invalid_compounds) == 2
        assert len(result.validation_errors) == 2
        assert 'Missing columns' in list(result.validation_errors.values())[0]

    def test_missing_smiles_column(self, tmp_path):
        missing_smiles_df = pd.DataFrame({
            'ID': ['C1', 'C2'],
            'structure': ['CCO', 'CCC']
        })

        result = validate_compound_pool(
            missing_smiles_df,
            featurizer_type='morgan',
            cache_dir=tmp_path
        )

        assert len(result.valid_compounds) == 0
        assert len(result.invalid_compounds) == 2
        assert len(result.validation_errors) == 2
        assert 'Missing columns' in list(result.validation_errors.values())[0]

    def test_missing_both_columns(self, tmp_path):
        missing_both_df = pd.DataFrame({
            'compound_id': ['C1', 'C2'],
            'structure': ['CCO', 'CCC']
        })

        result = validate_compound_pool(
            missing_both_df,
            featurizer_type='morgan',
            cache_dir=tmp_path
        )

        assert len(result.valid_compounds) == 0
        assert len(result.invalid_compounds) == 2
        assert len(result.validation_errors) == 2

    def test_validation_errors_contain_messages(self, tmp_path):
        invalid_df = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002'],
            'SMILES': ['INVALID_SMILES', 'ALSO_INVALID']
        })

        result = validate_compound_pool(
            invalid_df,
            featurizer_type='morgan',
            cache_dir=tmp_path
        )

        assert len(result.validation_errors) == 2
        for error_msg in result.validation_errors.values():
            assert isinstance(error_msg, str)
            assert len(error_msg) > 0

    def test_cache_benefit_from_repeated_calls(self, sample_compounds, tmp_path):
        result_1 = validate_compound_pool(
            sample_compounds,
            featurizer_type='morgan',
            cache_dir=tmp_path
        )

        result_2 = validate_compound_pool(
            sample_compounds,
            featurizer_type='morgan',
            cache_dir=tmp_path
        )

        assert len(result_1.valid_compounds) == len(result_2.valid_compounds)
        assert len(result_1.invalid_compounds) == len(result_2.invalid_compounds)
        assert result_1.success_rate == result_2.success_rate

    def test_validation_result_structure(self, sample_compounds, tmp_path):
        result = validate_compound_pool(
            sample_compounds,
            featurizer_type='morgan',
            cache_dir=tmp_path
        )

        assert hasattr(result, 'valid_compounds')
        assert hasattr(result, 'invalid_compounds')
        assert hasattr(result, 'validation_errors')
        assert hasattr(result, 'success_rate')

        assert isinstance(result.valid_compounds, pd.DataFrame)
        assert isinstance(result.invalid_compounds, pd.DataFrame)
        assert isinstance(result.validation_errors, dict)
        assert isinstance(result.success_rate, float)

    def test_success_rate_calculation(self, tmp_path):
        mixed_df = pd.DataFrame({
            'ID': ['C1', 'C2', 'C3', 'C4', 'C5'],
            'SMILES': ['CCO', 'CCC', 'INVALID', 'CCN', 'BAD']
        })

        result = validate_compound_pool(
            mixed_df,
            featurizer_type='morgan',
            cache_dir=tmp_path
        )

        expected_rate = 3 / 5
        assert result.success_rate == pytest.approx(expected_rate)

    def test_different_featurizers(self, tmp_path):
        compounds = pd.DataFrame({
            'ID': ['C1', 'C2'],
            'SMILES': ['CCO', 'CCC']
        })

        for featurizer in ['morgan', 'maccs', 'ecfp6']:
            result = validate_compound_pool(
                compounds,
                featurizer_type=featurizer,
                cache_dir=tmp_path
            )

            assert len(result.valid_compounds) == 2
            assert result.success_rate == 1.0

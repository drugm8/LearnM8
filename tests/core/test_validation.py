import pytest
import pandas as pd
from pathlib import Path

from learnm8.core.validation import validate_compound_pool, ValidationResult, _validate_smiles


class TestValidateSMILES:

    def test_validate_smiles_valid(self):
        is_valid, std_smiles, error = _validate_smiles('CCO')
        assert is_valid
        assert std_smiles == 'CCO'
        assert error == ''

    def test_validate_smiles_benzene(self):
        is_valid, std_smiles, error = _validate_smiles('c1ccccc1')
        assert is_valid
        assert std_smiles != ''
        assert error == ''

    def test_validate_smiles_invalid(self):
        is_valid, std_smiles, error = _validate_smiles('INVALID')
        assert not is_valid
        assert std_smiles == ''
        assert len(error) > 0

    def test_validate_smiles_invalid_badsmiles(self):
        is_valid, std_smiles, error = _validate_smiles('BADSMILES')
        assert not is_valid
        assert std_smiles == ''
        assert len(error) > 0

    def test_validate_smiles_empty(self):
        is_valid, std_smiles, error = _validate_smiles('')
        assert not is_valid
        assert std_smiles == ''
        assert len(error) > 0


class TestValidateCompoundPool:

    def test_all_valid_compounds(self, sample_compounds):
        result = validate_compound_pool(
            sample_compounds,
            n_jobs=4,
            progress=False
        )

        assert isinstance(result, ValidationResult)
        assert len(result.valid_compounds) == len(sample_compounds)
        assert len(result.invalid_compounds) == 0
        assert len(result.validation_errors) == 0
        assert result.success_rate == 1.0

    def test_all_invalid_compounds(self):
        invalid_df = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['INVALID', 'ALSO_INVALID', 'NOT_A_SMILES']
        })

        result = validate_compound_pool(
            invalid_df,
            n_jobs=4,
            progress=False
        )

        assert len(result.valid_compounds) == 0
        assert len(result.invalid_compounds) == 3
        assert len(result.validation_errors) == 3
        assert result.success_rate == 0.0

    def test_mixed_valid_invalid(self):
        mixed_df = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004'],
            'SMILES': ['CCO', 'INVALID', 'CCC', 'NOT_SMILES']
        })

        result = validate_compound_pool(
            mixed_df,
            n_jobs=4,
            progress=False
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

    def test_empty_compound_pool(self):
        empty_df = pd.DataFrame(columns=['ID', 'SMILES'])

        result = validate_compound_pool(
            empty_df,
            n_jobs=4,
            progress=False
        )

        assert len(result.valid_compounds) == 0
        assert len(result.invalid_compounds) == 0
        assert len(result.validation_errors) == 0
        assert result.success_rate == 0.0

    def test_missing_id_column(self):
        missing_id_df = pd.DataFrame({
            'compound_id': ['C1', 'C2'],
            'SMILES': ['CCO', 'CCC']
        })

        result = validate_compound_pool(
            missing_id_df,
            n_jobs=4,
            progress=False
        )

        assert len(result.valid_compounds) == 0
        assert len(result.invalid_compounds) == 2
        assert len(result.validation_errors) >= 1
        assert 'Missing columns' in list(result.validation_errors.values())[0]

    def test_missing_smiles_column(self):
        missing_smiles_df = pd.DataFrame({
            'ID': ['C1', 'C2'],
            'structure': ['CCO', 'CCC']
        })

        result = validate_compound_pool(
            missing_smiles_df,
            n_jobs=4,
            progress=False
        )

        assert len(result.valid_compounds) == 0
        assert len(result.invalid_compounds) == 2
        assert len(result.validation_errors) == 2
        assert 'Missing columns' in list(result.validation_errors.values())[0]

    def test_missing_both_columns(self):
        missing_both_df = pd.DataFrame({
            'compound_id': ['C1', 'C2'],
            'structure': ['CCO', 'CCC']
        })

        result = validate_compound_pool(
            missing_both_df,
            n_jobs=4,
            progress=False
        )

        assert len(result.valid_compounds) == 0
        assert len(result.invalid_compounds) == 2
        assert len(result.validation_errors) >= 1

    def test_validation_errors_contain_messages(self):
        invalid_df = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002'],
            'SMILES': ['INVALID_SMILES', 'ALSO_INVALID']
        })

        result = validate_compound_pool(
            invalid_df,
            n_jobs=4,
            progress=False
        )

        assert len(result.validation_errors) == 2
        for error_msg in result.validation_errors.values():
            assert isinstance(error_msg, str)
            assert len(error_msg) > 0

    def test_parallel_execution(self):
        compounds = pd.DataFrame({
            'ID': [f'mol{i}' for i in range(100)],
            'SMILES': ['CCO'] * 100
        })

        result = validate_compound_pool(
            compounds,
            n_jobs=4,
            progress=False
        )

        assert len(result.valid_compounds) == 100
        assert result.success_rate == 1.0

    def test_validation_result_structure(self, sample_compounds):
        result = validate_compound_pool(
            sample_compounds,
            n_jobs=4,
            progress=False
        )

        assert hasattr(result, 'valid_compounds')
        assert hasattr(result, 'invalid_compounds')
        assert hasattr(result, 'validation_errors')
        assert hasattr(result, 'success_rate')

        assert isinstance(result.valid_compounds, pd.DataFrame)
        assert isinstance(result.invalid_compounds, pd.DataFrame)
        assert isinstance(result.validation_errors, dict)
        assert isinstance(result.success_rate, float)

    def test_success_rate_calculation(self):
        mixed_df = pd.DataFrame({
            'ID': ['C1', 'C2', 'C3', 'C4', 'C5'],
            'SMILES': ['CCO', 'CCC', 'INVALID', 'CCN', 'BAD']
        })

        result = validate_compound_pool(
            mixed_df,
            n_jobs=4,
            progress=False
        )

        expected_rate = 3 / 5
        assert result.success_rate == pytest.approx(expected_rate)


class TestValidationResult:

    def test_success_rate_all_valid(self):
        result = ValidationResult(
            valid_compounds=pd.DataFrame({'ID': ['C1', 'C2'], 'SMILES': ['CCO', 'CCC']}),
            invalid_compounds=pd.DataFrame(columns=['ID', 'SMILES']),
            validation_errors={}
        )
        assert result.success_rate == 1.0

    def test_success_rate_all_invalid(self):
        result = ValidationResult(
            valid_compounds=pd.DataFrame(columns=['ID', 'SMILES']),
            invalid_compounds=pd.DataFrame({'ID': ['C1', 'C2'], 'SMILES': ['BAD', 'INVALID']}),
            validation_errors={'C1': 'error1', 'C2': 'error2'}
        )
        assert result.success_rate == 0.0

    def test_success_rate_mixed(self):
        result = ValidationResult(
            valid_compounds=pd.DataFrame({'ID': ['C1'], 'SMILES': ['CCO']}),
            invalid_compounds=pd.DataFrame({'ID': ['C2'], 'SMILES': ['BAD']}),
            validation_errors={'C2': 'error'}
        )
        assert result.success_rate == 0.5

    def test_success_rate_empty(self):
        result = ValidationResult(
            valid_compounds=pd.DataFrame(columns=['ID', 'SMILES']),
            invalid_compounds=pd.DataFrame(columns=['ID', 'SMILES']),
            validation_errors={}
        )
        assert result.success_rate == 0.0

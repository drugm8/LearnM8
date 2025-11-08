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

    def test_duplicate_ids_all_valid_smiles(self):
        """Test that duplicate IDs are detected and marked as invalid."""
        duplicate_df = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_001', 'COMP_003', 'COMP_002'],
            'SMILES': ['CCO', 'CCC', 'CCN', 'CCCC', 'CCCCC']
        })

        result = validate_compound_pool(
            duplicate_df,
            n_jobs=4,
            progress=False
        )

        # First occurrences should be valid (COMP_001, COMP_002, COMP_003)
        assert len(result.valid_compounds) == 3
        # Duplicates should be invalid (2 duplicate rows)
        assert len(result.invalid_compounds) == 2
        # validation_errors has one entry per unique duplicate ID (COMP_001, COMP_002)
        assert len(result.validation_errors) == 2

        # Check that first occurrences are kept
        valid_ids = set(result.valid_compounds['ID'])
        assert 'COMP_001' in valid_ids
        assert 'COMP_002' in valid_ids
        assert 'COMP_003' in valid_ids

        # Check that duplicates have correct error message
        assert result.validation_errors['COMP_001'] == "Duplicate ID"
        assert result.validation_errors['COMP_002'] == "Duplicate ID"

    def test_duplicate_ids_with_invalid_smiles(self):
        """Test duplicate IDs when some have invalid SMILES."""
        duplicate_df = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_001', 'COMP_003'],
            'SMILES': ['CCO', 'INVALID', 'CCC', 'BADSMILES']
        })

        result = validate_compound_pool(
            duplicate_df,
            n_jobs=4,
            progress=False
        )

        # Only first COMP_001 with valid SMILES should be valid
        assert len(result.valid_compounds) == 1
        assert result.valid_compounds.iloc[0]['ID'] == 'COMP_001'

        # Should have 3 invalid: duplicate COMP_001, invalid COMP_002, invalid COMP_003
        assert len(result.invalid_compounds) == 3
        assert len(result.validation_errors) == 3

        # Check error messages
        assert result.validation_errors['COMP_001'] == "Duplicate ID"
        assert 'COMP_002' in result.validation_errors
        assert 'COMP_003' in result.validation_errors

    def test_no_duplicate_ids(self):
        """Test that validation works normally when there are no duplicates."""
        no_dup_df = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCN']
        })

        result = validate_compound_pool(
            no_dup_df,
            n_jobs=4,
            progress=False
        )

        assert len(result.valid_compounds) == 3
        assert len(result.invalid_compounds) == 0
        assert len(result.validation_errors) == 0
        assert result.success_rate == 1.0

    def test_all_duplicate_ids(self):
        """Test when all compounds are duplicates of the first."""
        all_dup_df = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_001', 'COMP_001'],
            'SMILES': ['CCO', 'CCC', 'CCN']
        })

        result = validate_compound_pool(
            all_dup_df,
            n_jobs=4,
            progress=False
        )

        # First occurrence should be valid
        assert len(result.valid_compounds) == 1
        assert result.valid_compounds.iloc[0]['ID'] == 'COMP_001'

        # Two duplicates should be invalid
        assert len(result.invalid_compounds) == 2
        # Note: validation_errors dict has one entry per unique ID
        assert len(result.validation_errors) == 1
        assert 'COMP_001' in result.validation_errors

        # Error should be for duplicate ID
        assert result.validation_errors['COMP_001'] == "Duplicate ID"

    def test_duplicate_ids_keeps_first_occurrence(self):
        """Test that keep='first' behavior is maintained."""
        duplicate_df = pd.DataFrame({
            'ID': ['A', 'B', 'A', 'C', 'B'],
            'SMILES': ['CCO', 'CCC', 'CCN', 'CCCC', 'CCCCC']
        })

        result = validate_compound_pool(
            duplicate_df,
            n_jobs=4,
            progress=False
        )

        # Check that first occurrences are in valid compounds
        assert len(result.valid_compounds) == 3
        valid_ids_list = result.valid_compounds['ID'].tolist()

        # First A should be kept (index 0)
        assert valid_ids_list[0] == 'A'
        assert result.valid_compounds.iloc[0]['SMILES'] == 'CCO'

        # First B should be kept (index 1)
        assert valid_ids_list[1] == 'B'
        assert result.valid_compounds.iloc[1]['SMILES'] == 'CCC'

        # C should be kept
        assert 'C' in valid_ids_list

    def test_single_compound_no_duplicates(self):
        """Test edge case with single compound."""
        single_df = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO']
        })

        result = validate_compound_pool(
            single_df,
            n_jobs=4,
            progress=False
        )

        assert len(result.valid_compounds) == 1
        assert len(result.invalid_compounds) == 0
        assert len(result.validation_errors) == 0
        assert result.success_rate == 1.0


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

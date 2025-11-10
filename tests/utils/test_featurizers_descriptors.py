import pytest
import numpy as np
import polars as pl

from learnm8.utils.featurizers import _compute_mordred_descriptors


pytestmark = pytest.mark.skipif(
    not pytest.importorskip("mordred", reason="Mordred not available"),
    reason="Mordred not available"
)


class TestComputeMordredDescriptors:

    def test_basic_functionality_single_molecule(self):
        smiles_list = ['CCO']
        result = _compute_mordred_descriptors(smiles_list)

        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 1
        assert result.shape[1] > 1000

    def test_basic_functionality_multiple_molecules(self):
        smiles_list = ['CCO', 'CCC', 'CCN']
        result = _compute_mordred_descriptors(smiles_list)

        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 3
        assert result.shape[1] > 1000

    def test_output_shape_matches_input_length(self):
        smiles_list = ['CCO', 'CCC', 'c1ccccc1', 'CCN', 'CCCC']
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == len(smiles_list)

    def test_output_contains_numeric_columns(self):
        smiles_list = ['CCO', 'CCC']
        result = _compute_mordred_descriptors(smiles_list)

        for col in result.columns:
            dtype = result[col].dtype
            assert dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]

    def test_descriptor_count_approximately_1600(self):
        pytest.importorskip("mordred")

        smiles_list = ['CCO']
        result = _compute_mordred_descriptors(smiles_list)

        assert 1500 < result.shape[1] < 2000

    def test_all_values_numeric(self):
        smiles_list = ['CCO', 'CCC', 'CCN']
        result = _compute_mordred_descriptors(smiles_list)

        for col in result.columns:
            dtype = result[col].dtype
            assert dtype.is_numeric()

    def test_handles_nan_values(self):
        smiles_list = ['CCO', 'CCC', 'c1ccccc1']
        result = _compute_mordred_descriptors(smiles_list)

        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 3

    def test_no_inf_values_in_output(self):
        smiles_list = ['CCO', 'CCC', 'c1ccccc1']
        result = _compute_mordred_descriptors(smiles_list)

        for col in result.columns:
            if result[col].dtype.is_float():
                assert not result[col].is_infinite().any()

    def test_invalid_smiles_filled_with_zeros(self):
        smiles_list = ['INVALID_SMILES']
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 1
        assert np.allclose(result.values, 0.0)

    def test_mixed_valid_invalid_smiles(self):
        smiles_list = ['CCO', 'INVALID', 'CCC', 'ALSO_INVALID', 'CCN']
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 5
        assert np.allclose(np.array(result.row(1)), 0.0)
        assert np.allclose(np.array(result.row(3)), 0.0)
        assert not np.allclose(np.array(result.row(0)), 0.0)
        assert not np.allclose(np.array(result.row(2)), 0.0)
        assert not np.allclose(np.array(result.row(4)), 0.0)

    def test_empty_input_list(self):
        smiles_list = []
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 0
        assert result.shape[1] > 1000

    def test_chemically_diverse_molecules(self, diverse_real_compounds):
        smiles_list = diverse_real_compounds['SMILES'].tolist()[:10]
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 10
        assert result.shape[1] > 1000

    def test_aromatic_molecules(self):
        smiles_list = ['c1ccccc1', 'c1ccc(O)cc1', 'c1ccc(N)cc1']
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 3
        assert result.shape[1] > 1000

    def test_aliphatic_molecules(self):
        smiles_list = ['CCCC', 'CCCCC', 'CCCCCC']
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 3
        assert result.shape[1] > 1000

    def test_small_molecules(self):
        smiles_list = ['C', 'CC', 'CCC']
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 3
        assert result.shape[1] > 1000

    def test_complex_molecules(self, small_real_compounds):
        smiles_list = small_real_compounds['SMILES'].tolist()[:5]
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 5
        assert result.shape[1] > 1000

    def test_consistency_across_multiple_calls(self):
        smiles_list = ['CCO', 'CCC', 'CCN']
        result1 = _compute_mordred_descriptors(smiles_list)
        result2 = _compute_mordred_descriptors(smiles_list)

        assert result1.equals(result2)

    def test_descriptor_values_are_reasonable(self):
        smiles_list = ['CCO']
        result = _compute_mordred_descriptors(smiles_list)

        for col in result.columns:
            if result[col].dtype.is_float():
                col_min = result[col].min()
                col_max = result[col].max()
                if col_min is not None and not np.isnan(col_min):
                    assert col_min >= -1e6
                if col_max is not None and not np.isnan(col_max):
                    assert col_max <= 1e6

    def test_different_molecules_produce_different_descriptors(self):
        smiles_list = ['C', 'CCCCCCCCCC']
        result = _compute_mordred_descriptors(smiles_list)

        assert not np.allclose(np.array(result.row(0)), np.array(result.row(1)))

    def test_all_invalid_smiles(self):
        smiles_list = ['INVALID1', 'INVALID2', 'INVALID3']
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 3
        assert np.allclose(result.values, 0.0)

    def test_single_valid_molecule_among_invalid(self):
        smiles_list = ['INVALID1', 'CCO', 'INVALID2']
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 3
        assert np.allclose(np.array(result.row(0)), 0.0)
        assert not np.allclose(np.array(result.row(1)), 0.0)
        assert np.allclose(np.array(result.row(2)), 0.0)

    def test_output_is_dataframe(self):
        smiles_list = ['CCO']
        result = _compute_mordred_descriptors(smiles_list)

        assert isinstance(result, pl.DataFrame)

    def test_edge_case_molecules(self, edge_case_compounds):
        smiles_list = edge_case_compounds['SMILES'].tolist()[:5]
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 5
        assert result.shape[1] > 1000

    def test_molecules_with_heteroatoms(self):
        smiles_list = ['CCO', 'CCN', 'CCS', 'CCCl', 'CCBr']
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 5
        assert result.shape[1] > 1000

    def test_cyclic_molecules(self):
        smiles_list = ['C1CCCCC1', 'c1ccccc1', 'C1CCCC1']
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 3
        assert result.shape[1] > 1000

    def test_molecules_with_multiple_functional_groups(self):
        smiles_list = ['CC(=O)O', 'CC(N)C(=O)O', 'c1ccc(O)c(N)c1']
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 3
        assert result.shape[1] > 1000

    def test_real_pharmaceutical_molecules(self, small_real_compounds):
        smiles_list = small_real_compounds['SMILES'].tolist()[:20]
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 20
        assert result.shape[1] > 1000

    def test_batch_processing_maintains_order(self):
        smiles_list = ['C', 'CC', 'CCC', 'CCCC', 'CCCCC']
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 5
        for i in range(len(smiles_list) - 1):
            assert not np.array_equal(np.array(result.row(i)), np.array(result.row(i+1)))

    def test_descriptor_column_names_are_strings(self):
        smiles_list = ['CCO']
        result = _compute_mordred_descriptors(smiles_list)

        assert all(isinstance(col, str) for col in result.columns)

    def test_no_duplicate_column_names(self):
        smiles_list = ['CCO']
        result = _compute_mordred_descriptors(smiles_list)

        assert len(result.columns) == len(set(result.columns))

    def test_multi_target_compounds(self, multi_target_compounds):
        smiles_list = multi_target_compounds['SMILES'].tolist()[:15]
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 15
        assert result.shape[1] > 1000

    def test_medium_dataset_compounds(self, medium_real_compounds):
        smiles_list = medium_real_compounds['SMILES'].tolist()[:25]
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 25
        assert result.shape[1] > 1000

    def test_large_batch_processing(self, medium_real_compounds):
        smiles_list = medium_real_compounds['SMILES'].tolist()[:50]
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 50
        assert result.shape[1] > 1000

    def test_none_handling_in_rdkit_conversion(self):
        from unittest.mock import patch
        from rdkit import Chem

        smiles_list = ['CCO', 'CCC']

        with patch.object(Chem, 'MolFromSmiles', side_effect=[None, Chem.MolFromSmiles('CCC')]):
            result = _compute_mordred_descriptors(smiles_list)

            assert result.shape[0] == 2
            assert np.allclose(np.array(result.row(0)), 0.0)
            assert not np.allclose(np.array(result.row(1)), 0.0)

    def test_exception_handling_in_rdkit_conversion(self):
        from unittest.mock import patch
        from rdkit import Chem

        smiles_list = ['CCO', 'CCC']

        with patch.object(Chem, 'MolFromSmiles', side_effect=[Exception('Test error'), Chem.MolFromSmiles('CCC')]):
            result = _compute_mordred_descriptors(smiles_list)

            assert result.shape[0] == 2
            assert np.allclose(np.array(result.row(0)), 0.0)
            assert not np.allclose(np.array(result.row(1)), 0.0)

    def test_all_descriptors_are_numeric(self):
        smiles_list = ['CCO', 'CCC']
        result = _compute_mordred_descriptors(smiles_list)

        for col in result.columns:
            dtype = result[col].dtype
            assert dtype.is_numeric()

    def test_regression_compounds(self, regression_compounds):
        smiles_list = regression_compounds['SMILES'].tolist()[:30]
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 30
        assert result.shape[1] > 1000

    def test_classification_compounds(self, classification_compounds):
        smiles_list = classification_compounds['SMILES'].tolist()[:30]
        result = _compute_mordred_descriptors(smiles_list)

        assert result.shape[0] == 30
        assert result.shape[1] > 1000

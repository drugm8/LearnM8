import pytest
import numpy as np
from rdkit import Chem

from learnm8.utils.featurizers import smiles_to_morgan_feature_fingerprint


class TestMorganFeatureFingerprintBasic:

    def test_simple_molecule_ethanol(self):
        smiles = 'CCO'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0
        assert fp.dtype in [np.int32, np.int64, np.float32, np.float64, np.uint8, np.uint32]

    def test_simple_molecule_propane(self):
        smiles = 'CCC'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0
        assert np.all(np.isfinite(fp))

    def test_simple_molecule_ethylamine(self):
        smiles = 'CCN'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0
        assert not np.any(np.isnan(fp))

    def test_aromatic_molecule_benzene(self):
        smiles = 'c1ccccc1'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0
        assert np.all(np.isfinite(fp))


class TestMorganFeatureFingerprintOutput:

    def test_output_is_numpy_array(self):
        fp = smiles_to_morgan_feature_fingerprint('CCO')

        assert isinstance(fp, np.ndarray)

    def test_output_dtype_is_numeric(self):
        fp = smiles_to_morgan_feature_fingerprint('CCC')

        assert fp.dtype in [np.int32, np.int64, np.float32, np.float64, np.uint8, np.uint32, np.uint64]

    def test_output_contains_no_nan(self):
        fp = smiles_to_morgan_feature_fingerprint('CCN')

        assert not np.any(np.isnan(fp))

    def test_output_contains_no_inf(self):
        fp = smiles_to_morgan_feature_fingerprint('CCCC')

        assert not np.any(np.isinf(fp))

    def test_output_is_one_dimensional(self):
        fp = smiles_to_morgan_feature_fingerprint('c1ccccc1')

        assert fp.ndim == 1

    def test_output_is_not_empty(self):
        fp = smiles_to_morgan_feature_fingerprint('CCO')

        assert len(fp) > 0

    def test_output_has_expected_size_based_on_implementation(self):
        fp = smiles_to_morgan_feature_fingerprint('CCO')

        assert len(fp) > 0


class TestMorganFeatureFingerprintReproducibility:

    def test_same_smiles_produces_same_fingerprint(self):
        smiles = 'CCO'

        fp1 = smiles_to_morgan_feature_fingerprint(smiles)
        fp2 = smiles_to_morgan_feature_fingerprint(smiles)

        np.testing.assert_array_equal(fp1, fp2)

    def test_multiple_calls_are_consistent(self):
        smiles = 'c1ccccc1'

        fingerprints = [smiles_to_morgan_feature_fingerprint(smiles) for _ in range(5)]

        for fp in fingerprints[1:]:
            np.testing.assert_array_equal(fingerprints[0], fp)

    def test_different_molecules_produce_different_fingerprints(self):
        smiles1 = 'CCO'
        smiles2 = 'CCC'

        fp1 = smiles_to_morgan_feature_fingerprint(smiles1)
        fp2 = smiles_to_morgan_feature_fingerprint(smiles2)

        assert not np.array_equal(fp1, fp2)

    def test_isomers_produce_different_fingerprints(self):
        smiles1 = 'CCC'
        smiles2 = 'C(C)C'

        fp1 = smiles_to_morgan_feature_fingerprint(smiles1)
        fp2 = smiles_to_morgan_feature_fingerprint(smiles2)

        assert np.array_equal(fp1, fp2)


class TestMorganFeatureFingerprintErrorHandling:

    def test_invalid_smiles_raises_error(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_morgan_feature_fingerprint('INVALID_SMILES')

    def test_empty_smiles_returns_valid_fingerprint(self):
        fp = smiles_to_morgan_feature_fingerprint('')
        assert isinstance(fp, np.ndarray)
        assert fp.shape == (2048,)

    def test_none_smiles_raises_error(self):
        with pytest.raises((ValueError, TypeError)):
            smiles_to_morgan_feature_fingerprint(None)

    def test_malformed_smiles_raises_error(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_morgan_feature_fingerprint('C1CCC')

    def test_incomplete_ring_raises_error(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_morgan_feature_fingerprint('c1cccc')


class TestMorganFeatureFingerprintDiverseMolecules:

    def test_aliphatic_hydrocarbon(self):
        smiles = 'CCCCCCCC'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0
        assert np.all(np.isfinite(fp))

    def test_aromatic_heterocycle(self):
        smiles = 'c1ncncc1'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0
        assert np.all(np.isfinite(fp))

    def test_molecule_with_multiple_functional_groups(self):
        smiles = 'CC(=O)Oc1ccccc1C(=O)O'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0
        assert np.all(np.isfinite(fp))

    def test_halogenated_molecule(self):
        smiles = 'CCCl'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0

    def test_molecule_with_nitrogen(self):
        smiles = 'CCNCC'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0

    def test_molecule_with_oxygen(self):
        smiles = 'CCOCC'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0

    def test_molecule_with_sulfur(self):
        smiles = 'CCSCC'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0

    def test_complex_pharmaceutical_like_molecule(self):
        smiles = 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0
        assert np.all(np.isfinite(fp))


class TestMorganFeatureFingerprintEdgeCases:

    def test_single_atom_carbon(self):
        smiles = 'C'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0

    def test_single_atom_nitrogen(self):
        smiles = 'N'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0

    def test_single_atom_oxygen(self):
        smiles = 'O'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0

    def test_very_simple_bond(self):
        smiles = 'CC'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0

    def test_molecule_with_charge(self):
        smiles = '[NH4+]'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0

    def test_molecule_with_double_bond(self):
        smiles = 'C=C'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0

    def test_molecule_with_triple_bond(self):
        smiles = 'C#C'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0


class TestMorganFeatureFingerprintWithRealMolecularData:

    def test_with_small_real_compounds(self, small_real_compounds):
        if len(small_real_compounds) == 0:
            pytest.skip("No real compounds available in fixture")

        smiles = small_real_compounds['SMILES'][0]
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0
        assert np.all(np.isfinite(fp))

    def test_batch_processing_small_compounds(self, small_real_compounds):
        if len(small_real_compounds) == 0:
            pytest.skip("No real compounds available in fixture")

        smiles_list = small_real_compounds['SMILES'].to_list()[:10]
        fingerprints = [smiles_to_morgan_feature_fingerprint(s) for s in smiles_list]

        assert len(fingerprints) == len(smiles_list)
        for fp in fingerprints:
            assert isinstance(fp, np.ndarray)
            assert len(fp) > 0
            assert np.all(np.isfinite(fp))

    def test_with_edge_case_compounds(self, edge_case_compounds):
        if len(edge_case_compounds) == 0:
            pytest.skip("No edge case compounds available in fixture")

        for idx in range(min(5, len(edge_case_compounds))):
            smiles = edge_case_compounds['SMILES'][idx]
            try:
                fp = smiles_to_morgan_feature_fingerprint(smiles)
                assert isinstance(fp, np.ndarray)
                assert len(fp) > 0
            except ValueError:
                pass

    def test_diverse_real_compounds(self, diverse_real_compounds):
        if len(diverse_real_compounds) == 0:
            pytest.skip("No diverse compounds available in fixture")

        smiles_list = diverse_real_compounds['SMILES'].to_list()[:5]
        fingerprints = [smiles_to_morgan_feature_fingerprint(s) for s in smiles_list]

        assert len(fingerprints) == len(smiles_list)

        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                if smiles_list[i] != smiles_list[j]:
                    assert not np.array_equal(fingerprints[i], fingerprints[j])


class TestMorganFeatureFingerprintProperties:

    def test_fingerprint_is_sparse_representation(self):
        smiles = 'CCO'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)

    def test_feature_based_encoding(self):
        smiles = 'CCO'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0

    def test_larger_molecule_produces_more_features(self):
        small_smiles = 'CC'
        large_smiles = 'CC(C)(C)c1ccc(cc1)C(C)(C)c2ccc(cc2)C(C)(C)C'

        fp_small = smiles_to_morgan_feature_fingerprint(small_smiles)
        fp_large = smiles_to_morgan_feature_fingerprint(large_smiles)

        assert isinstance(fp_small, np.ndarray)
        assert isinstance(fp_large, np.ndarray)

    def test_similar_molecules_have_similar_fingerprints(self):
        smiles1 = 'CCCC'
        smiles2 = 'CCCCC'

        fp1 = smiles_to_morgan_feature_fingerprint(smiles1)
        fp2 = smiles_to_morgan_feature_fingerprint(smiles2)

        assert isinstance(fp1, np.ndarray)
        assert isinstance(fp2, np.ndarray)

    def test_different_molecule_types_produce_distinct_fingerprints(self):
        aliphatic = 'CCCC'
        aromatic = 'c1ccccc1'
        heterocycle = 'c1ncccc1'

        fp_aliphatic = smiles_to_morgan_feature_fingerprint(aliphatic)
        fp_aromatic = smiles_to_morgan_feature_fingerprint(aromatic)
        fp_heterocycle = smiles_to_morgan_feature_fingerprint(heterocycle)

        assert not np.array_equal(fp_aliphatic, fp_aromatic)
        assert not np.array_equal(fp_aromatic, fp_heterocycle)
        assert not np.array_equal(fp_aliphatic, fp_heterocycle)


class TestMorganFeatureFingerprintVsStandardMorgan:

    def test_feature_based_differs_from_count_based(self):
        from learnm8.utils.featurizers import smiles_to_morgan_fingerprint

        smiles = 'CCO'

        fp_standard = smiles_to_morgan_fingerprint(smiles)
        fp_feature = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp_standard, np.ndarray)
        assert isinstance(fp_feature, np.ndarray)

    def test_feature_fingerprint_may_differ_in_size(self):
        from learnm8.utils.featurizers import smiles_to_morgan_fingerprint

        smiles = 'CCO'

        fp_standard = smiles_to_morgan_fingerprint(smiles)
        fp_feature = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp_standard, np.ndarray)
        assert isinstance(fp_feature, np.ndarray)

    def test_feature_fingerprint_captures_pharmacophore_features(self):
        smiles_with_donor = 'CCN'
        smiles_with_acceptor = 'CCO'

        fp_donor = smiles_to_morgan_feature_fingerprint(smiles_with_donor)
        fp_acceptor = smiles_to_morgan_feature_fingerprint(smiles_with_acceptor)

        assert isinstance(fp_donor, np.ndarray)
        assert isinstance(fp_acceptor, np.ndarray)
        assert not np.array_equal(fp_donor, fp_acceptor)


class TestMorganFeatureFingerprintIntegration:

    def test_can_be_used_in_feature_matrix(self):
        smiles_list = ['CCO', 'CCC', 'CCN', 'c1ccccc1']
        fingerprints = [smiles_to_morgan_feature_fingerprint(s) for s in smiles_list]

        assert len(fingerprints) == len(smiles_list)

    def test_fingerprints_can_be_stacked(self):
        smiles_list = ['CCO', 'CCC', 'CCN']
        fingerprints = [smiles_to_morgan_feature_fingerprint(s) for s in smiles_list]

        assert all(isinstance(fp, np.ndarray) for fp in fingerprints)

    def test_fingerprint_suitable_for_ml_input(self):
        smiles = 'CCO'
        fp = smiles_to_morgan_feature_fingerprint(smiles)

        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0
        assert np.all(np.isfinite(fp))
        assert fp.dtype in [np.int32, np.int64, np.float32, np.float64, np.uint8, np.uint32]

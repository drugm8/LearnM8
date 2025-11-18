import pytest
import numpy as np
from rdkit import Chem

from learnm8.utils.featurizers import smiles_to_ecfp6_fingerprint, smiles_to_morgan_fingerprint


class TestECFP6BasicFunctionality:

    def test_simple_molecule_ethanol(self):
        fp = smiles_to_ecfp6_fingerprint('CCO')

        assert fp.shape == (2048,)
        assert fp.dtype == np.uint8
        assert np.all(np.isfinite(fp))

    def test_simple_molecule_propane(self):
        fp = smiles_to_ecfp6_fingerprint('CCC')

        assert fp.shape == (2048,)
        assert fp.dtype == np.uint8
        assert np.all((fp == 0) | (fp == 1))

    def test_simple_molecule_ethylamine(self):
        fp = smiles_to_ecfp6_fingerprint('CCN')

        assert fp.shape == (2048,)
        assert fp.dtype == np.uint8
        assert np.sum(fp) > 0

    def test_benzene_ring(self):
        fp = smiles_to_ecfp6_fingerprint('c1ccccc1')

        assert fp.shape == (2048,)
        assert fp.dtype == np.uint8
        assert np.sum(fp) > 0


class TestECFP6OutputProperties:

    def test_output_is_binary_vector(self):
        fp = smiles_to_ecfp6_fingerprint('CCO')

        assert np.all((fp == 0) | (fp == 1))

    def test_fingerprint_sparsity_simple_molecule(self):
        fp = smiles_to_ecfp6_fingerprint('CCO')

        num_set_bits = np.sum(fp)
        assert num_set_bits > 0
        assert num_set_bits < 100

    def test_fingerprint_sparsity_complex_molecule(self):
        fp = smiles_to_ecfp6_fingerprint('c1ccc(cc1)C(=O)O')

        num_set_bits = np.sum(fp)
        assert num_set_bits > 0

    def test_no_nan_values(self):
        fp = smiles_to_ecfp6_fingerprint('CCO')

        assert not np.any(np.isnan(fp))

    def test_no_infinite_values(self):
        fp = smiles_to_ecfp6_fingerprint('c1ccccc1')

        assert not np.any(np.isinf(fp))


class TestECFP6Reproducibility:

    def test_same_smiles_same_fingerprint(self):
        fp1 = smiles_to_ecfp6_fingerprint('CCO')
        fp2 = smiles_to_ecfp6_fingerprint('CCO')

        np.testing.assert_array_equal(fp1, fp2)

    def test_multiple_calls_consistent(self):
        smiles = 'c1ccccc1'
        fps = [smiles_to_ecfp6_fingerprint(smiles) for _ in range(5)]

        for fp in fps[1:]:
            np.testing.assert_array_equal(fps[0], fp)

    def test_canonicalization_different_representations(self):
        fp1 = smiles_to_ecfp6_fingerprint('c1ccccc1')
        fp2 = smiles_to_ecfp6_fingerprint('C1=CC=CC=C1')

        np.testing.assert_array_equal(fp1, fp2)


class TestECFP6ErrorHandling:

    def test_invalid_smiles_raises_error(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_ecfp6_fingerprint('INVALID_SMILES')

    def test_empty_string_valid_molecule(self):
        fp = smiles_to_ecfp6_fingerprint('')

        assert fp.shape == (2048,)
        assert fp.dtype == np.uint8

    def test_none_input_raises_error(self):
        with pytest.raises((ValueError, TypeError)):
            smiles_to_ecfp6_fingerprint(None)

    def test_malformed_smiles_raises_error(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_ecfp6_fingerprint('C(C(C')

    def test_invalid_atom_symbols(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_ecfp6_fingerprint('CXO')


class TestECFP6ChemicalDiversity:

    def test_different_molecules_different_fingerprints(self):
        fp_ethanol = smiles_to_ecfp6_fingerprint('CCO')
        fp_propane = smiles_to_ecfp6_fingerprint('CCC')

        assert not np.array_equal(fp_ethanol, fp_propane)

    def test_aliphatic_vs_aromatic(self):
        fp_cyclohexane = smiles_to_ecfp6_fingerprint('C1CCCCC1')
        fp_benzene = smiles_to_ecfp6_fingerprint('c1ccccc1')

        assert not np.array_equal(fp_cyclohexane, fp_benzene)

    def test_functional_group_differences(self):
        fp_alcohol = smiles_to_ecfp6_fingerprint('CCO')
        fp_amine = smiles_to_ecfp6_fingerprint('CCN')
        fp_thiol = smiles_to_ecfp6_fingerprint('CCS')

        assert not np.array_equal(fp_alcohol, fp_amine)
        assert not np.array_equal(fp_alcohol, fp_thiol)
        assert not np.array_equal(fp_amine, fp_thiol)

    def test_structural_isomers_different(self):
        fp_propanol_1 = smiles_to_ecfp6_fingerprint('CCCO')
        fp_propanol_2 = smiles_to_ecfp6_fingerprint('CC(C)O')

        assert not np.array_equal(fp_propanol_1, fp_propanol_2)


class TestECFP6RadiusEffect:

    def test_larger_radius_captures_more_context(self):
        fp_morgan = smiles_to_morgan_fingerprint('c1ccc(cc1)C(=O)O')
        fp_ecfp6 = smiles_to_ecfp6_fingerprint('c1ccc(cc1)C(=O)O')

        assert not np.array_equal(fp_morgan, fp_ecfp6)

    def test_simple_molecule_radius_difference(self):
        fp_morgan = smiles_to_morgan_fingerprint('CCCCCCCC')
        fp_ecfp6 = smiles_to_ecfp6_fingerprint('CCCCCCCC')

        assert not np.array_equal(fp_morgan, fp_ecfp6)

    def test_complex_molecule_radius_difference(self):
        smiles = 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'
        fp_morgan = smiles_to_morgan_fingerprint(smiles)
        fp_ecfp6 = smiles_to_ecfp6_fingerprint(smiles)

        assert not np.array_equal(fp_morgan, fp_ecfp6)

    def test_branched_molecule_radius_sensitivity(self):
        smiles = 'CC(C)(C)c1ccc(cc1)C(C)(C)C'
        fp_morgan = smiles_to_morgan_fingerprint(smiles)
        fp_ecfp6 = smiles_to_ecfp6_fingerprint(smiles)

        assert fp_morgan.shape == fp_ecfp6.shape
        assert not np.array_equal(fp_morgan, fp_ecfp6)


class TestECFP6EdgeCases:

    def test_single_atom_molecule(self):
        fp = smiles_to_ecfp6_fingerprint('C')

        assert fp.shape == (2048,)
        assert np.sum(fp) > 0

    def test_two_atom_molecule(self):
        fp = smiles_to_ecfp6_fingerprint('CC')

        assert fp.shape == (2048,)
        assert np.sum(fp) > 0

    def test_large_molecule(self, small_real_compounds):
        if len(small_real_compounds) > 0:
            large_smiles = small_real_compounds['SMILES'][0]
            fp = smiles_to_ecfp6_fingerprint(large_smiles)

            assert fp.shape == (2048,)
            assert fp.dtype == np.uint8

    def test_molecule_with_charge(self):
        fp = smiles_to_ecfp6_fingerprint('[NH4+]')

        assert fp.shape == (2048,)
        assert np.sum(fp) > 0

    def test_molecule_with_multiple_rings(self):
        fp = smiles_to_ecfp6_fingerprint('c1ccc2c(c1)ccc3c2cccc3')

        assert fp.shape == (2048,)
        assert np.sum(fp) > 0

    def test_molecule_with_heteroatoms(self):
        fp = smiles_to_ecfp6_fingerprint('c1cnc(nc1)N')

        assert fp.shape == (2048,)
        assert np.sum(fp) > 0


class TestECFP6RealMolecularData:

    def test_small_real_compounds(self, small_real_compounds):
        if len(small_real_compounds) == 0:
            pytest.skip("No real compounds available")

        smiles_list = small_real_compounds['SMILES'].to_list()[:10]
        fps = [smiles_to_ecfp6_fingerprint(smiles) for smiles in smiles_list]

        assert len(fps) == len(smiles_list)
        for fp in fps:
            assert fp.shape == (2048,)
            assert fp.dtype == np.uint8

    def test_diverse_real_compounds(self, diverse_real_compounds):
        if len(diverse_real_compounds) == 0:
            pytest.skip("No diverse compounds available")

        smiles_list = diverse_real_compounds['SMILES'].to_list()[:5]
        fps = [smiles_to_ecfp6_fingerprint(smiles) for smiles in smiles_list]

        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                unique_fps = not np.array_equal(fps[i], fps[j])
                if unique_fps:
                    break

        assert len(fps) == len(smiles_list)

    def test_medium_real_compounds(self, medium_real_compounds):
        if len(medium_real_compounds) == 0:
            pytest.skip("No medium compounds available")

        smiles_list = medium_real_compounds['SMILES'].to_list()[:5]
        fps = [smiles_to_ecfp6_fingerprint(smiles) for smiles in smiles_list]

        assert len(fps) == len(smiles_list)
        for fp in fps:
            assert np.all(np.isfinite(fp))

    def test_edge_case_compounds(self, edge_case_compounds):
        if len(edge_case_compounds) == 0:
            pytest.skip("No edge case compounds available")

        smiles_list = edge_case_compounds['SMILES'].to_list()

        for smiles in smiles_list:
            try:
                fp = smiles_to_ecfp6_fingerprint(smiles)
                assert fp.shape == (2048,)
                assert fp.dtype == np.uint8
            except ValueError:
                pass


class TestECFP6Integration:

    def test_consistent_with_rdkit(self):
        smiles = 'CCO'
        fp_custom = smiles_to_ecfp6_fingerprint(smiles)

        from rdkit import DataStructs
        from rdkit.Chem import rdFingerprintGenerator

        mol = Chem.MolFromSmiles(smiles)
        ecfp6_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
        fp_rdkit = ecfp6_gen.GetFingerprint(mol)

        arr_rdkit = np.zeros(len(fp_rdkit), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp_rdkit, arr_rdkit)

        np.testing.assert_array_equal(fp_custom, arr_rdkit)

    def test_batch_processing_consistency(self):
        smiles_list = ['CCO', 'CCC', 'CCN', 'c1ccccc1']
        fps_individual = [smiles_to_ecfp6_fingerprint(s) for s in smiles_list]

        fps_batch = [smiles_to_ecfp6_fingerprint(s) for s in smiles_list]

        for fp_ind, fp_batch in zip(fps_individual, fps_batch):
            np.testing.assert_array_equal(fp_ind, fp_batch)

    def test_fingerprint_similarity_calculation(self):
        fp1 = smiles_to_ecfp6_fingerprint('CCO')
        fp2 = smiles_to_ecfp6_fingerprint('CCCO')
        fp3 = smiles_to_ecfp6_fingerprint('c1ccccc1')

        similarity_12 = np.dot(fp1, fp2) / (np.linalg.norm(fp1) * np.linalg.norm(fp2) + 1e-10)
        similarity_13 = np.dot(fp1, fp3) / (np.linalg.norm(fp1) * np.linalg.norm(fp3) + 1e-10)

        assert similarity_12 >= 0
        assert similarity_13 >= 0


class TestECFP6MolecularComplexity:

    def test_peptide_like_structure(self):
        smiles = 'CC(C)C(C(=O)NC(Cc1ccccc1)C(=O)O)N'
        fp = smiles_to_ecfp6_fingerprint(smiles)

        assert fp.shape == (2048,)
        assert np.sum(fp) > 0

    def test_sugar_like_structure(self):
        smiles = 'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O'
        fp = smiles_to_ecfp6_fingerprint(smiles)

        assert fp.shape == (2048,)
        assert np.sum(fp) > 0

    def test_drug_like_structure(self):
        smiles = 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'
        fp = smiles_to_ecfp6_fingerprint(smiles)

        assert fp.shape == (2048,)
        assert np.sum(fp) > 0

    def test_macrocycle_structure(self):
        smiles = 'C1CCCCCCCCCCC1'
        fp = smiles_to_ecfp6_fingerprint(smiles)

        assert fp.shape == (2048,)
        assert np.sum(fp) > 0

    def test_fused_ring_system(self):
        smiles = 'c1ccc2c(c1)ccc3c2cccc3'
        fp = smiles_to_ecfp6_fingerprint(smiles)

        assert fp.shape == (2048,)
        assert np.sum(fp) > 0


class TestECFP6TypeConsistency:

    def test_return_type_is_ndarray(self):
        fp = smiles_to_ecfp6_fingerprint('CCO')

        assert isinstance(fp, np.ndarray)

    def test_dtype_is_uint8(self):
        fp = smiles_to_ecfp6_fingerprint('CCO')

        assert fp.dtype == np.uint8

    def test_shape_always_2048(self):
        smiles_list = ['C', 'CCO', 'c1ccccc1', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O']

        for smiles in smiles_list:
            fp = smiles_to_ecfp6_fingerprint(smiles)
            assert fp.shape == (2048,)

    def test_ndim_is_one(self):
        fp = smiles_to_ecfp6_fingerprint('CCO')

        assert fp.ndim == 1

    def test_length_is_2048(self):
        fp = smiles_to_ecfp6_fingerprint('CCO')

        assert len(fp) == 2048

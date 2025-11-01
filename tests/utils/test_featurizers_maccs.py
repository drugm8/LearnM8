import pytest
import numpy as np
from rdkit import RDLogger

from learnm8.utils.featurizers import smiles_to_maccs_fingerprint

RDLogger.DisableLog('rdApp.*')


class TestSmilesToMaccsFingerprint:

    def test_basic_functionality_simple_molecules(self):
        fp = smiles_to_maccs_fingerprint('CCO')

        assert isinstance(fp, np.ndarray)
        assert fp.shape == (167,)
        assert fp.dtype == np.uint8

    def test_output_shape_validation(self):
        simple_molecules = ['CCO', 'CCC', 'CCN', 'c1ccccc1']

        for smiles in simple_molecules:
            fp = smiles_to_maccs_fingerprint(smiles)
            assert fp.shape == (167,), f"Failed for {smiles}"
            assert len(fp) == 167

    def test_output_dtype_validation(self):
        fp = smiles_to_maccs_fingerprint('CCO')

        assert fp.dtype == np.uint8
        assert np.all((fp == 0) | (fp == 1))

    def test_binary_values(self):
        fp = smiles_to_maccs_fingerprint('c1ccccc1')

        assert np.all(np.isin(fp, [0, 1]))
        assert fp.min() >= 0
        assert fp.max() <= 1

    def test_reproducibility_same_molecule(self):
        smiles = 'CCO'

        fp1 = smiles_to_maccs_fingerprint(smiles)
        fp2 = smiles_to_maccs_fingerprint(smiles)
        fp3 = smiles_to_maccs_fingerprint(smiles)

        np.testing.assert_array_equal(fp1, fp2)
        np.testing.assert_array_equal(fp2, fp3)

    def test_reproducibility_multiple_calls(self):
        test_molecules = ['CCO', 'CCC', 'c1ccccc1', 'CCN']

        for smiles in test_molecules:
            fp1 = smiles_to_maccs_fingerprint(smiles)
            fp2 = smiles_to_maccs_fingerprint(smiles)

            np.testing.assert_array_equal(fp1, fp2)

    def test_invalid_smiles_raises_error(self):
        invalid_smiles = 'INVALID_SMILES_STRING'

        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_maccs_fingerprint(invalid_smiles)

    def test_various_invalid_smiles(self):
        invalid_smiles_list = [
            'XXX',
            '123456',
            'C(C(C',
            'ABCDEF',
            'C1CCC',
        ]

        for smiles in invalid_smiles_list:
            with pytest.raises(ValueError, match="Invalid SMILES"):
                smiles_to_maccs_fingerprint(smiles)

    def test_single_atom_molecules(self):
        single_atoms = ['C', 'N', 'O', 'F']

        for smiles in single_atoms:
            fp = smiles_to_maccs_fingerprint(smiles)
            assert fp.shape == (167,)
            assert np.sum(fp) > 0

    def test_aliphatic_molecules(self):
        fp_ethanol = smiles_to_maccs_fingerprint('CCO')
        fp_propane = smiles_to_maccs_fingerprint('CCC')
        fp_ethylamine = smiles_to_maccs_fingerprint('CCN')

        assert fp_ethanol.shape == (167,)
        assert fp_propane.shape == (167,)
        assert fp_ethylamine.shape == (167,)

        assert not np.array_equal(fp_ethanol, fp_propane)
        assert not np.array_equal(fp_ethanol, fp_ethylamine)

    def test_aromatic_molecules(self):
        fp_benzene = smiles_to_maccs_fingerprint('c1ccccc1')
        fp_toluene = smiles_to_maccs_fingerprint('Cc1ccccc1')
        fp_phenol = smiles_to_maccs_fingerprint('Oc1ccccc1')

        assert fp_benzene.shape == (167,)
        assert fp_toluene.shape == (167,)
        assert fp_phenol.shape == (167,)

        assert np.sum(fp_benzene) > 0
        assert not np.array_equal(fp_benzene, fp_toluene)

    def test_different_molecules_different_fingerprints(self):
        molecules = ['CCO', 'CCC', 'c1ccccc1', 'CCN', 'CCCC']
        fingerprints = [smiles_to_maccs_fingerprint(s) for s in molecules]

        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                assert not np.array_equal(fingerprints[i], fingerprints[j])

    def test_functional_group_recognition(self):
        fp_alcohol = smiles_to_maccs_fingerprint('CCO')
        fp_amine = smiles_to_maccs_fingerprint('CCN')
        fp_ether = smiles_to_maccs_fingerprint('COCC')

        assert not np.array_equal(fp_alcohol, fp_amine)
        assert not np.array_equal(fp_alcohol, fp_ether)
        assert not np.array_equal(fp_amine, fp_ether)

    def test_structural_keys_activated(self):
        fp_simple = smiles_to_maccs_fingerprint('C')
        fp_complex = smiles_to_maccs_fingerprint('c1ccc(cc1)C(=O)O')

        simple_count = np.sum(fp_simple)
        complex_count = np.sum(fp_complex)

        assert simple_count > 0
        assert complex_count > simple_count

    def test_chemically_diverse_structures(self):
        diverse_molecules = [
            'CCO',
            'c1ccccc1',
            'CC(C)C',
            'CC(=O)O',
            'c1ccc2c(c1)ccc3c2cccc3',
            'CCN(CC)CC',
        ]

        fingerprints = []
        for smiles in diverse_molecules:
            fp = smiles_to_maccs_fingerprint(smiles)
            assert fp.shape == (167,)
            fingerprints.append(fp)

        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                assert not np.array_equal(fingerprints[i], fingerprints[j])

    def test_real_compounds_from_fixtures(self, small_real_compounds):
        smiles_list = small_real_compounds['SMILES'].tolist()[:10]

        for smiles in smiles_list:
            fp = smiles_to_maccs_fingerprint(smiles)
            assert fp.shape == (167,)
            assert fp.dtype == np.uint8
            assert np.all(np.isin(fp, [0, 1]))

    def test_diverse_real_compounds(self, diverse_real_compounds):
        smiles_list = diverse_real_compounds['SMILES'].tolist()[:15]

        fingerprints = []
        for smiles in smiles_list:
            fp = smiles_to_maccs_fingerprint(smiles)
            assert fp.shape == (167,)
            fingerprints.append(fp)

        fingerprints = np.array(fingerprints)
        assert fingerprints.shape == (15, 167)

    def test_edge_case_compounds(self, edge_case_compounds):
        smiles_list = edge_case_compounds['SMILES'].tolist()[:10]

        for smiles in smiles_list:
            try:
                fp = smiles_to_maccs_fingerprint(smiles)
                assert fp.shape == (167,)
                assert fp.dtype == np.uint8
            except ValueError:
                pass

    def test_no_nan_values(self):
        molecules = ['CCO', 'CCC', 'c1ccccc1', 'CCN']

        for smiles in molecules:
            fp = smiles_to_maccs_fingerprint(smiles)
            assert not np.any(np.isnan(fp))

    def test_no_inf_values(self):
        molecules = ['CCO', 'CCC', 'c1ccccc1', 'CCN']

        for smiles in molecules:
            fp = smiles_to_maccs_fingerprint(smiles)
            assert not np.any(np.isinf(fp))

    def test_all_values_finite(self):
        molecules = ['CCO', 'CCC', 'c1ccccc1', 'CCN', 'CCCC']

        for smiles in molecules:
            fp = smiles_to_maccs_fingerprint(smiles)
            assert np.all(np.isfinite(fp))

    def test_complex_pharmaceutical_molecules(self):
        complex_molecules = [
            'CC(C)Cc1ccc(cc1)C(C)C(O)=O',
            'CC(=O)Oc1ccccc1C(=O)O',
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        ]

        for smiles in complex_molecules:
            fp = smiles_to_maccs_fingerprint(smiles)
            assert fp.shape == (167,)
            assert np.sum(fp) > 0

    def test_stereochemistry_handling(self):
        chiral_molecule = 'C[C@H](O)c1ccccc1'

        fp = smiles_to_maccs_fingerprint(chiral_molecule)
        assert fp.shape == (167,)

    def test_charged_molecules(self):
        charged_molecules = [
            'C[NH3+]',
            'CC(=O)[O-]',
        ]

        for smiles in charged_molecules:
            fp = smiles_to_maccs_fingerprint(smiles)
            assert fp.shape == (167,)

    def test_halogenated_compounds(self):
        halogenated = [
            'CCCl',
            'CCBr',
            'CCI',
            'CCF',
        ]

        fingerprints = []
        for smiles in halogenated:
            fp = smiles_to_maccs_fingerprint(smiles)
            assert fp.shape == (167,)
            fingerprints.append(fp)

        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                assert not np.array_equal(fingerprints[i], fingerprints[j])

    def test_heterocyclic_compounds(self):
        heterocycles = [
            'c1ccncc1',
            'c1cncnc1',
            'c1ccc2ncccc2c1',
        ]

        for smiles in heterocycles:
            fp = smiles_to_maccs_fingerprint(smiles)
            assert fp.shape == (167,)
            assert np.sum(fp) > 0

    def test_maccs_specific_structural_keys(self):
        fp_carbonyl = smiles_to_maccs_fingerprint('CC(=O)C')
        fp_hydroxyl = smiles_to_maccs_fingerprint('CCO')
        fp_aromatic = smiles_to_maccs_fingerprint('c1ccccc1')
        fp_amine = smiles_to_maccs_fingerprint('CCN')

        assert not np.array_equal(fp_carbonyl, fp_hydroxyl)
        assert not np.array_equal(fp_carbonyl, fp_aromatic)
        assert not np.array_equal(fp_carbonyl, fp_amine)

    def test_integration_with_real_dataset(self, small_real_compounds):
        all_fingerprints = []

        for smiles in small_real_compounds['SMILES'].tolist():
            fp = smiles_to_maccs_fingerprint(smiles)
            all_fingerprints.append(fp)

        all_fingerprints = np.array(all_fingerprints)
        assert all_fingerprints.shape[1] == 167
        assert all_fingerprints.dtype == np.uint8

    def test_similarity_between_similar_molecules(self):
        fp1 = smiles_to_maccs_fingerprint('CCO')
        fp2 = smiles_to_maccs_fingerprint('CCCO')

        similarity = np.sum(fp1 & fp2) / np.sum(fp1 | fp2)
        assert similarity > 0.5

    def test_dissimilarity_between_different_molecules(self):
        fp1 = smiles_to_maccs_fingerprint('CCO')
        fp2 = smiles_to_maccs_fingerprint('c1ccccc1')

        similarity = np.sum(fp1 & fp2) / np.sum(fp1 | fp2)
        assert similarity < 0.8

    def test_empty_string_handling(self):
        fp = smiles_to_maccs_fingerprint('')
        assert fp.shape == (167,)
        assert fp.dtype == np.uint8

    def test_none_input_raises_error(self):
        with pytest.raises((ValueError, AttributeError, TypeError)):
            smiles_to_maccs_fingerprint(None)

    def test_numeric_input_raises_error(self):
        with pytest.raises((ValueError, AttributeError, TypeError)):
            smiles_to_maccs_fingerprint(12345)

    def test_multiple_compounds_consistency(self, small_real_compounds):
        smiles_list = small_real_compounds['SMILES'].tolist()[:5]

        first_pass = [smiles_to_maccs_fingerprint(s) for s in smiles_list]
        second_pass = [smiles_to_maccs_fingerprint(s) for s in smiles_list]

        for fp1, fp2 in zip(first_pass, second_pass):
            np.testing.assert_array_equal(fp1, fp2)

    def test_copy_independence(self):
        fp1 = smiles_to_maccs_fingerprint('CCO')
        fp1_copy = fp1.copy()
        fp1_copy[0] = 1 - fp1_copy[0]

        assert not np.array_equal(fp1, fp1_copy)

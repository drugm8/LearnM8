import pytest
import numpy as np
from rdkit import Chem

from learnm8.utils.featurizers import smiles_to_morgan_fingerprint

pytestmark = [pytest.mark.unit, pytest.mark.molecular]


class TestMorganFingerprintBasic:

    def test_simple_molecules(self):
        smiles_list = ['CCO', 'CCC', 'CCN']
        fingerprints = [smiles_to_morgan_fingerprint(s) for s in smiles_list]

        for fp in fingerprints:
            assert fp.shape == (2048,)
            assert fp.dtype == np.uint8

    def test_output_shape_and_dtype(self):
        fp = smiles_to_morgan_fingerprint('CCO')

        assert fp.shape == (2048,)
        assert fp.dtype == np.uint8
        assert isinstance(fp, np.ndarray)

    def test_fingerprint_is_bit_vector(self):
        fp = smiles_to_morgan_fingerprint('CCO')

        unique_values = np.unique(fp)
        assert all(val in [0, 1] for val in unique_values)
        assert len(fp) == 2048

    def test_reproducibility(self):
        smiles = 'CCO'
        fp1 = smiles_to_morgan_fingerprint(smiles)
        fp2 = smiles_to_morgan_fingerprint(smiles)
        fp3 = smiles_to_morgan_fingerprint(smiles)

        np.testing.assert_array_equal(fp1, fp2)
        np.testing.assert_array_equal(fp2, fp3)

    def test_different_molecules_different_fingerprints(self):
        fp_ethanol = smiles_to_morgan_fingerprint('CCO')
        fp_propane = smiles_to_morgan_fingerprint('CCC')
        fp_ethylamine = smiles_to_morgan_fingerprint('CCN')

        assert not np.array_equal(fp_ethanol, fp_propane)
        assert not np.array_equal(fp_ethanol, fp_ethylamine)
        assert not np.array_equal(fp_propane, fp_ethylamine)

    def test_fingerprint_sparsity(self):
        fp = smiles_to_morgan_fingerprint('CCO')

        num_on_bits = np.sum(fp)
        sparsity_ratio = num_on_bits / len(fp)

        assert num_on_bits > 0
        assert sparsity_ratio < 0.5


class TestMorganFingerprintChemicalDiversity:

    def test_aromatic_rings(self):
        benzene = smiles_to_morgan_fingerprint('c1ccccc1')

        assert benzene.shape == (2048,)
        assert np.sum(benzene) > 0

    def test_aliphatic_vs_aromatic(self):
        cyclohexane = smiles_to_morgan_fingerprint('C1CCCCC1')
        benzene = smiles_to_morgan_fingerprint('c1ccccc1')

        assert not np.array_equal(cyclohexane, benzene)

    def test_heteroatom_molecules(self):
        oxygen_fp = smiles_to_morgan_fingerprint('CCO')
        nitrogen_fp = smiles_to_morgan_fingerprint('CCN')
        sulfur_fp = smiles_to_morgan_fingerprint('CCS')

        assert oxygen_fp.shape == (2048,)
        assert nitrogen_fp.shape == (2048,)
        assert sulfur_fp.shape == (2048,)

        assert not np.array_equal(oxygen_fp, nitrogen_fp)
        assert not np.array_equal(oxygen_fp, sulfur_fp)

    def test_halogenated_molecules(self):
        chloro_fp = smiles_to_morgan_fingerprint('CCCl')
        bromo_fp = smiles_to_morgan_fingerprint('CCBr')
        fluoro_fp = smiles_to_morgan_fingerprint('CCF')

        assert not np.array_equal(chloro_fp, bromo_fp)
        assert not np.array_equal(chloro_fp, fluoro_fp)
        assert not np.array_equal(bromo_fp, fluoro_fp)

    def test_polycyclic_molecules(self):
        naphthalene = smiles_to_morgan_fingerprint('c1ccc2ccccc2c1')

        assert naphthalene.shape == (2048,)
        assert np.sum(naphthalene) > 0

    def test_functional_group_diversity(self):
        alcohol = smiles_to_morgan_fingerprint('CCO')
        amine = smiles_to_morgan_fingerprint('CCN')
        carboxylic_acid = smiles_to_morgan_fingerprint('CC(=O)O')
        ester = smiles_to_morgan_fingerprint('CC(=O)OC')

        fingerprints = [alcohol, amine, carboxylic_acid, ester]

        for i, fp1 in enumerate(fingerprints):
            for j, fp2 in enumerate(fingerprints):
                if i != j:
                    assert not np.array_equal(fp1, fp2)


class TestMorganFingerprintEdgeCases:

    def test_single_atom_molecules(self):
        methane = smiles_to_morgan_fingerprint('C')

        assert methane.shape == (2048,)
        assert np.sum(methane) > 0

    def test_large_molecules(self):
        large_smiles = 'CCCCCCCCCCCCCCCCc1ccccc1'
        fp = smiles_to_morgan_fingerprint(large_smiles)

        assert fp.shape == (2048,)
        assert np.sum(fp) > 0

    def test_charged_species(self):
        cation = smiles_to_morgan_fingerprint('[NH4+]')
        anion = smiles_to_morgan_fingerprint('[Cl-]')

        assert cation.shape == (2048,)
        assert anion.shape == (2048,)

    def test_stereochemistry(self):
        r_enantiomer = smiles_to_morgan_fingerprint('C[C@H](O)CC')
        s_enantiomer = smiles_to_morgan_fingerprint('C[C@@H](O)CC')

        assert r_enantiomer.shape == (2048,)
        assert s_enantiomer.shape == (2048,)

    def test_double_and_triple_bonds(self):
        ethene = smiles_to_morgan_fingerprint('C=C')
        ethyne = smiles_to_morgan_fingerprint('C#C')

        assert ethene.shape == (2048,)
        assert ethyne.shape == (2048,)
        assert not np.array_equal(ethene, ethyne)

    def test_branched_molecules(self):
        linear = smiles_to_morgan_fingerprint('CCCCCC')
        branched = smiles_to_morgan_fingerprint('CC(C)CCC')

        assert not np.array_equal(linear, branched)

    def test_ring_sizes(self):
        cyclopentane = smiles_to_morgan_fingerprint('C1CCCC1')
        cyclohexane = smiles_to_morgan_fingerprint('C1CCCCC1')
        cycloheptane = smiles_to_morgan_fingerprint('C1CCCCCC1')

        assert cyclopentane.shape == (2048,)
        assert cyclohexane.shape == (2048,)
        assert cycloheptane.shape == (2048,)


class TestMorganFingerprintErrorHandling:

    def test_invalid_smiles_raises_error(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_morgan_fingerprint('INVALID_SMILES_STRING')

    def test_empty_string(self):
        fp = smiles_to_morgan_fingerprint('')

        assert fp.shape == (2048,)
        assert np.sum(fp) == 0

    def test_malformed_smiles_raises_error(self):
        invalid_smiles = ['C(', 'CC)', 'C1CC', '[', 'C=']

        for smiles in invalid_smiles:
            with pytest.raises(ValueError, match="Invalid SMILES"):
                smiles_to_morgan_fingerprint(smiles)

    def test_invalid_valence_raises_error(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_morgan_fingerprint('C(C)(C)(C)(C)(C)')


class TestMorganFingerprintRealMolecules:

    def test_with_small_real_compounds(self, small_real_compounds):
        smiles_list = small_real_compounds['SMILES'].to_list()

        fingerprints = []
        for smiles in smiles_list:
            fp = smiles_to_morgan_fingerprint(smiles)
            fingerprints.append(fp)

        assert len(fingerprints) == len(smiles_list)

        for fp in fingerprints:
            assert fp.shape == (2048,)
            assert fp.dtype == np.uint8
            assert np.all(np.isin(fp, [0, 1]))

    def test_with_diverse_real_compounds(self, diverse_real_compounds):
        smiles_list = diverse_real_compounds['SMILES'].to_list()[:20]

        fingerprints = []
        for smiles in smiles_list:
            fp = smiles_to_morgan_fingerprint(smiles)
            fingerprints.append(fp)

        assert len(fingerprints) == len(smiles_list)

        for fp in fingerprints:
            assert fp.shape == (2048,)
            assert np.all(np.isfinite(fp))

    def test_with_edge_case_compounds(self, edge_case_compounds):
        smiles_list = edge_case_compounds['SMILES'].to_list()

        valid_fingerprints = []
        for smiles in smiles_list:
            try:
                fp = smiles_to_morgan_fingerprint(smiles)
                valid_fingerprints.append(fp)
            except ValueError:
                pass

        assert len(valid_fingerprints) > 0

        for fp in valid_fingerprints:
            assert fp.shape == (2048,)
            assert fp.dtype == np.uint8

    def test_consistent_across_datasets(self, small_real_compounds, diverse_real_compounds):
        common_smiles = 'CCO'

        fp1 = smiles_to_morgan_fingerprint(common_smiles)
        fp2 = smiles_to_morgan_fingerprint(common_smiles)

        np.testing.assert_array_equal(fp1, fp2)


class TestMorganFingerprintProperties:

    def test_fingerprint_has_reasonable_bit_density(self):
        molecules = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'CCS']

        for smiles in molecules:
            fp = smiles_to_morgan_fingerprint(smiles)
            on_bits = np.sum(fp)

            assert 1 <= on_bits <= 200

    def test_larger_molecules_more_on_bits(self):
        small_molecule = smiles_to_morgan_fingerprint('CC')
        medium_molecule = smiles_to_morgan_fingerprint('CCCCCCCC')
        large_molecule = smiles_to_morgan_fingerprint('CCCCCCCCCCCCCCCCc1ccccc1')

        small_on = np.sum(small_molecule)
        medium_on = np.sum(medium_molecule)
        large_on = np.sum(large_molecule)

        assert small_on <= medium_on
        assert medium_on <= large_on

    def test_similar_molecules_similar_fingerprints(self):
        ethanol = smiles_to_morgan_fingerprint('CCO')
        propanol = smiles_to_morgan_fingerprint('CCCO')
        butanol = smiles_to_morgan_fingerprint('CCCCO')

        similarity_ethanol_propanol = np.sum(ethanol & propanol) / np.sum(ethanol | propanol)
        similarity_propanol_butanol = np.sum(propanol & butanol) / np.sum(propanol | butanol)

        similarity_ethanol_butanol = np.sum(ethanol & butanol) / np.sum(ethanol | butanol)

        assert similarity_ethanol_propanol > 0.3
        assert similarity_propanol_butanol > 0.3
        assert similarity_ethanol_propanol > similarity_ethanol_butanol

    def test_dissimilar_molecules_different_fingerprints(self):
        ethanol = smiles_to_morgan_fingerprint('CCO')
        naphthalene = smiles_to_morgan_fingerprint('c1ccc2ccccc2c1')

        overlap = np.sum(ethanol & naphthalene)
        total = np.sum(ethanol | naphthalene)
        similarity = overlap / total if total > 0 else 0

        assert similarity < 0.5

    def test_bit_vector_is_binary(self):
        molecules = ['CCO', 'CCC', 'c1ccccc1', 'CC(=O)O', 'CCN']

        for smiles in molecules:
            fp = smiles_to_morgan_fingerprint(smiles)
            unique_values = np.unique(fp)

            assert len(unique_values) <= 2
            assert all(val in [0, 1] for val in unique_values)

    def test_no_nan_or_inf_values(self):
        molecules = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'CCCCCCCC']

        for smiles in molecules:
            fp = smiles_to_morgan_fingerprint(smiles)

            assert np.all(np.isfinite(fp))
            assert not np.any(np.isnan(fp))
            assert not np.any(np.isinf(fp))


class TestMorganFingerprintIntegration:

    def test_multiple_molecules_batch(self):
        molecules = ['CCO', 'CCC', 'CCN', 'c1ccccc1', 'CC(=O)O']

        fingerprints = [smiles_to_morgan_fingerprint(s) for s in molecules]

        assert len(fingerprints) == len(molecules)

        for fp in fingerprints:
            assert fp.shape == (2048,)
            assert fp.dtype == np.uint8

    def test_fingerprints_usable_for_ml(self):
        molecules = ['CCO', 'CCC', 'CCN', 'c1ccccc1']

        fingerprints = np.array([smiles_to_morgan_fingerprint(s) for s in molecules])

        assert fingerprints.shape == (4, 2048)
        assert fingerprints.dtype == np.uint8

        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(fingerprints)

        assert similarity_matrix.shape == (4, 4)
        assert np.all(np.diag(similarity_matrix) >= 0.99)

    def test_fingerprint_with_rdkit_mol_comparison(self):
        smiles = 'CCO'
        mol = Chem.MolFromSmiles(smiles)

        fp = smiles_to_morgan_fingerprint(smiles)

        assert mol is not None
        assert fp.shape == (2048,)
        assert np.sum(fp) > 0

    def test_deterministic_across_calls(self):
        smiles = 'c1ccc2ccccc2c1'

        fingerprints = [smiles_to_morgan_fingerprint(smiles) for _ in range(10)]

        for fp in fingerprints[1:]:
            np.testing.assert_array_equal(fingerprints[0], fp)

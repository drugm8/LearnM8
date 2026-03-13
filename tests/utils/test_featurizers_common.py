import pytest
import numpy as np
from numpy.testing import assert_array_equal

from learnm8.utils.featurizers import (
    smiles_to_ecfp6_fingerprint,
    smiles_to_maccs_fingerprint,
    smiles_to_morgan_feature_fingerprint,
    smiles_to_morgan_fingerprint,
)

pytestmark = [pytest.mark.unit, pytest.mark.molecular]

FEATURIZER_CONFIGS = [
    pytest.param(smiles_to_morgan_fingerprint, 2048, np.uint8, id="morgan"),
    pytest.param(smiles_to_ecfp6_fingerprint, 2048, np.uint8, id="ecfp6"),
    pytest.param(smiles_to_maccs_fingerprint, 167, np.uint8, id="maccs"),
    pytest.param(smiles_to_morgan_feature_fingerprint, 2048, None, id="morgan_feat"),
]


@pytest.mark.parametrize("func,expected_length,expected_dtype", FEATURIZER_CONFIGS)
class TestFeaturizerCommon:

    def test_basic_call(self, func, expected_length, expected_dtype):
        fp = func("CCO")
        assert isinstance(fp, np.ndarray)
        assert len(fp) > 0

    def test_output_shape(self, func, expected_length, expected_dtype):
        fp = func("CCO")
        assert fp.shape == (expected_length,)

    def test_output_dtype(self, func, expected_length, expected_dtype):
        if expected_dtype is None:
            pytest.skip("No specific dtype requirement for this featurizer")
        fp = func("CCO")
        assert fp.dtype == expected_dtype

    def test_output_is_finite(self, func, expected_length, expected_dtype):
        fp = func("CCO")
        assert np.all(np.isfinite(fp))
        assert not np.any(np.isnan(fp))
        assert not np.any(np.isinf(fp))

    def test_reproducibility(self, func, expected_length, expected_dtype):
        fp1 = func("CCO")
        fp2 = func("CCO")
        assert_array_equal(fp1, fp2)

    def test_different_molecules_different_output(self, func, expected_length, expected_dtype):
        fp1 = func("CCO")
        fp2 = func("CCC")
        assert not np.array_equal(fp1, fp2)

    def test_invalid_smiles_raises_error(self, func, expected_length, expected_dtype):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            func("INVALID_SMILES_STRING")

    def test_multiple_molecules(self, func, expected_length, expected_dtype):
        for smiles in ["CCO", "CCC", "CCN", "c1ccccc1"]:
            fp = func(smiles)
            assert isinstance(fp, np.ndarray)
            assert fp.shape == (expected_length,)

    def test_real_compounds(self, func, expected_length, expected_dtype, small_real_compounds):
        smiles_list = small_real_compounds["SMILES"].to_list()[:5]
        for smiles in smiles_list:
            fp = func(smiles)
            assert isinstance(fp, np.ndarray)
            assert fp.shape == (expected_length,)
            assert np.all(np.isfinite(fp))


class TestMaccsSpecific:

    def test_maccs_167_bits(self):
        fp = smiles_to_maccs_fingerprint("CCO")
        assert fp.shape == (167,)


class TestMorganSpecific:

    def test_morgan_bit_vector_is_binary(self):
        molecules = ["CCO", "CCC", "c1ccccc1", "CC(=O)O", "CCN"]
        for smiles in molecules:
            fp = smiles_to_morgan_fingerprint(smiles)
            unique_values = np.unique(fp)
            assert all(val in [0, 1] for val in unique_values)


class TestEcfp6Specific:

    def test_ecfp6_larger_radius_vs_morgan(self):
        smiles = "c1ccc(cc1)C(=O)O"
        fp_morgan = smiles_to_morgan_fingerprint(smiles)
        fp_ecfp6 = smiles_to_ecfp6_fingerprint(smiles)
        assert not np.array_equal(fp_morgan, fp_ecfp6)

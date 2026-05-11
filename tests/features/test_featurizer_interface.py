"""Test Featurizer interface compliance and base functionality."""

from abc import ABC

import numpy as np
import pytest

from learnm8.core.interfaces import Featurizer


class MinimalFeaturizer(Featurizer):
    """Minimal concrete featurizer for interface testing."""

    def transform(self, smiles_list):
        return np.random.rand(len(smiles_list), 10).astype(np.float32)

    def get_dimension(self):
        return 10

    def get_name(self):
        return 'minimal'


class CustomConfigFeaturizer(Featurizer):
    """Featurizer with custom configuration for cache testing."""

    def __init__(self, radius=2, fp_size=2048):
        self.radius = radius
        self.fp_size = fp_size

    def transform(self, smiles_list):
        return np.random.rand(len(smiles_list), self.fp_size).astype(np.float32)

    def get_dimension(self):
        return self.fp_size

    def get_name(self):
        return f'custom_r{self.radius}'

    def get_config(self):
        return {'radius': self.radius, 'fp_size': self.fp_size}


class Requires3DFeaturizer(Featurizer):
    """Featurizer that requires 3D conformers."""

    def transform(self, smiles_list):
        return np.random.rand(len(smiles_list), 12).astype(np.float32)

    def get_dimension(self):
        return 12

    def get_name(self):
        return 'usr_mock'

    def requires_3d(self):
        return True


@pytest.mark.unit
class TestFeaturizerInterface:
    """Test Featurizer ABC interface compliance."""

    def test_featurizer_is_abstract_base_class(self):
        """Verify Featurizer is an ABC."""
        assert issubclass(Featurizer, ABC)

    def test_cannot_instantiate_base_featurizer(self):
        """Cannot instantiate abstract Featurizer directly."""
        with pytest.raises(TypeError, match='abstract'):
            Featurizer()

    def test_minimal_featurizer_implements_required_methods(self):
        """Minimal implementation with all required methods works."""
        featurizer = MinimalFeaturizer()
        smiles = ['CCO', 'CCC', 'CCN']

        features = featurizer.transform(smiles)
        assert features.shape == (3, 10)
        assert featurizer.get_dimension() == 10
        assert featurizer.get_name() == 'minimal'

    def test_transform_method_required(self):
        """Featurizer must implement transform()."""

        class NoTransform(Featurizer):
            def get_dimension(self):
                return 10

            def get_name(self):
                return 'test'

        with pytest.raises(TypeError, match='abstract'):
            NoTransform()

    def test_get_dimension_method_required(self):
        """Featurizer must implement get_dimension()."""

        class NoDimension(Featurizer):
            def transform(self, smiles_list):
                return np.array([])

            def get_name(self):
                return 'test'

        with pytest.raises(TypeError, match='abstract'):
            NoDimension()

    def test_get_name_method_required(self):
        """Featurizer must implement get_name()."""

        class NoName(Featurizer):
            def transform(self, smiles_list):
                return np.array([])

            def get_dimension(self):
                return 10

        with pytest.raises(TypeError, match='abstract'):
            NoName()


@pytest.mark.unit
class TestFeaturizerDefaultMethods:
    """Test default method implementations."""

    def test_requires_3d_defaults_to_false(self):
        """Default requires_3d() returns False."""
        featurizer = MinimalFeaturizer()
        assert featurizer.requires_3d() is False

    def test_supports_caching_defaults_to_true(self):
        """Default supports_caching() returns True."""
        featurizer = MinimalFeaturizer()
        assert featurizer.supports_caching() is True

    def test_supports_batching_defaults_to_true(self):
        """Default supports_batching() returns True."""
        featurizer = MinimalFeaturizer()
        assert featurizer.supports_batching() is True

    def test_get_description_defaults_to_name(self):
        """Default get_description() returns get_name()."""
        featurizer = MinimalFeaturizer()
        assert featurizer.get_description() == featurizer.get_name()

    def test_get_config_defaults_to_empty_dict(self):
        """Default get_config() returns empty dict."""
        featurizer = MinimalFeaturizer()
        assert featurizer.get_config() == {}


@pytest.mark.unit
class TestFeaturizerConfigurationHash:
    """Test get_config_hash() for cache key generation."""

    def test_get_config_hash_with_empty_config(self):
        """Hash of empty config is deterministic."""
        featurizer = MinimalFeaturizer()
        hash1 = featurizer.get_config_hash()
        hash2 = featurizer.get_config_hash()
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hex digest length

    def test_get_config_hash_with_parameters(self):
        """Hash of config with parameters is deterministic."""
        featurizer = CustomConfigFeaturizer(radius=3, fp_size=4096)
        hash1 = featurizer.get_config_hash()
        hash2 = featurizer.get_config_hash()
        assert hash1 == hash2

    def test_different_configs_produce_different_hashes(self):
        """Different configurations produce different hashes."""
        feat1 = CustomConfigFeaturizer(radius=2, fp_size=2048)
        feat2 = CustomConfigFeaturizer(radius=3, fp_size=2048)
        feat3 = CustomConfigFeaturizer(radius=2, fp_size=4096)

        hash1 = feat1.get_config_hash()
        hash2 = feat2.get_config_hash()
        hash3 = feat3.get_config_hash()

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_same_configs_produce_same_hashes(self):
        """Identical configurations produce identical hashes."""
        feat1 = CustomConfigFeaturizer(radius=3, fp_size=4096)
        feat2 = CustomConfigFeaturizer(radius=3, fp_size=4096)

        assert feat1.get_config_hash() == feat2.get_config_hash()

    def test_config_hash_includes_name(self):
        """Config hash includes featurizer name to prevent collisions."""

        class Featurizer1(MinimalFeaturizer):
            def get_name(self):
                return 'feat1'

        class Featurizer2(MinimalFeaturizer):
            def get_name(self):
                return 'feat2'

        assert Featurizer1().get_config_hash() != Featurizer2().get_config_hash()


@pytest.mark.unit
class TestFeaturizerValidation:
    """Test validate_smiles() method."""

    def test_validate_smiles_with_valid_smiles(self, small_real_compounds):
        """Validation succeeds for valid SMILES."""
        from learnm8.features import create_featurizer

        featurizer = create_featurizer('morgan')

    def test_validate_smiles_filters_invalid(self):
        """Validation identifies invalid SMILES."""
        from learnm8.features import create_featurizer

        featurizer = create_featurizer('morgan')

    def test_validate_smiles_empty_list(self):
        """Validation of empty list returns empty list."""
        from learnm8.features import create_featurizer

        featurizer = create_featurizer('morgan')

    def test_validate_smiles_all_invalid(self):
        """Validation of all invalid SMILES returns all False."""
        from learnm8.features import create_featurizer

        featurizer = create_featurizer('morgan')
        smiles_list = ['CCO', 'INVALID_SMILES', 'CCC', 'NOT@VALID', 'CCN']

        validated = featurizer.validate_smiles(smiles_list)
        assert len(validated) == 5
        assert validated[0] is True  # CCO
        assert validated[1] is False  # INVALID_SMILES
        assert validated[2] is True  # CCC
        assert validated[3] is False  # NOT@VALID
        assert validated[4] is True  # CCN

    def test_validate_smiles_empty_list(self):
        """Validation of empty list returns empty list."""
        from learnm8.features import create_featurizer

        featurizer = create_featurizer('morgan')
        assert featurizer.validate_smiles([]) == []

    def test_validate_smiles_all_invalid(self):
        """Validation of all invalid SMILES returns all False."""
        from learnm8.features import create_featurizer

        featurizer = create_featurizer('morgan')
        invalid_smiles = ['INVALID1', 'INVALID2', 'NOT@VALID']

        validated = featurizer.validate_smiles(invalid_smiles)
        assert len(validated) == 3
        assert not any(validated), 'All SMILES should be invalid'


@pytest.mark.unit
class TestFeaturizer3DRequirements:
    """Test 3D conformer requirement handling."""

    def test_2d_featurizer_does_not_require_3d(self):
        """2D featurizers do not require conformers."""
        featurizer = MinimalFeaturizer()
        assert featurizer.requires_3d() is False

    def test_3d_featurizer_requires_3d(self):
        """3D featurizers require conformers."""
        featurizer = Requires3DFeaturizer()
        assert featurizer.requires_3d() is True

    def test_override_requires_3d(self):
        """Subclasses can override requires_3d()."""

        class CustomFeaturizer(MinimalFeaturizer):
            def requires_3d(self):
                return True

        assert CustomFeaturizer().requires_3d() is True


@pytest.mark.unit
class TestFeaturizerOutputFormat:
    """Test output format requirements."""

    def test_transform_returns_numpy_array(self):
        """transform() must return numpy array."""
        featurizer = MinimalFeaturizer()
        result = featurizer.transform(['CCO', 'CCC'])
        assert isinstance(result, np.ndarray)

    def test_transform_returns_float32(self):
        """transform() should return float32 for memory efficiency."""
        featurizer = MinimalFeaturizer()
        result = featurizer.transform(['CCO', 'CCC'])
        assert result.dtype == np.float32

    def test_transform_returns_correct_shape(self):
        """transform() returns (n_compounds, n_features) shape."""
        featurizer = MinimalFeaturizer()
        smiles = ['CCO', 'CCC', 'CCN']
        result = featurizer.transform(smiles)

        assert result.shape == (3, 10)
        assert result.shape[0] == len(smiles)
        assert result.shape[1] == featurizer.get_dimension()

    def test_transform_empty_input(self):
        """transform() handles empty input gracefully."""
        featurizer = MinimalFeaturizer()
        result = featurizer.transform([])
        assert result.shape == (0, 10)


@pytest.mark.unit
class TestFeaturizerCacheability:
    """Test cache support flags."""

    def test_deterministic_featurizer_supports_caching(self):
        """Deterministic featurizers support caching."""
        featurizer = MinimalFeaturizer()
        assert featurizer.supports_caching() is True

    def test_can_disable_caching(self):
        """Featurizers can disable caching if needed."""

        class NonDeterministicFeaturizer(MinimalFeaturizer):
            def supports_caching(self):
                return False

        assert NonDeterministicFeaturizer().supports_caching() is False

"""Test featurizer integration with API and registry."""

import pytest
import numpy as np
from pathlib import Path
from learnm8.features.extraction import extract_features
from learnm8.features import create_featurizer, FEATURIZER_REGISTRY
from learnm8 import run_active_learning


@pytest.mark.integration
class TestFeaturizerRegistry:
    """Test FEATURIZER_REGISTRY."""

    def test_registry_contains_all_implemented_featurizers(self):
        """Registry contains all 11 implemented featurizers."""
        expected_featurizers = {
            'morgan',
            'maccs',
            'avalon',
            'atom_pair',
            'topological_torsion',
            'rdkit',
            'pubchem',
            'mordred',
            'whim',
            'usr',
            'e3fp',
        }

        registry_keys = set(FEATURIZER_REGISTRY)
        assert expected_featurizers.issubset(registry_keys)

    def test_registry_values_are_callable(self):
        """All registry entries produce valid featurizer instances."""
        for name in FEATURIZER_REGISTRY:
            featurizer = create_featurizer(name)
            assert featurizer is not None

    def test_registry_creates_valid_instances(self):
        """All registry entries create valid featurizer instances."""
        for name in FEATURIZER_REGISTRY:
            featurizer = create_featurizer(name)

            assert hasattr(featurizer, 'transform')
            assert hasattr(featurizer, 'get_dimension')
            assert hasattr(featurizer, 'get_name')


@pytest.mark.integration
class TestExtractFeaturesAPI:
    """Test extract_features() function integration."""

    def test_extract_features_with_string_featurizer(
        self, small_real_compounds, tmp_path
    ):
        """extract_features() works with string featurizer name."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features = extract_features(smiles, 'morgan', tmp_path, n_jobs=1)

        assert features.shape == (10, 2048)
        assert features.dtype == np.float32

    def test_extract_features_with_featurizer_instance(
        self, small_real_compounds, tmp_path
    ):
        """extract_features() works with Featurizer instance."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]
        featurizer = create_featurizer('morgan', radius=3, fp_size=4096)

        features = extract_features(smiles, featurizer, tmp_path, n_jobs=1)

        assert features.shape == (10, 4096)

    def test_extract_features_caching_integration(self, small_real_compounds, tmp_path):
        """extract_features() correctly uses HDF5 caching."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features1 = extract_features(smiles, 'morgan', tmp_path, n_jobs=1)
        features2 = extract_features(smiles, 'morgan', tmp_path, n_jobs=1)

        assert np.array_equal(features1, features2)

        cache_files = list(tmp_path.glob('*.h5'))
        assert len(cache_files) == 1

    @pytest.mark.slow
    def test_extract_features_with_different_n_jobs(
        self, small_real_compounds, tmp_path
    ):
        """extract_features() works with different n_jobs settings."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features_sequential = extract_features(smiles, 'morgan', tmp_path, n_jobs=1)
        features_parallel = extract_features(smiles, 'morgan', tmp_path, n_jobs=2)

        assert np.array_equal(features_sequential, features_parallel)

    @pytest.mark.slow
    def test_extract_features_with_3d_featurizer(self, small_real_compounds, tmp_path):
        """extract_features() works with 3D featurizers."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        features = extract_features(smiles, 'usr', tmp_path, n_jobs=1)

        assert features.shape == (5, 12)
        assert np.all(np.isfinite(features))


@pytest.mark.integration
@pytest.mark.molecular
@pytest.mark.slow
class TestRunActiveLearningWithFeaturizers:
    """Test run_active_learning() with different featurizers."""

    def test_run_active_learning_with_string_featurizer(
        self, small_real_compounds, tmp_path
    ):
        """run_active_learning() accepts string featurizer."""
        compounds = small_real_compounds.head(20).clone()
        compounds = compounds.with_columns(
            [compounds.get_column('Activity').alias('target')]
        )

        results = run_active_learning(
            compound_pool=compounds,
            oracle=compounds,
            learner='rf',
            featurizer='morgan',
            target_col='target',
            n_cycles=2,
            batch_fraction=0.2,
            cache_dir=tmp_path,
            output_dir=tmp_path / 'results',
            random_state=42,
        )

        assert 'compounds_df' in results
        assert len(results['cycle_metrics']) == 2

    def test_run_active_learning_with_featurizer_instance(
        self, small_real_compounds, tmp_path
    ):
        """run_active_learning() accepts Featurizer instance."""
        compounds = small_real_compounds.head(20).clone()
        compounds = compounds.with_columns(
            [compounds.get_column('Activity').alias('target')]
        )

        featurizer = create_featurizer('morgan', radius=3, fp_size=4096)

        results = run_active_learning(
            compound_pool=compounds,
            oracle=compounds,
            learner='rf',
            featurizer=featurizer,
            target_col='target',
            n_cycles=2,
            batch_fraction=0.2,
            cache_dir=tmp_path,
            output_dir=tmp_path / 'results',
            random_state=42,
        )

        assert 'compounds_df' in results
        assert len(results['cycle_metrics']) == 2

    def test_run_active_learning_with_maccs(self, small_real_compounds, tmp_path):
        """run_active_learning() works with MACCS featurizer."""
        compounds = small_real_compounds.head(20).clone()
        compounds = compounds.with_columns(
            [compounds.get_column('Activity').alias('target')]
        )

        results = run_active_learning(
            compound_pool=compounds,
            oracle=compounds,
            learner='rf',
            featurizer='maccs',
            target_col='target',
            n_cycles=2,
            batch_fraction=0.2,
            cache_dir=tmp_path,
            output_dir=tmp_path / 'results',
            random_state=42,
        )

        assert 'compounds_df' in results

    def test_run_active_learning_caches_features(self, small_real_compounds, tmp_path):
        """run_active_learning() creates HDF5 cache files."""
        compounds = small_real_compounds.head(20).clone()
        compounds = compounds.with_columns(
            [compounds.get_column('Activity').alias('target')]
        )

        cache_dir = tmp_path / 'cache'

        run_active_learning(
            compound_pool=compounds,
            oracle=compounds,
            learner='rf',
            featurizer='morgan',
            target_col='target',
            n_cycles=2,
            batch_fraction=0.2,
            cache_dir=cache_dir,
            output_dir=tmp_path / 'results',
            random_state=42,
        )

        assert cache_dir.exists()
        cache_files = list(cache_dir.glob('*.h5'))
        assert len(cache_files) >= 1


@pytest.mark.integration
class TestFeaturizerStringVsInstance:
    """Test equivalence between string and instance usage."""

    @pytest.mark.slow
    def test_string_and_instance_produce_same_features(
        self, small_real_compounds, tmp_path
    ):
        """String name and instance with defaults produce same features."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features_string = extract_features(smiles, 'morgan', tmp_path, n_jobs=1)

        featurizer_instance = create_featurizer('morgan')
        features_instance = extract_features(
            smiles, featurizer_instance, tmp_path, n_jobs=1
        )

        assert np.array_equal(features_string, features_instance)

    def test_custom_instance_differs_from_default(self, small_real_compounds, tmp_path):
        """Custom instance parameters produce different features."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features_default = extract_features(smiles, 'morgan', tmp_path, n_jobs=1)

        featurizer_custom = create_featurizer('morgan', radius=3)
        features_custom = extract_features(
            smiles, featurizer_custom, tmp_path, n_jobs=1
        )

        assert not np.array_equal(features_default, features_custom)


@pytest.mark.integration
class TestFeaturizerErrorHandling:
    """Test error handling in featurizer integration."""

    def test_extract_features_with_invalid_featurizer_string(
        self, small_real_compounds, tmp_path
    ):
        """extract_features() raises error for invalid string."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        with pytest.raises(ValueError, match='Unknown featurizer'):
            extract_features(smiles, 'invalid_name', tmp_path, n_jobs=1)

    def test_extract_features_with_none_featurizer(
        self, small_real_compounds, tmp_path
    ):
        """extract_features() raises error for None featurizer."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        with pytest.raises((ValueError, TypeError)):
            extract_features(smiles, None, tmp_path, n_jobs=1)

    def test_run_active_learning_with_invalid_featurizer(
        self, small_real_compounds, tmp_path
    ):
        """run_active_learning() raises error for invalid featurizer."""
        compounds = small_real_compounds.clone()
        compounds = compounds.with_columns(
            [compounds.get_column('Activity').alias('target')]
        )

        with pytest.raises(ValueError):
            run_active_learning(
                compound_pool=compounds,
                oracle=compounds,
                learner='rf',
                featurizer='nonexistent_featurizer',
                target_col='target',
                n_cycles=2,
                batch_fraction=0.1,
                cache_dir=tmp_path,
                output_dir=tmp_path / 'results',
                random_state=42,
            )

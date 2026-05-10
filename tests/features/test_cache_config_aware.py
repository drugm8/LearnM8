"""Test configuration-aware HDF5 caching for featurizers."""


import numpy as np
import pytest

from learnm8.features.extraction import extract_features
from learnm8.features.skfp_2d.morgan import MorganFeaturizer
from learnm8.features.skfp_3d.usr import USRFeaturizer


@pytest.mark.integration
class TestConfigurationAwareCaching:
    """Test cache keys include featurizer configuration hash."""

    def test_different_configs_use_different_cache_keys(self, small_real_compounds, tmp_path):
        """Different featurizer radii share the v3 cache file with distinct 128-bit hash keys."""
        import h5py

        from learnm8.features.cache import _cache_keys_bytes16

        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]
        test_smiles = smiles[0]

        featurizer1 = MorganFeaturizer(radius=2)
        featurizer2 = MorganFeaturizer(radius=3)

        features1 = extract_features(
            smiles, featurizer=featurizer1, cache_dir=tmp_path, n_jobs=1
        )
        features2 = extract_features(
            smiles, featurizer=featurizer2, cache_dir=tmp_path, n_jobs=1
        )

        assert not np.array_equal(features1, features2)

        cache_files = list(tmp_path.glob("*.h5"))
        assert len(cache_files) == 1

        cache_file = cache_files[0]
        with h5py.File(cache_file, 'r') as h5f:
            hash_index = h5f['hash_index'][:]
            key1 = _cache_keys_bytes16([test_smiles], featurizer1)[0]
            key2 = _cache_keys_bytes16([test_smiles], featurizer2)[0]
            assert key1 != key2
            assert key1 in hash_index
            assert key2 in hash_index

    def test_same_config_reuses_cache(self, small_real_compounds, tmp_path):
        """Identical configs reuse same cache entry."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features1 = extract_features(
            smiles,
            featurizer=MorganFeaturizer(radius=2),
            cache_dir=tmp_path,
            n_jobs=1
        )

        features2 = extract_features(
            smiles,
            featurizer=MorganFeaturizer(radius=2),
            cache_dir=tmp_path,
            n_jobs=1
        )

        assert np.array_equal(features1, features2)

        cache_files = list(tmp_path.glob("*.h5"))
        assert len(cache_files) == 1

    def test_cache_key_includes_featurizer_name(self, small_real_compounds, tmp_path):
        """Cache keys include featurizer name to prevent collisions."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        extract_features(
            smiles,
            featurizer=MorganFeaturizer(),
            cache_dir=tmp_path,
            n_jobs=1
        )

        extract_features(
            smiles,
            featurizer=USRFeaturizer(),
            cache_dir=tmp_path,
            n_jobs=1
        )

        cache_files = list(tmp_path.glob("*.h5"))
        assert len(cache_files) == 2

        cache_names = [f.stem for f in cache_files]
        assert any('morgan' in name.lower() for name in cache_names)
        assert any('usr' in name.lower() for name in cache_names)


@pytest.mark.integration
class TestCacheHitMiss:
    """Test cache hit and miss scenarios."""

    def test_cache_hit_on_second_call(self, small_real_compounds, tmp_path):
        """Second call with same config hits cache."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]
        featurizer = MorganFeaturizer(radius=2, fp_size=2048)

        features1 = extract_features(smiles, featurizer, cache_dir=tmp_path, n_jobs=1)
        features2 = extract_features(smiles, featurizer, cache_dir=tmp_path, n_jobs=1)

        assert np.array_equal(features1, features2)

    def test_cache_miss_on_config_change(self, small_real_compounds, tmp_path):
        """Changing config causes cache miss."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features1 = extract_features(
            smiles,
            featurizer=MorganFeaturizer(radius=2),
            cache_dir=tmp_path,
            n_jobs=1
        )

        features2 = extract_features(
            smiles,
            featurizer=MorganFeaturizer(radius=3),
            cache_dir=tmp_path,
            n_jobs=1
        )

        assert not np.array_equal(features1, features2)

    def test_cache_miss_on_different_featurizer(self, small_real_compounds, tmp_path):
        """Different featurizer causes cache miss."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        features1 = extract_features(
            smiles,
            featurizer=MorganFeaturizer(),
            cache_dir=tmp_path,
            n_jobs=1
        )

        features2 = extract_features(
            smiles,
            featurizer=USRFeaturizer(),
            cache_dir=tmp_path,
            n_jobs=1
        )

        assert features1.shape[1] != features2.shape[1]


@pytest.mark.integration
class TestCacheWithDifferentParameters:
    """Test caching with various parameter combinations."""

    def test_cache_fp_size_variations(self, small_real_compounds, tmp_path):
        """Different fp_size = different bit_count = old v2 file gets renamed."""
        import h5py

        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        feat1 = MorganFeaturizer(fp_size=2048)
        feat2 = MorganFeaturizer(fp_size=4096)

        feat_2048 = extract_features(
            smiles, featurizer=feat1, cache_dir=tmp_path, n_jobs=1
        )
        feat_4096 = extract_features(
            smiles, featurizer=feat2, cache_dir=tmp_path, n_jobs=1
        )

        assert feat_2048.shape[1] == 2048
        assert feat_4096.shape[1] == 4096

        # Active cache reflects the latest bit_count; previous file lives at .dim<N>.bak
        active = tmp_path / 'features_morgan.h5'
        backup = tmp_path / 'features_morgan.h5.dim2048.bak'
        assert active.exists()
        assert backup.exists()

        with h5py.File(active, 'r') as h5f:
            assert int(h5f.attrs['bit_count']) == 4096

    def test_cache_3d_conformer_params(self, small_real_compounds, tmp_path):
        """Different USR conformer params share file with distinct 128-bit hash keys."""
        import h5py

        from learnm8.features.cache import _cache_keys_bytes16

        smiles = small_real_compounds.get_column('SMILES').to_list()[:3]
        test_smiles = smiles[0]

        feat1 = USRFeaturizer(num_conformers=1)
        feat2 = USRFeaturizer(num_conformers=2, optimize_force_field='UFF')

        extract_features(smiles, featurizer=feat1, cache_dir=tmp_path, n_jobs=1)
        extract_features(smiles, featurizer=feat2, cache_dir=tmp_path, n_jobs=1)

        cache_files = list(tmp_path.glob("*.h5"))
        assert len(cache_files) == 1

        cache_file = cache_files[0]
        with h5py.File(cache_file, 'r') as h5f:
            hash_index = h5f['hash_index'][:]
            key1 = _cache_keys_bytes16([test_smiles], feat1)[0]
            key2 = _cache_keys_bytes16([test_smiles], feat2)[0]
            assert key1 != key2
            assert key1 in hash_index
            assert key2 in hash_index


@pytest.mark.integration
class TestCacheFileStructure:
    """Test HDF5 cache file structure and naming."""

    def test_cache_file_naming_convention(self, small_real_compounds, tmp_path):
        """Cache files follow naming convention (features_<featurizer_name>.h5)."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]
        featurizer = MorganFeaturizer(radius=2, fp_size=2048)

        extract_features(smiles, featurizer, cache_dir=tmp_path, n_jobs=1)

        cache_files = list(tmp_path.glob("*.h5"))
        assert len(cache_files) == 1

        cache_file = cache_files[0]
        # Format: features_<featurizer_name>.h5
        assert cache_file.name == 'features_morgan.h5'

    def test_multiple_featurizers_separate_files(self, small_real_compounds, tmp_path):
        """Same-name featurizer configs with same bit_count share file; different
        featurizer types or different bit_counts create separate active files."""
        from learnm8.features.skfp_2d.maccs import MACCSFeaturizer
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        # Same bit_count (radius differs only) → single file
        extract_features(smiles, MorganFeaturizer(radius=2), tmp_path, n_jobs=1)
        extract_features(smiles, MorganFeaturizer(radius=3), tmp_path, n_jobs=1)

        cache_files = list(tmp_path.glob("*.h5"))
        assert len(cache_files) == 1
        assert cache_files[0].name == 'features_morgan.h5'

        # Different featurizer type creates separate file
        extract_features(smiles, MACCSFeaturizer(), tmp_path, n_jobs=1)
        cache_files = list(tmp_path.glob("*.h5"))
        assert len(cache_files) == 2
        file_names = {f.name for f in cache_files}
        assert file_names == {'features_morgan.h5', 'features_maccs.h5'}


@pytest.mark.integration
class TestCachePersistence:
    """Test cache persistence across sessions."""

    def test_cache_persists_across_calls(self, small_real_compounds, tmp_path):
        """Cache persists and is reused across multiple calls."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]
        featurizer = MorganFeaturizer(radius=2)

        features1 = extract_features(smiles, featurizer, tmp_path, n_jobs=1)
        features2 = extract_features(smiles, featurizer, tmp_path, n_jobs=1)
        features3 = extract_features(smiles, featurizer, tmp_path, n_jobs=1)

        assert np.array_equal(features1, features2)
        assert np.array_equal(features2, features3)

        cache_files = list(tmp_path.glob("*.h5"))
        assert len(cache_files) == 1

    def test_cache_handles_subset_of_compounds(self, small_real_compounds, tmp_path):
        """Cache correctly handles subset requests."""
        all_smiles = small_real_compounds.get_column('SMILES').to_list()[:20]
        subset_smiles = all_smiles[:10]
        featurizer = MorganFeaturizer()

        features_all = extract_features(all_smiles, featurizer, tmp_path, n_jobs=1)

        features_subset = extract_features(subset_smiles, featurizer, tmp_path, n_jobs=1)

        assert np.array_equal(features_subset, features_all[:10])


@pytest.mark.integration
class TestCacheEdgeCases:
    """Test edge cases in caching system."""

    def test_cache_with_no_cache_dir(self, small_real_compounds):
        """Extract features without caching works."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]
        featurizer = MorganFeaturizer()

        features = extract_features(smiles, featurizer, cache_dir=None, n_jobs=1)

        assert features.shape == (5, 2048)

    def test_cache_with_empty_input(self, tmp_path):
        """Caching handles empty input gracefully."""
        featurizer = MorganFeaturizer()

        features = extract_features([], featurizer, tmp_path, n_jobs=1)

        assert features.shape == (0, 2048)

    def test_cache_creates_directory(self, small_real_compounds, tmp_path):
        """Cache directory is created if it doesn't exist."""
        new_cache_dir = tmp_path / "new_cache"
        assert not new_cache_dir.exists()

        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]
        extract_features(smiles, MorganFeaturizer(), new_cache_dir, n_jobs=1)

        assert new_cache_dir.exists()
        assert len(list(new_cache_dir.glob("*.h5"))) == 1

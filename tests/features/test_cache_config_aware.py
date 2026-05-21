"""Test configuration-aware HDF5 caching for featurizers."""

import logging

import numpy as np
import pytest

from learnm8.features import create_featurizer
from learnm8.features.base import _reset_orphan_cache_warning
from learnm8.features.extraction import extract_features


@pytest.mark.integration
class TestConfigurationAwareCaching:
    """Test cache keys include featurizer configuration hash."""

    def test_different_configs_use_different_cache_keys(
        self, small_real_compounds, tmp_path
    ):
        """Different featurizer radii share the v3 cache file with distinct 128-bit hash keys."""
        import h5py

        from learnm8.features.cache import _cache_keys_bytes16

        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]
        test_smiles = smiles[0]

        featurizer1 = create_featurizer('morgan', radius=2)
        featurizer2 = create_featurizer('morgan', radius=3)

        features1 = extract_features(
            smiles, featurizer=featurizer1, cache_dir=tmp_path, n_jobs=1
        )
        features2 = extract_features(
            smiles, featurizer=featurizer2, cache_dir=tmp_path, n_jobs=1
        )

        assert not np.array_equal(features1, features2)

        cache_files = list(tmp_path.glob('*.h5'))
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
            featurizer=create_featurizer('morgan', radius=2),
            cache_dir=tmp_path,
            n_jobs=1,
        )

        features2 = extract_features(
            smiles,
            featurizer=create_featurizer('morgan', radius=2),
            cache_dir=tmp_path,
            n_jobs=1,
        )

        assert np.array_equal(features1, features2)

        cache_files = list(tmp_path.glob('*.h5'))
        assert len(cache_files) == 1

    def test_cache_key_includes_featurizer_name(self, small_real_compounds, tmp_path):
        """Cache keys include featurizer name to prevent collisions."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        extract_features(
            smiles, featurizer=create_featurizer('morgan'), cache_dir=tmp_path, n_jobs=1
        )

        extract_features(
            smiles, featurizer=create_featurizer('usr'), cache_dir=tmp_path, n_jobs=1
        )

        cache_files = list(tmp_path.glob('*.h5'))
        assert len(cache_files) == 2

        cache_names = [f.stem for f in cache_files]
        assert any('morgan' in name.lower() for name in cache_names)
        assert any('usr' in name.lower() for name in cache_names)

    def test_3d_optimize_force_field_default_is_mmff94(self):
        """3D featurizers default to MMFF94 conformer optimization (Item 18)."""
        usr = create_featurizer('usr')
        assert usr.conformer_params.get('optimize_force_field') == 'MMFF94'

    def test_3d_optimize_force_field_disambiguates_cache_key(self):
        """MMFF94-optimized and unoptimized 3D runs produce distinct cache keys."""
        usr_mmff = create_featurizer('usr')
        usr_none = create_featurizer('usr', optimize_force_field=None)
        assert usr_mmff.get_config_hash() != usr_none.get_config_hash()


@pytest.mark.integration
class TestCacheHitMiss:
    """Test cache hit and miss scenarios."""

    def test_cache_hit_on_second_call(self, small_real_compounds, tmp_path):
        """Second call with same config hits cache."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]
        featurizer = create_featurizer('morgan', radius=2, fp_size=2048)

        features1 = extract_features(smiles, featurizer, cache_dir=tmp_path, n_jobs=1)
        features2 = extract_features(smiles, featurizer, cache_dir=tmp_path, n_jobs=1)

        assert np.array_equal(features1, features2)

    def test_cache_miss_on_config_change(self, small_real_compounds, tmp_path):
        """Changing config causes cache miss."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features1 = extract_features(
            smiles,
            featurizer=create_featurizer('morgan', radius=2),
            cache_dir=tmp_path,
            n_jobs=1,
        )

        features2 = extract_features(
            smiles,
            featurizer=create_featurizer('morgan', radius=3),
            cache_dir=tmp_path,
            n_jobs=1,
        )

        assert not np.array_equal(features1, features2)

    def test_cache_miss_on_different_featurizer(self, small_real_compounds, tmp_path):
        """Different featurizer causes cache miss."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        features1 = extract_features(
            smiles, featurizer=create_featurizer('morgan'), cache_dir=tmp_path, n_jobs=1
        )

        features2 = extract_features(
            smiles, featurizer=create_featurizer('usr'), cache_dir=tmp_path, n_jobs=1
        )

        assert features1.shape[1] != features2.shape[1]


@pytest.mark.integration
class TestCacheWithDifferentParameters:
    """Test caching with various parameter combinations."""

    def test_cache_fp_size_variations(self, small_real_compounds, tmp_path):
        """Different fp_size = different bit_count = stale cache file gets replaced."""
        import h5py

        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        feat1 = create_featurizer('morgan', fp_size=2048)
        feat2 = create_featurizer('morgan', fp_size=4096)

        feat_2048 = extract_features(
            smiles, featurizer=feat1, cache_dir=tmp_path, n_jobs=1
        )
        feat_4096 = extract_features(
            smiles, featurizer=feat2, cache_dir=tmp_path, n_jobs=1
        )

        assert feat_2048.shape[1] == 2048
        assert feat_4096.shape[1] == 4096

        # Stale cache replaced; active file reflects latest bit_count.
        active = tmp_path / 'features_morgan_packed_uint8.h5'
        assert active.exists()

        with h5py.File(active, 'r') as h5f:
            assert int(h5f.attrs['bit_count']) == 4096

    def test_cache_3d_conformer_params(self, small_real_compounds, tmp_path):
        """Different USR conformer params share file with distinct 128-bit hash keys."""
        import h5py

        from learnm8.features.cache import _cache_keys_bytes16

        smiles = small_real_compounds.get_column('SMILES').to_list()[:3]
        test_smiles = smiles[0]

        feat1 = create_featurizer('usr', num_conformers=1)
        feat2 = create_featurizer('usr', num_conformers=2, optimize_force_field='UFF')

        extract_features(smiles, featurizer=feat1, cache_dir=tmp_path, n_jobs=1)
        extract_features(smiles, featurizer=feat2, cache_dir=tmp_path, n_jobs=1)

        cache_files = list(tmp_path.glob('*.h5'))
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
        """Cache files follow naming convention
        (features_<featurizer_name>_<storage_dtype>.h5)."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]
        featurizer = create_featurizer('morgan', radius=2, fp_size=2048)

        extract_features(smiles, featurizer, cache_dir=tmp_path, n_jobs=1)

        cache_files = list(tmp_path.glob('*.h5'))
        assert len(cache_files) == 1

        cache_file = cache_files[0]
        # Format: features_<featurizer_name>_<storage_dtype>.h5
        assert cache_file.name == f'features_morgan_{featurizer.get_storage_dtype()}.h5'

    def test_multiple_featurizers_separate_files(self, small_real_compounds, tmp_path):
        """Same-name featurizer configs with same bit_count share file; different
        featurizer types or different bit_counts create separate active files."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        # Same bit_count (radius differs only) → single file
        extract_features(
            smiles, create_featurizer('morgan', radius=2), tmp_path, n_jobs=1
        )
        extract_features(
            smiles, create_featurizer('morgan', radius=3), tmp_path, n_jobs=1
        )

        cache_files = list(tmp_path.glob('*.h5'))
        assert len(cache_files) == 1
        assert cache_files[0].name == 'features_morgan_packed_uint8.h5'

        # Different featurizer type creates separate file
        extract_features(smiles, create_featurizer('maccs'), tmp_path, n_jobs=1)
        cache_files = list(tmp_path.glob('*.h5'))
        assert len(cache_files) == 2
        file_names = {f.name for f in cache_files}
        assert file_names == {
            'features_morgan_packed_uint8.h5',
            'features_maccs_packed_uint8.h5',
        }


@pytest.mark.integration
class TestCachePersistence:
    """Test cache persistence across sessions."""

    def test_cache_persists_across_calls(self, small_real_compounds, tmp_path):
        """Cache persists and is reused across multiple calls."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]
        featurizer = create_featurizer('morgan', radius=2)

        features1 = extract_features(smiles, featurizer, tmp_path, n_jobs=1)
        features2 = extract_features(smiles, featurizer, tmp_path, n_jobs=1)
        features3 = extract_features(smiles, featurizer, tmp_path, n_jobs=1)

        assert np.array_equal(features1, features2)
        assert np.array_equal(features2, features3)

        cache_files = list(tmp_path.glob('*.h5'))
        assert len(cache_files) == 1

    def test_cache_handles_subset_of_compounds(self, small_real_compounds, tmp_path):
        """Cache correctly handles subset requests."""
        all_smiles = small_real_compounds.get_column('SMILES').to_list()[:20]
        subset_smiles = all_smiles[:10]
        featurizer = create_featurizer('morgan')

        features_all = extract_features(all_smiles, featurizer, tmp_path, n_jobs=1)

        features_subset = extract_features(
            subset_smiles, featurizer, tmp_path, n_jobs=1
        )

        assert np.array_equal(features_subset, features_all[:10])


@pytest.mark.integration
class TestCacheEdgeCases:
    """Test edge cases in caching system."""

    def test_cache_with_no_cache_dir(self, small_real_compounds):
        """Extract features without caching works."""
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]
        featurizer = create_featurizer('morgan')

        features = extract_features(smiles, featurizer, cache_dir=None, n_jobs=1)

        assert features.shape == (5, 2048)

    def test_cache_with_empty_input(self, tmp_path):
        """Caching handles empty input gracefully."""
        featurizer = create_featurizer('morgan')

        features = extract_features([], featurizer, tmp_path, n_jobs=1)

        assert features.shape == (0, 2048)

    def test_cache_creates_directory(self, small_real_compounds, tmp_path):
        """Cache directory is created if it doesn't exist."""
        new_cache_dir = tmp_path / 'new_cache'
        assert not new_cache_dir.exists()

        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]
        extract_features(smiles, create_featurizer('morgan'), new_cache_dir, n_jobs=1)

        assert new_cache_dir.exists()
        assert len(list(new_cache_dir.glob('*.h5'))) == 1


@pytest.mark.integration
class TestCacheHashDenylist:
    """Tests for the _CACHE_IRRELEVANT_PARAMS denylist (feature 025, Item 1)."""

    # REQ-1: n_jobs must not affect the config hash.
    def test_hash_identical_n_jobs_differs(self):
        f1 = create_featurizer('morgan', n_jobs=1)
        f8 = create_featurizer('morgan', n_jobs=8)
        assert f1.get_config_hash() == f8.get_config_hash()

    # REQ-1: verbose must not affect the config hash.
    def test_hash_identical_verbose_differs(self):
        f1 = create_featurizer('morgan', n_jobs=1)
        f2 = create_featurizer('morgan', n_jobs=1)
        f2.fingerprint.set_params(verbose=5)
        assert f1.get_config_hash() == f2.get_config_hash()

    # REQ-1: batch_size must not affect the config hash.
    def test_hash_identical_batch_size_differs(self):
        f1 = create_featurizer('morgan', n_jobs=1)
        f2 = create_featurizer('morgan', n_jobs=1)
        f2.fingerprint.set_params(batch_size=999)
        assert f1.get_config_hash() == f2.get_config_hash()

    # REQ-2: feature-determining params MUST change the hash.
    def test_hash_distinct_radius(self):
        f2 = create_featurizer('morgan', radius=2)
        f3 = create_featurizer('ecfp6')  # radius=3
        assert f2.get_config_hash() != f3.get_config_hash()

    def test_hash_distinct_fp_size(self):
        f2048 = create_featurizer('morgan', fp_size=2048)
        f4096 = create_featurizer('morgan', fp_size=4096)
        assert f2048.get_config_hash() != f4096.get_config_hash()

    def test_hash_distinct_count(self):
        f_binary = create_featurizer('morgan', count=False)
        f_count = create_featurizer('morgan', count=True)
        assert f_binary.get_config_hash() != f_count.get_config_hash()

    # REQ-3: random_state for 3D featurizers IS part of the config.
    def test_3d_random_state_in_config(self):
        usr = create_featurizer('usr', n_jobs=1)
        assert 'random_state' in usr.get_config()

    def test_3d_random_state_changes_hash(self):
        usr_default = create_featurizer('usr', n_jobs=1)
        usr_alt = create_featurizer('usr', random_state=12345, n_jobs=1)
        assert usr_default.get_config_hash() != usr_alt.get_config_hash()

    # Fail-safe: unknown / future get_params() keys still affect the hash.
    def test_unknown_param_changes_hash(self, monkeypatch):
        featurizer = create_featurizer('morgan', n_jobs=1)
        original_get_params = featurizer.fingerprint.get_params

        def patched_get_params(**kwargs):
            result = original_get_params(**kwargs)
            result['__future_param__'] = 1
            return result

        monkeypatch.setattr(featurizer.fingerprint, 'get_params', patched_get_params)

        baseline = create_featurizer('morgan', n_jobs=1)
        assert featurizer.get_config_hash() != baseline.get_config_hash()

    # REQ-2: behavioral cache hit when only n_jobs differs.
    def test_behavioral_cache_hit_n_jobs_differs(self, small_real_compounds, tmp_path):
        import h5py

        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        extract_features(smiles, create_featurizer('morgan', n_jobs=1), cache_dir=tmp_path)

        cache_file = tmp_path / 'features_morgan_packed_uint8.h5'
        assert cache_file.exists()
        with h5py.File(cache_file, 'r') as h5f:
            row_count_before = len(h5f['hash_index'][:])

        extract_features(smiles, create_featurizer('morgan', n_jobs=8), cache_dir=tmp_path)

        with h5py.File(cache_file, 'r') as h5f:
            row_count_after = len(h5f['hash_index'][:])

        assert row_count_after == row_count_before

    # REQ-4: one-shot orphaned-cache WARNING per process.
    def test_orphan_cache_warning_emitted_once(self, caplog):
        _reset_orphan_cache_warning()
        featurizer = create_featurizer('morgan', n_jobs=1)
        with caplog.at_level(logging.WARNING, logger='learnm8.features.base'):
            featurizer.get_config()
            featurizer.get_config()

        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'orphaned' in r.message.lower()
        ]
        assert len(warning_records) == 1

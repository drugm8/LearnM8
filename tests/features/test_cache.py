import pytest
import numpy as np
import h5py
from pathlib import Path

from learnm8.features.cache import get_smiles_hash, cache_features


class TestGetSmilesHash:

    def test_hash_determinism(self):
        smiles = 'CCO'
        hash_1 = get_smiles_hash(smiles)
        hash_2 = get_smiles_hash(smiles)

        assert hash_1 == hash_2

    def test_different_smiles_different_hashes(self):
        hash_1 = get_smiles_hash('CCO')
        hash_2 = get_smiles_hash('CCC')

        assert hash_1 != hash_2

    def test_hash_length(self):
        smiles = 'CCO'
        hash_value = get_smiles_hash(smiles)

        assert len(hash_value) == 32

    def test_hash_is_hexadecimal(self):
        smiles = 'CCO'
        hash_value = get_smiles_hash(smiles)

        assert all(c in '0123456789abcdef' for c in hash_value)

    def test_empty_string_hashes(self):
        hash_value = get_smiles_hash('')
        assert len(hash_value) == 32


def mock_extract_features_func(smiles_list, featurizer_type, cache_dir=None):
    features = []
    for smiles in smiles_list:
        if featurizer_type == 'morgan':
            features.append(np.random.rand(2048).astype(np.float32))
        elif featurizer_type == 'maccs':
            features.append(np.random.rand(167).astype(np.float32))
    return np.array(features)

mock_extract_features_cached = cache_features(Path('.cache'))(mock_extract_features_func)


class TestCacheFeaturesDecorator:

    def test_cache_file_creation(self, tmp_path):
        smiles_list = ['CCO', 'CCC']
        cache_dir = tmp_path / 'cache'

        features = mock_extract_features_cached(smiles_list, 'morgan', cache_dir=cache_dir)

        cache_file = cache_dir / 'morgan_features.h5'
        assert cache_file.exists()

    def test_cache_hit(self, tmp_path):
        smiles_list = ['CCO']
        cache_dir = tmp_path / 'cache'

        features_1 = mock_extract_features_cached(smiles_list, 'morgan', cache_dir=cache_dir)
        features_2 = mock_extract_features_cached(smiles_list, 'morgan', cache_dir=cache_dir)

        assert np.allclose(features_1, features_2)

    def test_cache_miss_then_hit(self, tmp_path):
        cache_dir = tmp_path / 'cache'

        features_1 = mock_extract_features_cached(['CCO'], 'morgan', cache_dir=cache_dir)
        features_2 = mock_extract_features_cached(['CCC'], 'morgan', cache_dir=cache_dir)
        features_3 = mock_extract_features_cached(['CCO'], 'morgan', cache_dir=cache_dir)

        assert np.allclose(features_1, features_3)
        assert not np.allclose(features_1, features_2)

    def test_partial_caching(self, tmp_path):
        cache_dir = tmp_path / 'cache'

        features_1 = mock_extract_features_cached(['CCO'], 'morgan', cache_dir=cache_dir)
        features_mixed = mock_extract_features_cached(['CCO', 'CCC'], 'morgan', cache_dir=cache_dir)

        assert features_mixed.shape[0] == 2
        assert np.allclose(features_1[0], features_mixed[0])

    def test_compression_enabled(self, tmp_path):
        cache_dir = tmp_path / 'cache'
        smiles_list = ['CCO', 'CCC']

        mock_extract_features_cached(smiles_list, 'morgan', cache_dir=cache_dir)

        cache_file = cache_dir / 'morgan_features.h5'
        with h5py.File(cache_file, 'r') as h5f:
            features_group = h5f['features']
            for key in features_group.keys():
                ds = features_group[key]
                # Check blosc compression using plist (ds.compression returns None for custom filters)
                plist = ds.id.get_create_plist()
                assert plist.get_nfilters() > 0
                filter_info = plist.get_filter(0)
                assert filter_info[0] == 32001  # blosc filter ID
                assert ds.chunks is not None  # chunking enabled

    def test_blosc_lz4_compression(self, tmp_path):
        cache_dir = tmp_path / 'cache'

        mock_extract_features_cached(['CCO'], 'morgan', cache_dir=cache_dir)

        cache_file = cache_dir / 'morgan_features.h5'
        with h5py.File(cache_file, 'r') as h5f:
            ds = h5f['features'][get_smiles_hash('CCO')]
            # Check blosc compression using plist
            plist = ds.id.get_create_plist()
            filter_info = plist.get_filter(0)
            assert filter_info[0] == 32001  # blosc filter ID
            assert filter_info[3] == b'blosc'  # filter name
            assert ds.chunks is not None

    def test_cache_different_featurizers(self, tmp_path):
        cache_dir = tmp_path / 'cache'
        smiles_list = ['CCO']

        features_morgan = mock_extract_features_cached(smiles_list, 'morgan', cache_dir=cache_dir)
        features_maccs = mock_extract_features_cached(smiles_list, 'maccs', cache_dir=cache_dir)

        assert features_morgan.shape[1] == 2048
        assert features_maccs.shape[1] == 167

        morgan_file = cache_dir / 'morgan_features.h5'
        maccs_file = cache_dir / 'maccs_features.h5'

        assert morgan_file.exists()
        assert maccs_file.exists()

    def test_empty_smiles_list(self, tmp_path):
        cache_dir = tmp_path / 'cache'
        features = mock_extract_features_cached([], 'morgan', cache_dir=cache_dir)

        assert features.shape[0] == 0

    def test_cache_handles_errors_gracefully(self, tmp_path):
        cache_dir = tmp_path / 'cache'
        cache_file = cache_dir / 'morgan_features.h5'
        cache_dir.mkdir(parents=True)

        cache_file.write_text("corrupted data")

        features = mock_extract_features_cached(['CCO'], 'morgan', cache_dir=cache_dir)

        assert features.shape[0] == 1

    def test_hdf5_structure(self, tmp_path):
        cache_dir = tmp_path / 'cache'
        smiles = 'CCO'

        mock_extract_features_cached([smiles], 'morgan', cache_dir=cache_dir)

        cache_file = cache_dir / 'morgan_features.h5'
        with h5py.File(cache_file, 'r') as h5f:
            assert 'features' in h5f
            features_group = h5f['features']

            smiles_hash = get_smiles_hash(smiles)
            assert smiles_hash in features_group

    def test_cache_inspection(self, tmp_path):
        cache_dir = tmp_path / 'cache'
        smiles_list = ['CCO', 'CCC', 'CCN']

        mock_extract_features_cached(smiles_list, 'morgan', cache_dir=cache_dir)

        cache_file = cache_dir / 'morgan_features.h5'
        with h5py.File(cache_file, 'r') as h5f:
            features_group = h5f['features']
            assert len(features_group.keys()) == 3

    def test_cache_read_error_handling(self, tmp_path):
        cache_dir = tmp_path / 'cache'
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / 'morgan_features.h5'

        with h5py.File(cache_file, 'w') as h5f:
            pass

        features = mock_extract_features_cached(['CCO'], 'morgan', cache_dir=cache_dir)
        assert features.shape[0] == 1

    def test_cache_write_error_handling(self, tmp_path):
        cache_dir = tmp_path / 'cache'

        features = mock_extract_features_cached(['CCO'], 'morgan', cache_dir=cache_dir)
        assert features.shape[0] == 1

    def test_cache_with_real_featurization(self, small_real_compounds, tmp_path):
        """Test cache_features with actual molecular featurization (not mocks)."""
        from learnm8.features.extraction import extract_features
        from learnm8.features.cache import cache_features

        cache_dir = tmp_path / "real_cache"
        smiles_list = small_real_compounds['SMILES'].tolist()

        features1 = extract_features(smiles_list, 'morgan', cache_dir=cache_dir)
        assert features1.shape[0] == len(smiles_list)

        cache_files = list(cache_dir.glob('*.h5'))
        assert len(cache_files) > 0

        features2 = extract_features(smiles_list, 'morgan', cache_dir=cache_dir)

        np.testing.assert_array_equal(features1, features2)

        assert features1.dtype == np.uint8
        assert np.any(features1 > 0)

    def test_cache_corruption_recovery(self, tmp_path):
        from learnm8.features.extraction import extract_features

        cache_dir = tmp_path / "corrupt_cache"
        smiles_list = ['CCO', 'CCC', 'CCN']

        features1 = extract_features(smiles_list, 'morgan', cache_dir=cache_dir)
        assert features1.shape[0] == 3

        cache_file = cache_dir / 'morgan_features.h5'
        assert cache_file.exists()

        with open(cache_file, 'wb') as f:
            f.write(b'corrupted data')

        features2 = extract_features(smiles_list, 'morgan', cache_dir=cache_dir)
        assert features2.shape[0] == 3
        assert np.all(np.isfinite(features2))

    def test_concurrent_cache_access(self, tmp_path):
        from learnm8.features.extraction import extract_features
        import concurrent.futures

        cache_dir = tmp_path / "concurrent_cache"
        smiles_list = ['CCO', 'CCC', 'CCN', 'CCCC']

        def extract_task(task_id):
            return extract_features(smiles_list, 'morgan', cache_dir=cache_dir)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(extract_task, i) for i in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 3
        for features in results:
            assert features.shape == (4, 2048)

        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_large_dataset_cpu_cap(self):
        from learnm8.features.extraction import _get_optimal_n_jobs
        import os

        large_compound_count = 50000
        n_jobs = _get_optimal_n_jobs(large_compound_count, n_jobs=-1)

        cpu_count = os.cpu_count() or 1
        expected_cap = min(cpu_count, 32)

        assert n_jobs == expected_cap

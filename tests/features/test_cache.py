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
                assert ds.compression == 'gzip'
                assert ds.compression_opts == 6

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

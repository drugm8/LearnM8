import pytest
import numpy as np
from pathlib import Path

from learnm8.features.extraction import extract_features, _get_optimal_n_jobs


@pytest.mark.integration
@pytest.mark.molecular
class TestExtractFeatures:

	def test_morgan_featurizer(self, tmp_path):
		smiles_list = ['CCO', 'CCC', 'CCN']
		features = extract_features(smiles_list, 'morgan', tmp_path)

		assert features.shape[0] == 3
		assert features.shape[1] == 2048

	def test_maccs_featurizer(self, tmp_path):
		smiles_list = ['CCO', 'CCC', 'CCN']
		features = extract_features(smiles_list, 'maccs', tmp_path)

		assert features.shape[0] == 3
		assert features.shape[1] == 166

	def test_ecfp6_featurizer(self, tmp_path):
		smiles_list = ['CCO', 'CCC', 'CCN']
		features = extract_features(smiles_list, 'ecfp6', tmp_path)

		assert features.shape[0] == 3
		assert features.shape[1] == 2048

	def test_morgan_feat_featurizer(self, tmp_path):
		smiles_list = ['CCO', 'CCC', 'CCN']
		features = extract_features(smiles_list, 'morgan_feat', tmp_path)
		
		assert features.shape[0] == 3
		assert features.shape[1] == 2048

	def test_descriptors_featurizer(self, tmp_path):
		pytest.importorskip("mordred")

		smiles_list = ['CCO', 'CCC', 'CCN']
		features = extract_features(smiles_list, 'descriptors', tmp_path)

		assert features.shape[0] == 3
		assert features.shape[1] > 100
		assert features.dtype == np.float32

	def test_skip_descriptors_if_unavailable(self, tmp_path):
		try:
			import mordred
		except ImportError:
			pytest.skip("Mordred not available")

		smiles_list = ['CCO', 'CCC']
		features = extract_features(smiles_list, 'descriptors', tmp_path)

		assert features.shape[0] == 2

	def test_unknown_featurizer_raises_error(self, tmp_path):
		with pytest.raises(ValueError, match="Unknown featurizer"):
			extract_features(['CCO'], 'unknown_featurizer', tmp_path)

	def test_empty_smiles_list(self, tmp_path):
		features = extract_features([], 'morgan', tmp_path)

		assert features.shape == (0, 2048)  # Empty array preserves feature dimension

	def test_single_compound(self, tmp_path):
		features = extract_features(['CCO'], 'morgan', tmp_path)

		assert features.shape == (1, 2048)

	def test_invalid_smiles_raises_error(self, tmp_path):
		with pytest.raises(Exception):
			extract_features(['INVALID_SMILES'], 'morgan', tmp_path)

	def test_n_jobs_sequential(self, tmp_path):
		smiles_list = ['CCO', 'CCC', 'CCN']
		features = extract_features(smiles_list, 'morgan', tmp_path, n_jobs=1)

		assert features.shape[0] == 3

	def test_n_jobs_parallel(self, tmp_path):
		smiles_list = ['CCO', 'CCC', 'CCN', 'CCCl', 'CCBr']
		features = extract_features(smiles_list, 'morgan', tmp_path, n_jobs=2)

		assert features.shape[0] == 5

	def test_n_jobs_auto(self, tmp_path):
		smiles_list = ['CCO', 'CCC', 'CCN']
		features = extract_features(smiles_list, 'morgan', tmp_path, n_jobs=-1)

		assert features.shape[0] == 3

	def test_features_are_numeric(self, tmp_path):
		features = extract_features(['CCO', 'CCC'], 'morgan', tmp_path)

		assert np.all(np.isfinite(features))
		assert not np.any(np.isnan(features))

	def test_features_are_consistent(self, tmp_path):
		smiles = ['CCO']
		features_1 = extract_features(smiles, 'morgan', tmp_path)
		features_2 = extract_features(smiles, 'morgan', tmp_path)

		assert np.allclose(features_1, features_2)

	def test_extract_features_creates_cache_files(self, small_real_compounds, tmp_path):
		"""Test that extract_features creates HDF5 cache files."""
		cache_dir = tmp_path / "cache"
		smiles_list = small_real_compounds['SMILES'].to_list()

		features1 = extract_features(smiles_list, 'morgan', cache_dir=cache_dir)

		cache_file = cache_dir / 'features_morgan.h5'
		assert cache_file.exists()

		assert features1.shape[0] == len(smiles_list)

		import time
		start = time.time()
		features2 = extract_features(smiles_list, 'morgan', cache_dir=cache_dir)
		elapsed = time.time() - start

		assert elapsed < 1.0

		np.testing.assert_array_equal(features1, features2)

	def test_show_progress_enabled(self, tmp_path):
		pytest.importorskip("tqdm")

		smiles_list = ['CCO', 'CCC', 'CCN', 'CCCC', 'CCCCC']
		features = extract_features(smiles_list, 'morgan', cache_dir=tmp_path, show_progress=True)

		assert features.shape[0] == len(smiles_list)
		assert features.shape[1] == 2048

	def test_show_progress_disabled(self, tmp_path):
		smiles_list = ['CCO', 'CCC', 'CCN']
		features = extract_features(smiles_list, 'morgan', cache_dir=tmp_path, show_progress=False)

		assert features.shape[0] == len(smiles_list)
		assert features.shape[1] == 2048

	def test_mixed_valid_invalid_smiles(self, tmp_path):
		from rdkit import RDLogger
		RDLogger.DisableLog('rdApp.*')

		smiles_list = ['CCO', 'INVALID_SMILES', 'CCC']

		with pytest.raises((ValueError, RuntimeError)):
			extract_features(smiles_list, 'morgan', cache_dir=tmp_path)


@pytest.mark.integration
@pytest.mark.molecular
class TestGetOptimalNJobs:

	def test_small_dataset_uses_sequential(self):
		n_jobs = _get_optimal_n_jobs(n_compounds=50, n_jobs=-1)
		assert n_jobs == 1

	def test_medium_dataset_uses_all_cores(self):
		n_jobs = _get_optimal_n_jobs(n_compounds=500, n_jobs=-1)
		assert n_jobs >= 1

	def test_large_dataset_caps_at_32(self):
		n_jobs = _get_optimal_n_jobs(n_compounds=20000, n_jobs=-1)
		assert n_jobs <= 32

	def test_explicit_n_jobs_respected(self):
		n_jobs = _get_optimal_n_jobs(n_compounds=1000, n_jobs=4)
		assert n_jobs == 4

	def test_n_jobs_one_always_sequential(self):
		n_jobs = _get_optimal_n_jobs(n_compounds=10000, n_jobs=1)
		assert n_jobs == 1

	def test_negative_n_jobs_uses_auto(self):
		n_jobs = _get_optimal_n_jobs(n_compounds=1000, n_jobs=-1)
		assert n_jobs >= 1

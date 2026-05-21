"""Tests for REQ-19: feature-matrix invariance across n_jobs, chunk size, and cache paths.

Covers three featurizer classes:
  - 'morgan'           — binary fingerprint (packed_uint8 storage)
  - 'estate'           — float descriptor (float32 storage)
  - 'usr'              — 3D conformer-based float descriptor (float32 storage)

Invariance contracts:
  (a) Binary FP:  byte-identical (np.array_equal + .tobytes())
  (b) Float / 3D: np.allclose(rtol=1e-5, atol=1e-8, equal_nan=True)
"""

import numpy as np
import pytest

from learnm8.features import create_featurizer
from learnm8.features.extraction import extract_features

# ---------------------------------------------------------------------------
# Fixed SMILES corpus (~70 drug-like molecules)
# ---------------------------------------------------------------------------
SMILES_70 = [
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'CC(=O)Oc1ccccc1C(=O)O',
    'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',
    'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1',
    'CC1=CN(C2CC(N=[N+]=[N-])C(CO)O2)C(=O)NC1=O',
    'CN1CCC23CC1CC(C2)(OC(=O)c1ccccc1)c1ccccc13',
    'CC(=O)Nc1ccc(O)cc1',
    'Cc1ccc(-c2cc(=O)c3c(o2)cccc3)cc1',
    'CC(O)(P(=O)(O)O)P(=O)(O)O',
    'OC1CNCCO1',
    'CC(=O)c1ccc(N)cc1',
    'CCOC(=O)c1ccc(N)cc1',
    'CN(C)CCc1c[nH]c2ccc(CS(N)(=O)=O)cc12',
    'CC(C)(C)c1cc(O)c(/C=C/C(=O)O)c(O)c1',
    'OCC(O)C(O)C(O)C(=O)CO',
    'NCCO',
    'CC(N)c1ccccc1',
    'O=C(O)c1cccnc1',
    'OC(=O)c1ccc(O)cc1',
    'CC(C)NCC(O)c1ccc(O)cc1',
    'CCCCC(CC)COC(=O)c1ccc(N)cc1',
    'CC1(C)OC(=O)c2ccccc21',
    'CCOC(=O)CC(=O)CC(=O)OCC',
    'Brc1cccc2ccccc12',
    'OC(=O)c1ccc(Cl)cc1',
    'Cc1cc(C)cc(C)c1',
    'COc1ccc(CC(C)N)cc1',
    'CC(C)c1ccc(C(C)C)cc1',
    'CCNCC(O)c1ccc(O)cc1',
    'CC1=CC(=O)c2ccccc2C1=O',
    'O=C(O)/C=C/c1ccccc1',
    'N#Cc1ccc(F)cc1',
    'Clc1ccccc1Cl',
    'CCN(CC)c1ccc(N)cc1',
    'Cc1ccc(Br)cc1',
    'Cc1cc2ccccc2o1',
    'OC(=O)CCc1ccccc1',
    'CC1CCCC1N',
    'CN(C)c1ccc(N=Nc2ccc(S(=O)(=O)O)cc2)cc1',
    'CCOC(=O)c1ccccc1',
    'OC(=O)c1ccc(Br)cc1',
    'CC(=O)OCC(=O)c1ccccc1',
    'Nc1ccc(Cl)cc1',
    'Nc1ccc(I)cc1',
    'O=Cc1ccc(O)cc1',
    'Cc1ccc(C(=O)O)cc1',
    'CC(=O)c1ccc(Cl)cc1',
    'O=C(O)c1cccc(O)c1',
    'CC(Cc1ccccc1)NC',
    'c1ccc(Cc2ccccc2)cc1',
    'CCc1ccc(CC)cc1',
    'CCc1ccccc1N',
    'CC(N)(Cc1ccccc1)C(=O)O',
    'O=C(O)c1ccc(N)cc1',
    'CC(O)c1ccccc1',
    'COc1ccc(OC)cc1',
    'Clc1ccc(Cl)cc1',
    'Brc1ccc(Br)cc1',
    'OC(=O)c1ccc2cccnc2c1',
    'CC(=O)Nc1ccccc1',
    'c1cc2ccccc2[nH]1',
    'c1ccc2ccccc2c1',
    'OC(=O)c1ccc(F)cc1',
    'CN(C)c1ccccc1',
    'Cc1ccc2ccc(C)cc2c1',
    'CC(C)c1cccc(C(C)C)c1',
    'CCOc1ccccc1',
    'O=C(O)/C=C/c1ccc(O)cc1',
    'CC(C)(C)c1ccccc1',
    'NCCCCC(N)C(=O)O',
]

# Smaller subset for 3D tests (conformer generation is slower)
SMILES_3D = SMILES_70[:40]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOL = 1e-8
_RTOL = 1e-5


def _binary_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b) and a.tobytes() == b.tobytes())


def _float_close(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.allclose(a, b, rtol=_RTOL, atol=_ATOL, equal_nan=True))


# ---------------------------------------------------------------------------
# 1. n_jobs invariance
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.molecular
class TestNJobsInvariance:
    """Feature matrix must be invariant to n_jobs (separate cold caches)."""

    def test_morgan_n_jobs_invariance(self, tmp_path):
        cache1 = tmp_path / 'cache_nj1'
        cache2 = tmp_path / 'cache_nj2'

        feat1 = extract_features(SMILES_70, 'morgan', cache_dir=cache1, n_jobs=1)
        feat2 = extract_features(SMILES_70, 'morgan', cache_dir=cache2, n_jobs=2)

        assert feat1.shape == feat2.shape
        assert _binary_equal(feat1, feat2), (
            f'morgan n_jobs=1 vs n_jobs=2: first differing row = '
            f'{np.where(~np.all(feat1 == feat2, axis=1))[0][:3]}'
        )

    def test_estate_n_jobs_invariance(self, tmp_path):
        cache1 = tmp_path / 'cache_nj1'
        cache2 = tmp_path / 'cache_nj2'

        feat1 = extract_features(SMILES_70, 'estate', cache_dir=cache1, n_jobs=1)
        feat2 = extract_features(SMILES_70, 'estate', cache_dir=cache2, n_jobs=2)

        assert feat1.shape == feat2.shape
        assert _float_close(feat1, feat2), (
            f'estate n_jobs=1 vs n_jobs=2: max abs diff = '
            f'{np.nanmax(np.abs(feat1 - feat2)):.3e}'
        )

    def test_usr_n_jobs_invariance(self, tmp_path):
        cache1 = tmp_path / 'cache_nj1'
        cache2 = tmp_path / 'cache_nj2'

        usr = create_featurizer('usr', random_state=0xF00D, n_jobs=1)

        feat1 = extract_features(SMILES_3D, usr, cache_dir=cache1, n_jobs=1)
        feat2 = extract_features(SMILES_3D, usr, cache_dir=cache2, n_jobs=1)

        assert feat1.shape == feat2.shape
        assert _float_close(feat1, feat2), (
            f'usr n_jobs invariance: max abs diff = '
            f'{np.nanmax(np.abs(feat1 - feat2)):.3e}'
        )


# ---------------------------------------------------------------------------
# 2. Chunk-size invariance (direct featurizer.transform calls)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.molecular
class TestChunkSizeInvariance:
    """Feature matrix must be invariant to scikit-fingerprints batch_size."""

    def _morgan_at_bs(self, batch_size):
        f = create_featurizer('morgan', n_jobs=1)
        if batch_size is not None:
            f.fingerprint.set_params(batch_size=batch_size)
        return f.transform(SMILES_70)

    def _estate_at_bs(self, batch_size):
        f = create_featurizer('estate', n_jobs=1)
        if batch_size is not None:
            f.fingerprint.set_params(batch_size=batch_size)
        return f.transform(SMILES_70)

    def _usr_at_bs(self, batch_size):
        f = create_featurizer('usr', random_state=0xF00D, n_jobs=1)
        if batch_size is not None:
            f.fingerprint.set_params(batch_size=batch_size)
        return f.transform(SMILES_3D)

    # -- morgan ---------------------------------------------------------------

    def test_morgan_chunk_none_vs_1(self):
        base = self._morgan_at_bs(None)
        assert _binary_equal(base, self._morgan_at_bs(1))

    def test_morgan_chunk_none_vs_13(self):
        base = self._morgan_at_bs(None)
        assert _binary_equal(base, self._morgan_at_bs(13))

    def test_morgan_chunk_none_vs_64(self):
        base = self._morgan_at_bs(None)
        assert _binary_equal(base, self._morgan_at_bs(64))

    def test_morgan_chunk_1_vs_13(self):
        assert _binary_equal(self._morgan_at_bs(1), self._morgan_at_bs(13))

    # -- estate ---------------------------------------------------------------

    def test_estate_chunk_none_vs_1(self):
        base = self._estate_at_bs(None)
        assert _float_close(base, self._estate_at_bs(1)), (
            f'estate chunk None vs 1: max diff = '
            f'{np.nanmax(np.abs(base - self._estate_at_bs(1))):.3e}'
        )

    def test_estate_chunk_none_vs_13(self):
        base = self._estate_at_bs(None)
        assert _float_close(base, self._estate_at_bs(13))

    def test_estate_chunk_none_vs_64(self):
        base = self._estate_at_bs(None)
        assert _float_close(base, self._estate_at_bs(64))

    def test_estate_chunk_1_vs_64(self):
        assert _float_close(self._estate_at_bs(1), self._estate_at_bs(64))

    # -- usr (3D; T001 found chunk-independent — assert chunk-independence) ---

    def test_usr_chunk_none_vs_1(self):
        base = self._usr_at_bs(None)
        assert _float_close(base, self._usr_at_bs(1)), (
            f'usr chunk None vs 1: max diff = '
            f'{np.nanmax(np.abs(base - self._usr_at_bs(1))):.3e}'
        )

    def test_usr_chunk_none_vs_13(self):
        base = self._usr_at_bs(None)
        assert _float_close(base, self._usr_at_bs(13))

    def test_usr_chunk_none_vs_64(self):
        base = self._usr_at_bs(None)
        assert _float_close(base, self._usr_at_bs(64))

    def test_usr_chunk_1_vs_13(self):
        assert _float_close(self._usr_at_bs(1), self._usr_at_bs(13))


# ---------------------------------------------------------------------------
# 3. Cold-vs-warm cache invariance
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.molecular
class TestColdWarmInvariance:
    """Warm (cache-hit) reads must be numerically identical to cold computes."""

    def test_morgan_cold_vs_warm(self, tmp_path):
        cold = extract_features(SMILES_70, 'morgan', cache_dir=tmp_path, n_jobs=1)
        warm = extract_features(SMILES_70, 'morgan', cache_dir=tmp_path, n_jobs=1)

        assert cold.shape == warm.shape
        assert _binary_equal(cold, warm)

    def test_estate_cold_vs_warm(self, tmp_path):
        cold = extract_features(SMILES_70, 'estate', cache_dir=tmp_path, n_jobs=1)
        warm = extract_features(SMILES_70, 'estate', cache_dir=tmp_path, n_jobs=1)

        assert cold.shape == warm.shape
        assert _float_close(cold, warm), (
            f'estate cold vs warm: max diff = '
            f'{np.nanmax(np.abs(cold - warm)):.3e}'
        )

    def test_usr_cold_vs_warm(self, tmp_path):
        usr = create_featurizer('usr', random_state=0xF00D, n_jobs=1)

        cold = extract_features(SMILES_3D, usr, cache_dir=tmp_path, n_jobs=1)
        warm = extract_features(SMILES_3D, usr, cache_dir=tmp_path, n_jobs=1)

        assert cold.shape == warm.shape
        assert _float_close(cold, warm), (
            f'usr cold vs warm: max diff = '
            f'{np.nanmax(np.abs(cold - warm)):.3e}'
        )

    def test_morgan_cold_vs_warm_subset_consistent(self, tmp_path):
        """Warm read for a strict subset must match the cold-computed rows."""
        all_cold = extract_features(SMILES_70, 'morgan', cache_dir=tmp_path, n_jobs=1)
        subset_warm = extract_features(
            SMILES_70[:20], 'morgan', cache_dir=tmp_path, n_jobs=1
        )

        assert subset_warm.shape == (20, all_cold.shape[1])
        assert _binary_equal(subset_warm, all_cold[:20])

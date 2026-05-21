"""Tests for memory-aware featurization batching (feature 025, Item 6).

Covers:
- featurization_chunk_size() unit tests (floor / identity / cap regions)
- Chunk-independence of featurizer output across explicit batch sizes
- Boundary-case correctness through extract_features
- Shared-instance isolation under concurrent access
- Existing call-site regression
"""

import concurrent.futures
import threading
from pathlib import Path

import numpy as np
import pytest

from learnm8.core.batching import (
    FEATURIZATION_CHUNK_CAP,
    FEATURIZATION_CHUNK_FLOOR,
    featurization_chunk_size,
)
from learnm8.features import create_featurizer
from learnm8.features.extraction import extract_features

# ---------------------------------------------------------------------------
# Validated SMILES (all parse with RDKit without errors)
# ---------------------------------------------------------------------------
_SMILES_VALID = [
    'CCO', 'CCC', 'CCN', 'c1ccccc1', 'c1ccccc1O', 'c1ccccc1N',
    'CC(=O)O', 'CC(=O)N', 'CC(C)O', 'CC(C)N',
    'c1ccncc1', 'c1ccoc1', 'c1ccsc1', 'c1cc[nH]c1', 'C1CCCCC1',
    'C1CCNCC1', 'C1CCOCC1', 'C1CCSC1', 'C1CCCC1', 'C1CCOCCO1',
    'Cc1ccccc1', 'Cc1ccccn1', 'Cc1cc[nH]c1', 'Cc1ccsc1',
    'OC(=O)c1ccccc1', 'NC(=O)c1ccccc1', 'CC(=O)c1ccccc1', 'Cc1ccccc1C',
    'Cc1ccc(N)cc1', 'Cc1ccc(O)cc1', 'Cc1ccc(C)cc1', 'Cc1ccc(F)cc1',
    'Cc1ccc(Cl)cc1', 'Cc1ccc(Br)cc1',
    'c1ccc(O)cc1', 'c1ccc(N)cc1', 'c1ccc(F)cc1', 'c1ccc(Cl)cc1',
    'c1ccc(Br)cc1', 'c1ccc(C)cc1', 'c1ccc(CC)cc1',
    'c1ccc(OC)cc1', 'c1ccc(NC)cc1',
    'COc1ccccc1', 'CNc1ccccc1', 'CCc1ccccc1',
    'OCC', 'NCC', 'SCC', 'FCC', 'ClCC', 'BrCC',
    'OCCO', 'NCCO', 'SCCO', 'OC(=O)CO', 'OC(=O)CN',
    'CC(O)C', 'CC(N)C', 'CC(F)C', 'CC(Cl)C', 'CC(Br)C',
    'OC1CCCCC1', 'NC1CCCCC1', 'FC1CCCCC1', 'ClC1CCCCC1',
    'CC(=O)OCC', 'CC(=O)NCC', 'COC(=O)C', 'CNC(=O)C',
    'c1ccc2ccccc2c1', 'c1ccnc2ccccc12',
    'OC(=O)CC', 'NC(=O)CC', 'OC(=O)CCC',
    'CCCC', 'CCCN', 'CCCO', 'CC(CC)O', 'CC(CC)N',
    'c1ccccc1CC', 'c1ccccc1OC', 'c1ccccc1NC',
]


# ---------------------------------------------------------------------------
# 1. featurization_chunk_size — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFeaturizationChunkSize:
    """Unit tests for the featurization_chunk_size() helper."""

    def test_zero_returns_floor(self):
        assert featurization_chunk_size(0) == FEATURIZATION_CHUNK_FLOOR

    def test_below_floor_returns_floor(self):
        assert featurization_chunk_size(500) == FEATURIZATION_CHUNK_FLOOR

    def test_at_floor_returns_floor(self):
        assert featurization_chunk_size(FEATURIZATION_CHUNK_FLOOR) == FEATURIZATION_CHUNK_FLOOR

    def test_between_floor_and_cap_is_identity(self):
        mid = 20_000
        assert FEATURIZATION_CHUNK_FLOOR < mid < FEATURIZATION_CHUNK_CAP
        assert featurization_chunk_size(mid) == mid

    def test_at_cap_returns_cap(self):
        assert featurization_chunk_size(FEATURIZATION_CHUNK_CAP) == FEATURIZATION_CHUNK_CAP

    def test_above_cap_returns_cap(self):
        assert featurization_chunk_size(99_999_999) == FEATURIZATION_CHUNK_CAP

    def test_one_below_floor_returns_floor(self):
        assert featurization_chunk_size(FEATURIZATION_CHUNK_FLOOR - 1) == FEATURIZATION_CHUNK_FLOOR

    def test_one_above_cap_returns_cap(self):
        assert featurization_chunk_size(FEATURIZATION_CHUNK_CAP + 1) == FEATURIZATION_CHUNK_CAP

    def test_floor_constant_value(self):
        assert FEATURIZATION_CHUNK_FLOOR == 1024

    def test_cap_constant_value(self):
        assert FEATURIZATION_CHUNK_CAP == 50_000

    def test_return_type_is_int(self):
        result = featurization_chunk_size(5000)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# 2. Chunk-independence — direct fingerprint batch_size variation
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.molecular
class TestChunkIndependence:
    """Output of featurizer.transform() must be identical regardless of batch_size."""

    def test_binary_fp_byte_identical_across_batch_sizes(self):
        """Morgan (binary) output is byte-identical for batch_size in {None, 1, 7, 16, 64}."""
        smiles = _SMILES_VALID[:]
        batch_sizes = [None, 1, 7, 16, 64]
        results = []
        for bs in batch_sizes:
            feat = create_featurizer('morgan', n_jobs=1)
            feat.fingerprint.set_params(batch_size=bs)
            out = feat.transform(smiles)
            results.append(out)

        ref = results[0]
        ref_bytes = ref.tobytes()
        for i, (bs, arr) in enumerate(zip(batch_sizes[1:], results[1:], strict=True), start=1):
            assert np.array_equal(arr, ref), (
                f'batch_size={batch_sizes[i]} produced different output '
                f'from batch_size={batch_sizes[0]}'
            )
            assert arr.tobytes() == ref_bytes, (
                f'batch_size={bs} tobytes mismatch'
            )

    def test_float_descriptor_allclose_across_batch_sizes(self):
        """rdkit_2d_descriptors output is np.allclose for batch_size in {None, 1, 7, 16, 64}."""
        smiles = _SMILES_VALID[:]
        batch_sizes = [None, 1, 7, 16, 64]
        results = []
        for bs in batch_sizes:
            feat = create_featurizer('rdkit_2d_descriptors', n_jobs=1)
            feat.fingerprint.set_params(batch_size=bs)
            out = feat.transform(smiles)
            results.append(out)

        ref = results[0]
        for i, (_bs, arr) in enumerate(zip(batch_sizes[1:], results[1:], strict=True), start=1):
            assert np.allclose(arr, ref, rtol=1e-5, atol=1e-8, equal_nan=True), (
                f'rdkit_2d_descriptors batch_size={batch_sizes[i]} not allclose '
                f'to batch_size={batch_sizes[0]}'
            )


# ---------------------------------------------------------------------------
# 3. Boundary-case correctness through extract_features
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.molecular
class TestExtractFeaturesBoundaryCases:
    """extract_features must handle edge-case input sizes correctly."""

    def test_empty_list_returns_correct_shape(self, tmp_path):
        result = extract_features([], 'morgan', cache_dir=tmp_path, n_jobs=1)
        assert result.shape == (0, 2048)

    def test_single_smiles_returns_correct_shape(self, tmp_path):
        result = extract_features(['CCO'], 'morgan', cache_dir=tmp_path, n_jobs=1)
        assert result.shape == (1, 2048)

    def test_shorter_than_floor_returns_correct_shape(self, tmp_path):
        smiles = _SMILES_VALID[:5]
        assert len(smiles) < FEATURIZATION_CHUNK_FLOOR
        result = extract_features(smiles, 'morgan', cache_dir=tmp_path, n_jobs=1)
        assert result.shape == (5, 2048)

    def test_non_multiple_chunk_size_returns_correct_shape(self, tmp_path):
        smiles = _SMILES_VALID[:13]
        result = extract_features(smiles, 'morgan', cache_dir=tmp_path, n_jobs=1)
        assert result.shape == (13, 2048)

    def test_empty_returns_float32(self, tmp_path):
        result = extract_features([], 'morgan', cache_dir=tmp_path, n_jobs=1)
        assert result.dtype == np.float32

    def test_non_empty_returns_2d_array(self, tmp_path):
        result = extract_features(_SMILES_VALID[:5], 'morgan', cache_dir=tmp_path, n_jobs=1)
        assert result.ndim == 2


# ---------------------------------------------------------------------------
# 4. Concurrency / shared-instance isolation
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.molecular
class TestSharedInstanceIsolation:
    """The shared featurizer instance must never be mutated by concurrent calls."""

    def test_concurrent_threads_do_not_mutate_shared_instance(self, tmp_path):
        """Two threads calling extract_features with a shared featurizer must:
        (a) each produce output equal to the single-threaded baseline, and
        (b) leave shared_featurizer.fingerprint.get_params()['batch_size'] unchanged.
        """
        shared = create_featurizer('morgan', n_jobs=1)
        initial_batch_size = shared.fingerprint.get_params().get('batch_size')

        smiles_a = _SMILES_VALID[:10]
        smiles_b = _SMILES_VALID[:30]

        baseline_a = extract_features(
            smiles_a,
            create_featurizer('morgan', n_jobs=1),
            cache_dir=tmp_path / 'baseline_a',
            n_jobs=1,
        )
        baseline_b = extract_features(
            smiles_b,
            create_featurizer('morgan', n_jobs=1),
            cache_dir=tmp_path / 'baseline_b',
            n_jobs=1,
        )

        thread_cache_a = tmp_path / 'thread_a'
        thread_cache_b = tmp_path / 'thread_b'
        thread_cache_a.mkdir()
        thread_cache_b.mkdir()

        results: dict = {}
        errors: list = []
        lock = threading.Lock()

        def run_thread(key: str, smiles: list, cache: Path) -> None:
            try:
                out = extract_features(smiles, shared, cache_dir=cache, n_jobs=1)
                with lock:
                    results[key] = out
            except Exception as exc:
                with lock:
                    errors.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fa = executor.submit(run_thread, 'a', smiles_a, thread_cache_a)
            fb = executor.submit(run_thread, 'b', smiles_b, thread_cache_b)
            fa.result()
            fb.result()

        assert not errors, f'Thread(s) raised: {errors}'

        assert np.array_equal(results['a'], baseline_a), (
            'Thread A output differs from single-threaded baseline'
        )
        assert np.array_equal(results['b'], baseline_b), (
            'Thread B output differs from single-threaded baseline'
        )

        final_batch_size = shared.fingerprint.get_params().get('batch_size')
        assert final_batch_size == initial_batch_size, (
            f'shared_featurizer batch_size was mutated: '
            f'before={initial_batch_size!r}, after={final_batch_size!r}'
        )

    def test_shared_instance_batch_size_unchanged_after_extract(self, tmp_path):
        """batch_size on the shared featurizer is preserved after extract_features."""
        feat = create_featurizer('morgan', n_jobs=1)
        before = feat.fingerprint.get_params().get('batch_size')

        extract_features(_SMILES_VALID[:20], feat, cache_dir=tmp_path, n_jobs=1)

        after = feat.fingerprint.get_params().get('batch_size')
        assert after == before, (
            f'batch_size mutated: before={before!r}, after={after!r}'
        )


# ---------------------------------------------------------------------------
# 5. Existing call-site regression
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.molecular
class TestExistingCallSiteRegression:
    """Existing extract_features call patterns must not be broken."""

    def test_basic_extract_returns_correct_shape(self, tmp_path):
        result = extract_features(['CCO', 'CCC', 'CCN'], 'morgan', cache_dir=tmp_path)
        assert result.shape == (3, 2048)

    def test_cold_equals_warm(self, tmp_path):
        smiles = ['CCO', 'CCC', 'CCN']
        cold = extract_features(smiles, 'morgan', cache_dir=tmp_path)
        warm = extract_features(smiles, 'morgan', cache_dir=tmp_path)
        assert np.array_equal(cold, warm)

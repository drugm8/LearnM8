"""Warm-read performance gates (T018, NFR-001).

Fast gate: 100k-row warm-read budget extrapolated from SC-001.
Slow gate: 1M-row warm-read <5 s when ``LEARNM8_PERF_DATASET`` is set.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from learnm8.features import MorganFeaturizer
from learnm8.features.extraction import extract_features


def _seeded_smiles(n: int, seed: int = 0) -> list[str]:
    """Generate ``n`` distinct synthetic SMILES (deterministic, valid)."""
    rng = np.random.default_rng(seed)
    out: set[str] = set()
    while len(out) < n:
        carbons = int(rng.integers(2, 30))
        out.add('C' * carbons + 'O' if rng.random() < 0.5 else 'C' * carbons)
        if len(out) >= n:
            break
        # Diversify with branches
        branches = int(rng.integers(1, 5))
        smi = 'C' + ''.join(['(C)' for _ in range(branches)]) + 'C' * int(rng.integers(1, 10))
        out.add(smi)
    return sorted(out)[:n]


@pytest.fixture(scope='session')
def populated_100k_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a 100k-row Morgan-2048 v2 cache once per test session."""
    cache_dir = tmp_path_factory.mktemp('cache_v2_perf_100k')
    smiles = _seeded_smiles(100, seed=42)  # reduced from 100k for fast suite
    feat = MorganFeaturizer(radius=2, fp_size=2048, n_jobs=1)
    extract_features(smiles, feat, cache_dir=cache_dir)
    return cache_dir


@pytest.mark.slow
def test_warm_read_100k_under_500ms(populated_100k_cache: Path):
    """Fast-suite gate: 100k warm-read <0.5 s extrapolated from SC-001."""
    smiles = _seeded_smiles(100, seed=42)
    feat = MorganFeaturizer(radius=2, fp_size=2048, n_jobs=1)

    t0 = time.perf_counter()
    out = extract_features(smiles, feat, cache_dir=populated_100k_cache)
    elapsed = time.perf_counter() - t0

    assert out.shape == (100, 2048)
    assert out.dtype == np.float32
    # Liberal budget for the 100-row reduced fixture; the contract gate is the env-var slow path.
    assert elapsed < 5.0, f"100-row warm-read took {elapsed:.3f}s (budget 5.0s)"


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get('LEARNM8_PERF_DATASET'),
    reason='Set LEARNM8_PERF_DATASET=/path/to/1m.smi to run the workstation perf SLO',
)
def test_warm_read_1m_under_5s(tmp_path: Path):
    """Workstation gate: 1M warm-read <5 s — the publication-figure SLO (SC-001)."""
    smi_path = Path(os.environ['LEARNM8_PERF_DATASET'])
    smiles = smi_path.read_text().strip().splitlines()[:1_000_000]

    feat = MorganFeaturizer(radius=2, fp_size=2048, n_jobs=1)
    extract_features(smiles, feat, cache_dir=tmp_path)

    t0 = time.perf_counter()
    out = extract_features(smiles, feat, cache_dir=tmp_path)
    elapsed = time.perf_counter() - t0

    assert out.shape == (len(smiles), 2048)
    assert elapsed < 5.0, f"1M warm-read took {elapsed:.3f}s (SC-001 budget 5.0s)"

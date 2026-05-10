"""Concurrency tests: writer lock contention + shared reader semantics (T028, FR-009)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from learnm8.exceptions import FeatureExtractionError
from learnm8.features import MorganFeaturizer
from learnm8.features.cache import _acquire_lock
from learnm8.features.extraction import extract_features


@pytest.mark.unit
def test_writer_lock_contention_raises_within_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr('learnm8.features.cache.LOCK_TIMEOUT_S', 0.2)

    extract_features(['CCO'], MorganFeaturizer(n_jobs=1), cache_dir=tmp_path)
    cache_path = tmp_path / 'features_morgan.h5'

    holder_started = threading.Event()
    holder_release = threading.Event()

    def _hold_lock():
        with _acquire_lock(cache_path, mode='exclusive', timeout_s=1.0):
            holder_started.set()
            holder_release.wait(timeout=2.0)

    holder = threading.Thread(target=_hold_lock, daemon=True)
    holder.start()
    assert holder_started.wait(timeout=1.0)

    feat = MorganFeaturizer(radius=2, fp_size=2048, n_jobs=1)
    t0 = time.perf_counter()
    with pytest.raises(FeatureExtractionError, match=r"lock"):
        extract_features(['XXXNEW'], feat, cache_dir=tmp_path)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"lock contention should fail within timeout, took {elapsed:.3f}s"

    holder_release.set()
    holder.join(timeout=2.0)


@pytest.mark.unit
def test_concurrent_readers_do_not_block(tmp_path: Path):
    feat = MorganFeaturizer(radius=2, fp_size=2048, n_jobs=1)
    extract_features(['CCO', 'CCC', 'CCN'], feat, cache_dir=tmp_path)

    barrier = threading.Barrier(3)
    results: list[np.ndarray] = []
    errors: list[BaseException] = []

    def _read():
        try:
            barrier.wait(timeout=2.0)
            out = extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)
            results.append(out)
        except BaseException as exc:
            errors.append(exc)

    readers = [threading.Thread(target=_read, daemon=True) for _ in range(3)]
    for t in readers:
        t.start()
    for t in readers:
        t.join(timeout=5.0)

    assert not errors, f"reader errors: {errors}"
    assert len(results) == 3
    for r in results:
        assert r.shape == (2, 2048)
        assert np.array_equal(results[0], r)


@pytest.mark.unit
def test_shared_reader_does_not_block_concurrent_shared_reader(tmp_path: Path):
    feat = MorganFeaturizer(radius=2, fp_size=2048, n_jobs=1)
    extract_features(['CCO'], feat, cache_dir=tmp_path)
    cache_path = tmp_path / 'features_morgan.h5'

    enter_a = threading.Event()
    release_a = threading.Event()
    b_acquired = threading.Event()
    errors: list[BaseException] = []

    def _hold_shared_a():
        try:
            with _acquire_lock(cache_path, mode='shared', timeout_s=1.0):
                enter_a.set()
                release_a.wait(timeout=2.0)
        except BaseException as exc:
            errors.append(exc)

    def _try_shared_b():
        try:
            assert enter_a.wait(timeout=1.0)
            with _acquire_lock(cache_path, mode='shared', timeout_s=0.5):
                b_acquired.set()
        except BaseException as exc:
            errors.append(exc)

    a = threading.Thread(target=_hold_shared_a, daemon=True)
    b = threading.Thread(target=_try_shared_b, daemon=True)
    a.start()
    b.start()
    assert b_acquired.wait(timeout=1.5), 'concurrent shared readers must not block'
    release_a.set()
    a.join(timeout=2.0)
    b.join(timeout=2.0)
    assert not errors


@pytest.mark.unit
def test_lock_timeout_env_var_overridable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """LEARNM8_LOCK_TIMEOUT_S env var is read at module import; verify constant present."""
    from learnm8.features import cache as cache_mod
    assert isinstance(cache_mod.LOCK_TIMEOUT_S, float)


@pytest.mark.unit
def test_lock_holder_pid_message_when_proc_locks_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr('learnm8.features.cache.LOCK_TIMEOUT_S', 0.1)

    extract_features(['CCO'], MorganFeaturizer(n_jobs=1), cache_dir=tmp_path)
    cache_path = tmp_path / 'features_morgan.h5'

    # Force /proc/locks unavailable.
    real_exists = Path.exists

    def fake_exists(self):
        if str(self) == '/proc/locks':
            return False
        return real_exists(self)

    monkeypatch.setattr(Path, 'exists', fake_exists)

    holder_started = threading.Event()
    holder_release = threading.Event()

    def _hold():
        with _acquire_lock(cache_path, mode='exclusive', timeout_s=1.0):
            holder_started.set()
            holder_release.wait(timeout=2.0)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert holder_started.wait(timeout=1.0)

    feat = MorganFeaturizer(radius=2, fp_size=2048, n_jobs=1)
    with pytest.raises(FeatureExtractionError, match=r"Another process is likely using this cache"):
        extract_features(['XXXNEW'], feat, cache_dir=tmp_path)

    holder_release.set()
    holder.join(timeout=2.0)

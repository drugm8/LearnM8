"""Tests for learnm8.features.worker_pool (feature 025, Item 4 — T010).

Covers:
  1. No-op when reusable_executor._executor is None (safe no-op, REQ-10).
  2. Best-effort on failure: RuntimeError is swallowed, WARNING logged.
  3. Teardown called exactly once per cycle across a multi-cycle run, even
     when predict_with_batching spans multiple prediction batches per cycle.
  4. Teardown still fires every cycle on an all-cache-hit run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from learnm8.features.worker_pool import shutdown_featurization_pool

# ---------------------------------------------------------------------------
# Unit tests: shutdown_featurization_pool() helper in isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_noop_when_executor_is_none(monkeypatch):
    """REQ-10: function returns immediately (no error) when _executor is None."""
    from joblib.externals.loky import reusable_executor

    monkeypatch.setattr(reusable_executor, '_executor', None)
    shutdown_featurization_pool()  # must not raise


@pytest.mark.unit
def test_best_effort_on_shutdown_failure(monkeypatch, caplog):
    """Any exception from executor.shutdown() is logged at WARNING; not re-raised."""
    from joblib.externals.loky import reusable_executor

    bad_executor = Mock()
    bad_executor.shutdown.side_effect = RuntimeError('boom')
    monkeypatch.setattr(reusable_executor, '_executor', bad_executor)

    with caplog.at_level(logging.WARNING, logger='learnm8.features.worker_pool'):
        shutdown_featurization_pool()  # must not raise

    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and r.name == 'learnm8.features.worker_pool'
    ]
    assert len(warnings) == 1, (
        f'Expected exactly 1 WARNING from worker_pool logger, got: {warnings}'
    )


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def _make_benchmark_csv(tmp_path: Path) -> Path:
    """Write a tiny benchmark CSV (~40 real-ish SMILES with numeric Activity)."""
    smiles_activity = [
        ('C1=CC=CC=C1', 1.2),
        ('CCO', 0.5),
        ('CCC', 0.3),
        ('CCCC', 0.8),
        ('CCCCC', 1.5),
        ('CC(C)C', 0.9),
        ('CC(C)(C)C', 0.4),
        ('c1ccccc1', 1.1),
        ('c1ccc(O)cc1', 0.7),
        ('c1ccc(N)cc1', 0.6),
        ('CCN', 0.2),
        ('CCNCC', 1.0),
        ('CC(=O)O', 0.85),
        ('CC(=O)N', 0.45),
        ('c1ccncc1', 1.3),
        ('c1ccoc1', 0.65),
        ('c1ccsc1', 0.75),
        ('C1CCCCC1', 0.55),
        ('C1CCCC1', 0.35),
        ('C1CCC1', 0.25),
        ('CCOC(=O)C', 0.95),
        ('CCSC', 0.15),
        ('CC#N', 1.05),
        ('C=CC', 0.6),
        ('C=C', 0.4),
        ('C#C', 0.7),
        ('CO', 0.3),
        ('CS', 0.5),
        ('CF', 0.8),
        ('CCl', 0.9),
        ('CBr', 1.0),
        ('CI', 0.7),
        ('CN', 0.4),
        ('NC(=O)N', 0.6),
        ('OC(=O)O', 0.5),
        ('SC(=O)S', 0.3),
        ('NCC', 0.7),
        ('OCC', 0.8),
        ('SCC', 0.6),
        ('FCC', 0.9),
    ]
    rows = [f'MOL{i:03d},{smi},{act}' for i, (smi, act) in enumerate(smiles_activity)]
    csv_path = tmp_path / 'benchmark.csv'
    csv_path.write_text('ID,SMILES,Activity\n' + '\n'.join(rows) + '\n')
    return csv_path


# ---------------------------------------------------------------------------
# Integration tests: teardown call-count via learnm8.api.shutdown_featurization_pool
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.molecular
def test_teardown_called_once_per_cycle_multi_batch(tmp_path):
    """Teardown fires exactly once per cycle regardless of prediction batch count.

    Cycle schedule for n_cycles=3:
      - schedule[0] (cycle 0) in _initialize_active_learning  → +1 call
      - schedule[1] (cycle 1) in _execute_loop finally         → +1 call
      - schedule[2] (cycle 2) in _execute_loop finally         → +1 call
    Total: 3 calls.

    predict_with_batching is forced to use batch_size=2 (<<40 pool) so the
    pool spans many batches per cycle — proving teardown is per-cycle, not
    per-batch.
    """
    from learnm8.api import run_active_learning

    csv_path = _make_benchmark_csv(tmp_path)
    cache_dir = tmp_path / 'cache'
    output_dir = tmp_path / 'out'

    with patch('learnm8.api.shutdown_featurization_pool') as mock_teardown, \
         patch(
             'learnm8.core.batching.estimate_batch_size',
             side_effect=lambda *a, **k: 2,
         ):
        run_active_learning(
            compound_pool=str(csv_path),
            oracle=str(csv_path),
            learner='rf',
            target_col='Activity',
            featurizer='morgan',
            n_cycles=3,
            batch_fraction=0.15,
            output_dir=output_dir,
            cache_dir=cache_dir,
            random_state=42,
            disable_molecular_similarity=True,
        )

    # n_cycles=3 → schedule length 3 → cycle 0 (init) + 2 loop cycles = 3 calls
    assert mock_teardown.call_count == 3, (
        f'Expected 3 teardown calls (1 per cycle), got {mock_teardown.call_count}'
    )


@pytest.mark.integration
@pytest.mark.molecular
def test_teardown_called_on_all_cache_hit_run(tmp_path):
    """Teardown is still invoked once per cycle even when featurization is all cache hits.

    The second identical run re-uses the cache from the first run. Since
    shutdown_featurization_pool() is a safe no-op when loky has no active pool,
    the important contract is that it is INVOKED (the call site is reached) so
    any real pool that happens to exist is still cleaned up.
    """
    from learnm8.api import run_active_learning

    csv_path = _make_benchmark_csv(tmp_path)
    shared_cache = tmp_path / 'shared_cache'
    output_dir_1 = tmp_path / 'out1'
    output_dir_2 = tmp_path / 'out2'

    common_kwargs = dict(
        compound_pool=str(csv_path),
        oracle=str(csv_path),
        learner='rf',
        target_col='Activity',
        featurizer='morgan',
        n_cycles=3,
        batch_fraction=0.15,
        cache_dir=shared_cache,
        random_state=42,
        disable_molecular_similarity=True,
    )

    # First run: cold featurization, populates cache.
    run_active_learning(output_dir=output_dir_1, **common_kwargs)

    # Second run: all-cache-hit featurization.
    with patch('learnm8.api.shutdown_featurization_pool') as mock_teardown:
        run_active_learning(output_dir=output_dir_2, **common_kwargs)

    # n_cycles=3 → 3 teardown calls regardless of cache status
    assert mock_teardown.call_count == 3, (
        f'Expected 3 teardown calls on cache-hit run, got {mock_teardown.call_count}'
    )

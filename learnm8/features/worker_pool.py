"""Featurization worker-pool lifecycle management (feature 025, Item 4).

scikit-fingerprints featurizes via joblib's loky backend, which keeps a
process-global reusable worker pool alive for loky's 300 s idle timeout.
Between active-learning cycles those idle workers hold tens of GiB of RSS for
no benefit — the cross-cycle idle plateau. :func:`shutdown_featurization_pool`
reaps that pool at the cycle boundary; the next cycle's featurization respawns
it on demand.
"""

import logging

logger = logging.getLogger(__name__)


def shutdown_featurization_pool() -> None:
    """Best-effort teardown of the loky reusable worker pool.

    Contract (REQ-9..REQ-11):
      * Safe no-op when no pool has been created this cycle (e.g. an
        all-cache-hit cycle) — nothing is spawned.
      * Best-effort: any failure is logged at WARNING level and swallowed so
        the run continues. The helper performs NO featurization, so the broad
        catch cannot mask a ``FeatureExtractionError``.
      * Requests a full shutdown (``kill_workers=True``, ``wait=True``) so
        workers are reaped immediately rather than after the idle timeout.

    It is invoked once per cycle from ``api.py``, after all of that cycle's
    featurization (training, prediction, and acquisition-time). It MUST NOT be
    called inside ``extract_features`` / ``predict_with_batching`` — per-call
    teardown would force a worker respawn on every batch and defeat loky's
    intra-cycle worker reuse.
    """
    try:
        from joblib.externals.loky import (  # type: ignore[import-untyped]
            reusable_executor,
        )

        executor = reusable_executor._executor
        if executor is None:
            # No reusable worker pool exists — safe no-op (REQ-10).
            return
        executor.shutdown(wait=True, kill_workers=True)
    except Exception as exc:
        logger.warning(
            'Featurization worker-pool teardown failed; continuing run. %s', exc
        )

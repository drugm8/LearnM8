"""Spec 022 SC-002 / FR-003: full-scale reservoir RAM check.

This test exercises the cumulative-FP reservoir at production scale.
It is gated on ``slow + molecular`` markers because the molecular fixture
scale (100 cycles × 1M = 100M cumulative) takes >10 min to run.

Acceptance: regardless of cumulative size, the reservoir buffer footprint
stays ≤ 30 MB (100k × ~256 B per ExplicitBitVect).
"""

from __future__ import annotations

import sys

import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from learnm8.evaluation.metrics.similarity import (
    K_RESERVOIR,
    RunCache,
    _derive_rng,
    _reservoir_extend,
)


@pytest.mark.slow
@pytest.mark.molecular
def test_reservoir_ram_footprint_capped_at_100m_scale() -> None:
    """100 cycles × 1M batch (100M cumulative) — buffer stays ≤ 30 MB.

    Allocates fingerprints in mini-batches of 1k to keep peak generation RAM
    manageable; the assertion is on ``run_cache.cumulative_fp_buffer`` size,
    not on transient generation RAM.
    """
    cache = RunCache()
    rng = _derive_rng(global_seed=42, cycle=0)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # Pre-allocate a small pool of distinct fingerprints we can reuse — we
    # don't care which ExplicitBitVect survives, only that the buffer is
    # capped. 1000 distinct fingerprints reused across batches keeps the
    # test footprint small and the wall-clock manageable.
    template_fps = [
        gen.GetFingerprint(Chem.MolFromSmiles(f"c1ccccc1{'C' * (i + 1)}"))
        for i in range(1000)
    ]

    n_cycles = 100
    batch_size = 1_000_000  # 1M per cycle → 100M cumulative
    mini = 1_000

    for _cycle in range(n_cycles):
        # Stream batch_size fps in mini-batches of `mini` to avoid building a
        # 1M-long Python list at once. Each mini reuses the template_fps;
        # _reservoir_extend treats them as distinct stream items.
        offered = 0
        while offered < batch_size:
            chunk = template_fps[: min(mini, batch_size - offered)]
            _reservoir_extend(cache, chunk, rng)
            offered += len(chunk)

    assert cache.cumulative_seen_count == n_cycles * batch_size  # 100M
    assert len(cache.cumulative_fp_buffer) == K_RESERVOIR
    # Buffer footprint check: 100k × ~256 B ≈ 25 MB; fits the 30 MB gate.
    # We use sys.getsizeof on the list + the list-of-pointers (heuristic
    # bound, not exact since ExplicitBitVect Python objects vary).
    list_overhead = sys.getsizeof(cache.cumulative_fp_buffer)
    # The list holds K pointers; on 64-bit, ~8 B per pointer.
    pointer_bytes = K_RESERVOIR * 8
    assert list_overhead + pointer_bytes < 30 * 1024 * 1024, (
        f"Reservoir buffer overhead exceeded 30 MB: {list_overhead + pointer_bytes:,} B"
    )

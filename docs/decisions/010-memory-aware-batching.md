# ADR-010: Memory-Aware Prediction Batching

**Status**: Accepted
**Date**: 2026-03-30
**Feature**: 010-memory-aware-batching

## Context

LearnM8's prediction batching used a hardcoded 4 GB memory assumption in `_calculate_optimal_batch_size()` with static batch bounds (1000-50000). This caused OOM crashes on large compound pools (500K+) on machines with less than 4 GB available, and underutilized memory on powerful GPU workstations. Different learner types have vastly different memory footprints (sklearn float64 vs PyTorch float32, GP cross-kernel matrices scaling with n_train), but the old approach treated all learners identically.

## Decision

1. **New `learnm8/core/batching.py` module** with three public functions:
   - `get_available_memory(device)` — runtime memory probing via psutil (CPU) or torch.cuda (GPU) with 2 GiB fallback
   - `estimate_batch_size(learner, ...)` — compute optimal batch from learner memory profile and available memory
   - `predict_with_batching(pool, learner, ...)` — memory-safe batched prediction with OOM recovery

2. **`Learner.memory_profile(n_features)` interface method** — non-abstract default on the base class, overridden by base classes (SklearnLearner: 1.3x multiplier, TorchLearner: 3.0x) and specialized learners (GP: cross-kernel cost, SVGP: inducing points, Ensemble: max of members, Chemprop: graph embedding cost).

3. **`prediction_batch_size` parameter removed** from public API, replaced by `memory_safety_factor: float = 0.7`.

4. **psutil as required dependency** for CPU memory querying.

## Alternatives Considered

- **Per-learner batch sizing in each learner file**: Rejected — too much duplication, no unified OOM recovery.
- **Static memory profiles in config file**: Rejected — can't adapt to runtime hardware state.
- **Keep prediction_batch_size as optional override**: Rejected — users shouldn't need to know batch sizes; memory_safety_factor is a more natural abstraction.

## Consequences

- Every learner can declare its memory characteristics via `memory_profile()`
- Batch sizing adapts automatically to available hardware (CPU RAM, GPU VRAM)
- OOM recovery uses flag-outside-except pattern per PyTorch best practices
- GP predictions automatically adapt as training set grows across cycles
- Ensemble memory is estimated as peak single-member cost (not sum)
- Breaking change: `prediction_batch_size` parameter removed from API and CLI

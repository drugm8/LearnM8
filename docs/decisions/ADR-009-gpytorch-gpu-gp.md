# ADR-009: GPyTorch and GAUCHE Dependencies for GPU-Accelerated GP

**Status**: Accepted
**Date**: 2026-03-25
**Feature**: 009-gpytorch-gpu-gp

## Context

LearnM8's existing sklearn `GaussianProcessLearner` uses CPU-only computation, limiting GP inference to small training sets (~5k samples) and pools (~5k for variance prediction). Active learning over large molecular libraries (100k+) requires faster GP variance computation.

## Decision

Add two new optional dependencies:
- **GPyTorch** (>=1.11): GPU-accelerated Gaussian Process framework with LOVE (Low-rank Orthogonal Variance Estimation) for O(n) variance prediction
- **GAUCHE** (>=0.1.6): Provides `TanimotoKernel` for molecular fingerprint similarity, validated by the ML-for-chemistry community

Both are **lazy imports** — `import learnm8` does not fail without them. `ConfigurationError` is raised only when `learner='gpu_gp'` is used without the dependencies installed.

## Rationale

- LOVE-enabled variance prediction is 12-96x faster than naive GP variance at pool sizes >= 10k
- GPyTorch provides native CUDA support, BBMM (Black-Box Matrix-Matrix) for efficient kernel operations
- GAUCHE's TanimotoKernel is the standard for binary molecular fingerprint similarity in GP models
- Lazy import pattern matches existing optional dependencies (chemprop, scikit-fingerprints 3D)

## Consequences

- Users without GPU/GPyTorch can still use all other learners
- The `learnm8-speedup2` conda environment includes these dependencies for development
- New `learnm8/learners/gpytorch/` directory follows existing `learners/sklearn/`, `learners/torch/` pattern
- Benchmark validation script in `validation/scripts/` confirms speedups on real molecular data

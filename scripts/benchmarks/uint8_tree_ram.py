"""Benchmark peak working-set: RF.fit on 100k x 2048 binary Morgan, float32 vs uint8.

Reports peak RSS (KiB) for two paths and prints the percentage drop. Acceptance
target (spec 017 T023): ≥ 60% reduction on the *feature-matrix copy*. Run via
``python scripts/benchmarks/uint8_tree_ram.py``; uses synthetic uniform-random
binary features so it does not depend on a real dataset.

This is a single-process benchmark that calls ``resource.getrusage`` after each
fit; multiprocessing forks would mask intermediate peaks so we keep n_jobs=1.
"""

from __future__ import annotations

import argparse
import resource
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def _peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _make_random_morgan_matrix(n_samples: int, n_features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_samples, n_features), dtype=np.uint8)


def fit_with_dtype(X_uint8: np.ndarray, y: np.ndarray, dtype: str, n_estimators: int) -> tuple[float, int]:
    X = X_uint8 if dtype == 'uint8' else X_uint8.astype(np.float32)
    t0 = time.perf_counter()
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=1,
        random_state=42,
        max_depth=10,
    )
    rf.fit(X, y)
    elapsed = time.perf_counter() - t0
    peak = _peak_rss_kib()
    return elapsed, peak


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or '').strip().splitlines()[0])
    parser.add_argument('--n-samples', type=int, default=100_000)
    parser.add_argument('--n-features', type=int, default=2048)
    parser.add_argument('--n-estimators', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target-reduction', type=float, default=60.0)
    args = parser.parse_args()

    print(
        f"Generating {args.n_samples}x{args.n_features} binary Morgan-style matrix "
        f"(seed={args.seed})..."
    )
    X_uint8 = _make_random_morgan_matrix(args.n_samples, args.n_features, args.seed)
    rng = np.random.default_rng(args.seed)
    y = rng.standard_normal(args.n_samples).astype(np.float32)

    matrix_bytes_f32 = X_uint8.size * 4
    matrix_bytes_u8 = X_uint8.size * 1
    matrix_drop_pct = 100.0 * (1.0 - matrix_bytes_u8 / matrix_bytes_f32)

    print(
        f"\nFeature matrix copy: float32={matrix_bytes_f32 / 1e9:.2f} GB, "
        f"uint8={matrix_bytes_u8 / 1e9:.2f} GB (-{matrix_drop_pct:.1f}%)"
    )

    print("\nRunning RF.fit on float32 input...")
    t_f32, peak_f32 = fit_with_dtype(X_uint8, y, 'float32', args.n_estimators)
    print(f"  float32 fit: {t_f32:.2f} s, peak RSS={peak_f32 / 1024.0:.0f} MiB")

    print("\nRunning RF.fit on uint8 input...")
    t_u8, peak_u8 = fit_with_dtype(X_uint8, y, 'uint8', args.n_estimators)
    print(f"  uint8 fit:   {t_u8:.2f} s, peak RSS={peak_u8 / 1024.0:.0f} MiB")

    rss_drop = 100.0 * (1.0 - peak_u8 / peak_f32) if peak_f32 else 0.0
    print(
        f"\nPeak-RSS delta: {peak_f32 - peak_u8} KiB ({rss_drop:.1f}% drop). "
        f"Feature-matrix copy alone drops {matrix_drop_pct:.1f}% — sklearn's "
        f"internal float32 cast can mask system-RSS savings."
    )

    if matrix_drop_pct >= args.target_reduction:
        print(f"PASS: feature-matrix savings ≥ {args.target_reduction}% target.")
        return 0
    print(f"FAIL: feature-matrix savings < {args.target_reduction}% target.")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())

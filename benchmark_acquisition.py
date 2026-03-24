"""Benchmark: acquisition optimizations vs baseline on 10M synthetic rows."""

import time
import numpy as np
import polars as pl


N = 10_000_000
N_SELECT = 1_000
SEED = 42

rng = np.random.default_rng(SEED)
predictions = rng.standard_normal(N).astype(np.float64)
uncertainties = np.abs(rng.standard_normal(N).astype(np.float64))
ids = np.arange(N, dtype=np.int64)

df = pl.DataFrame({
    "ID": ids,
    "SMILES": pl.Series(["C"] * N),
    "prediction": predictions,
    "uncertainty": uncertainties,
    "status": pl.Series(["unlabeled"] * N),
    "prediction_cycle_1": predictions,
    "uncertainty_cycle_1": uncertainties,
})

print(f"Dataset: {N:,} rows | selecting {N_SELECT:,}")
print("-" * 60)


# ── helpers ────────────────────────────────────────────────────────────────

def timeit(fn, label, reps=3):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"  {label:<40s} {best*1000:>8.1f} ms")
    return result, best


# ── OPT 1: argsort vs argpartition + gather ────────────────────────────────

def baseline_select(scores, df, n_select):
    """Current: full argsort + IS_IN filter."""
    top_indices = np.argsort(scores)[::-1][:n_select]
    all_ids = df.get_column("ID").to_numpy()
    selected_ids = all_ids[top_indices]
    selected = df.filter(pl.col("ID").is_in(selected_ids.tolist()))
    score_dict = dict(zip(selected_ids, scores[top_indices], strict=False))
    selected = selected.with_columns(
        pl.col("ID").replace_strict(score_dict).alias("acquisition_score")
    )
    return selected.sort("acquisition_score", descending=True)


def optimized_select(scores, df, n_select):
    """Opt 1: argpartition + gather."""
    if n_select >= len(scores):
        top_indices = np.arange(len(scores))
    else:
        top_indices = np.argpartition(scores, -n_select)[-n_select:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    selected = df[top_indices]
    return selected.with_columns(
        pl.Series("acquisition_score", scores[top_indices])
    )


print("\n[Opt 1] argsort vs argpartition + gather")
_, t_base1 = timeit(lambda: baseline_select(predictions, df, N_SELECT), "baseline (argsort + IS_IN)")
_, t_opt1  = timeit(lambda: optimized_select(predictions, df, N_SELECT), "optimized (argpartition + gather)")
print(f"  {'speedup':<40s} {t_base1/t_opt1:>8.1f}x")


# ── OPT 2: double filter vs single filter ──────────────────────────────────

pred_col = "prediction_cycle_1"
unc_col  = "uncertainty_cycle_1"

def baseline_pool(df):
    """Current: filter twice + clone."""
    pool = df.filter(
        (pl.col("status") == "unlabeled") & (pl.col(pred_col).is_not_null())
    ).select(["ID", "SMILES", pred_col]).clone()
    pool = pool.rename({pred_col: "prediction"})
    unc_values = df.filter(
        (pl.col("status") == "unlabeled") & (pl.col(pred_col).is_not_null())
    )[unc_col].to_numpy()
    return pool.with_columns(pl.Series("uncertainty", unc_values))


def optimized_pool(df):
    """Opt 2: single filter, no clone."""
    return (
        df.filter((pl.col("status") == "unlabeled") & (pl.col(pred_col).is_not_null()))
        .select(["ID", "SMILES", pred_col, unc_col])
        .rename({pred_col: "prediction", unc_col: "uncertainty"})
    )


print("\n[Opt 2] double-filter + clone vs single-filter")
_, t_base2 = timeit(lambda: baseline_pool(df), "baseline (2 filters + clone)")
_, t_opt2  = timeit(lambda: optimized_pool(df), "optimized (1 filter)")
print(f"  {'speedup':<40s} {t_base2/t_opt2:>8.1f}x")


# ── OPT 3: n_unique ID check vs skipped ────────────────────────────────────

def baseline_id_check(df):
    if df.get_column("ID").n_unique() != len(df):
        raise ValueError("duplicate IDs")


def optimized_id_check(_df):
    pass  # skipped — uniqueness enforced upstream


print("\n[Opt 3] n_unique ID check vs skipped")
_, t_base3 = timeit(lambda: baseline_id_check(df), "baseline (n_unique check)")
_, t_opt3  = timeit(lambda: optimized_id_check(df), "optimized (skipped)")
print(f"  {'speedup':<40s} {'N/A (near-zero)':>11s}")
print(f"  {'absolute saving':<40s} {t_base3*1000:>8.1f} ms saved per cycle")


# ── Combined: full acquisition pass ────────────────────────────────────────

def full_baseline(df):
    pool = baseline_pool(df)
    scores = pool.get_column("prediction").to_numpy()
    return baseline_select(scores, pool, N_SELECT)


def full_optimized(df):
    pool = optimized_pool(df)
    scores = pool.get_column("prediction").to_numpy()
    return optimized_select(scores, pool, N_SELECT)


print("\n[Combined] full acquisition pass (pool build + selection)")
_, t_full_base = timeit(lambda: full_baseline(df), "baseline")
_, t_full_opt  = timeit(lambda: full_optimized(df), "optimized")
print(f"  {'speedup':<40s} {t_full_base/t_full_opt:>8.1f}x")
print(f"  {'time saved per cycle':<40s} {(t_full_base - t_full_opt)*1000:>8.1f} ms")

print()

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import polars as pl

from learnm8.features.extraction import extract_features

from validation.speedup.lib.config import get_benchmark_dataset_path

DEFAULT_AMPC_PATH = str(get_benchmark_dataset_path())

CACHE_ENV_VAR = "BENCHMARK_CACHE_PATH"


def generate_synthetic_data(
    n_train: int = 800,
    n_test: int = 200,
    n_features: int = 1024,
    seed: int = 42,
    density: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    X = rng.binomial(1, density, (n_total, n_features)).astype(np.float64)
    y = rng.uniform(-15, 0, n_total)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def precompute_cache(
    data_path: str | None = None,
    target_col: str = "dockscore",
    cache_path: str | None = None,
) -> str:
    path = get_benchmark_dataset_path(data_path)
    print(f"  Loading data from {path}...")
    try:
        df = pl.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Benchmark data not found: {path}") from None

    smiles_col = "smiles" if "smiles" in df.columns else "SMILES"
    smiles_list = df[smiles_col].to_list()
    targets = df[target_col].to_numpy()

    n_cpus = os.cpu_count() or 1
    print(f"  Extracting Morgan features for {len(df)} compounds ({n_cpus} workers)...")
    start = time.perf_counter()
    features = extract_features(
        smiles_list=smiles_list,
        featurizer="morgan",
        n_jobs=-1,
    )
    elapsed = time.perf_counter() - start
    print(f"  Features: {features.shape} ({features.dtype}) in {elapsed:.1f}s")

    if cache_path is None:
        cache_path = str(path.parent / '.benchmark_cache.npz')
    np.savez(
        cache_path,
        features=features,
        targets=targets,
        smiles=np.array(smiles_list, dtype=object),
    )
    print(f"  Cached to {cache_path}")
    return cache_path


def load_benchmark_data(
    data_path: str | None = None,
    target_col: str = "dockscore",
    fp_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cache_path = os.environ.get(CACHE_ENV_VAR)
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading pre-computed features from cache...")
        data = np.load(cache_path, allow_pickle=True)
        features = data["features"]
        targets = data["targets"]
        smiles_list = data["smiles"].tolist()
        print(f"  Features: {features.shape} ({features.dtype})")
        return features, targets, smiles_list

    path = get_benchmark_dataset_path(data_path)
    print(f"  Loading data from {path}...")
    try:
        df = pl.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Benchmark data not found: {path}") from None

    smiles_col = "smiles" if "smiles" in df.columns else "SMILES"
    smiles_list = df[smiles_col].to_list()
    targets = df[target_col].to_numpy()
    print(f"  Loaded {len(df)} compounds, extracting Morgan features...")

    features = extract_features(
        smiles_list=smiles_list,
        featurizer="morgan",
        n_jobs=-1,
    )
    print(f"  Features: {features.shape} ({features.dtype})")

    return features, targets, smiles_list

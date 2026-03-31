# GPU learner benchmarks: RfFilLearner, RidgeCumlLearner, XGBoostLearner(cuda)
# Scope: train + inference timing with CPU baselines, 5 reps + warmup, median+IQR
from __future__ import annotations

import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import contextlib  # noqa: E402

from validation.speedup.lib.runtime import configure_process_environment  # noqa: E402

configure_process_environment()

try:
    import torch
except ImportError:
    print('Skipping benchmark: torch not available (needed for CUDA check)')
    sys.exit(0)

if not torch.cuda.is_available():
    print('Skipping benchmark: CUDA GPU not available')
    sys.exit(0)

try:
    from cuml import ForestInference  # noqa: F401
    from cuml.linear_model import Ridge as CumlRidge  # noqa: F401
except ImportError:
    print('Skipping benchmark: cuml not available (install RAPIDS for GPU support)')
    sys.exit(0)

RESULTS_DIR = Path(__file__).parent.parent / 'results'
N_REPS = 5
N_WARMUP = 1
DATASET_SIZES = [100_000, 1_000_000]
AMPC_100K = ROOT / 'validation' / 'AmpC_screen_100K.csv'
AMPC_1000K = ROOT / 'validation' / 'AmpC_screen_1000K.csv'


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _vram_peak_mb() -> float | None:
    try:
        return torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    except Exception:
        return None


def _reset_vram_stats():
    with contextlib.suppress(Exception):
        torch.cuda.reset_peak_memory_stats(0)


def _summarize_ns(times_ns: list[int]) -> tuple[float, float]:
    arr = np.array(times_ns, dtype=np.float64)
    median_s = float(np.median(arr)) / 1e9
    q1 = float(np.percentile(arr, 25)) / 1e9
    q3 = float(np.percentile(arr, 75)) / 1e9
    iqr_s = q3 - q1
    return median_s, iqr_s


def _time_fn_ns(fn, n_warmup: int, n_reps: int, cuda_sync: bool = False) -> tuple[float, float, float]:
    for _ in range(n_warmup):
        fn()
        if cuda_sync:
            _cuda_sync()

    gc.disable()
    times = []
    try:
        for _ in range(n_reps):
            if cuda_sync:
                _cuda_sync()
            t0 = time.perf_counter_ns()
            fn()
            if cuda_sync:
                _cuda_sync()
            t1 = time.perf_counter_ns()
            times.append(t1 - t0)
    finally:
        gc.enable()

    median_s, iqr_s = _summarize_ns(times)
    return median_s, iqr_s, float(np.sum(times)) / 1e9


def _load_ampc_features(dataset_size: int) -> tuple[np.ndarray, np.ndarray] | None:
    path = AMPC_100K if dataset_size <= 100_000 else AMPC_1000K
    if not path.exists():
        print(f'  AmpC dataset not found: {path}')
        return None

    import polars as pl

    from learnm8.features.extraction import extract_features

    print(f'  Loading {path.name}...')
    df = pl.read_csv(path)
    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    smiles = df[smiles_col].to_list()
    targets = df['dockscore'].to_numpy()

    print(f'  Extracting Morgan features for {len(smiles)} compounds...')
    features = extract_features(smiles_list=smiles, featurizer='morgan', n_jobs=-1)
    n = min(dataset_size, len(features))
    return features[:n].astype(np.float32), targets[:n]


def _benchmark_rf_fil(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    dataset_size: int,
) -> dict:
    from learnm8.learners.gpu.rf_fil import RfFilLearner
    from learnm8.learners.sklearn.random_forest import RandomForestLearner

    print(f'  Benchmarking rf_fil vs rf_cpu (n_train={len(X_train)}, n_test={len(X_test)})')

    X_train_f64 = X_train.astype(np.float64)
    y_train_f64 = y_train.astype(np.float64)
    X_test_f64 = X_test.astype(np.float64)

    cpu_learner = RandomForestLearner(n_estimators=100, random_state=42, n_jobs=-1)

    def train_cpu():
        cpu_learner.train(X_train_f64, y_train_f64)

    train_cpu()
    train_cpu_median, train_cpu_iqr, _ = _time_fn_ns(train_cpu, N_WARMUP, N_REPS)

    def predict_cpu():
        cpu_learner.predict(X_test_f64)

    predict_cpu()
    predict_cpu_median, predict_cpu_iqr, _ = _time_fn_ns(predict_cpu, N_WARMUP, N_REPS)

    gpu_learner = RfFilLearner(n_estimators=100, random_state=42, n_jobs=-1)

    def train_gpu():
        gpu_learner.train(X_train, y_train)

    _reset_vram_stats()
    train_gpu()
    train_gpu_median, train_gpu_iqr, _ = _time_fn_ns(train_gpu, N_WARMUP, N_REPS, cuda_sync=True)

    def predict_gpu():
        gpu_learner.predict(X_test)

    _reset_vram_stats()
    predict_gpu()
    predict_gpu_median, predict_gpu_iqr, _ = _time_fn_ns(predict_gpu, N_WARMUP, N_REPS, cuda_sync=True)
    vram_peak = _vram_peak_mb()

    speedup = predict_cpu_median / predict_gpu_median if predict_gpu_median > 0 else float('inf')

    return {
        'learner_id': 'rf_fil',
        'dataset_size': dataset_size,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_estimators': 100,
        'train_time_s': {'cpu': train_cpu_median, 'cpu_iqr': train_cpu_iqr,
                         'gpu': train_gpu_median, 'gpu_iqr': train_gpu_iqr},
        'predict_time_s': {'cpu': predict_cpu_median, 'cpu_iqr': predict_cpu_iqr,
                           'gpu': predict_gpu_median, 'gpu_iqr': predict_gpu_iqr},
        'warmup_time_s': None,
        'gpu_vram_peak_mb': vram_peak,
        'predict_speedup': speedup,
        'chunked': False,
        'n_warmup': N_WARMUP,
        'n_reps': N_REPS,
    }


def _benchmark_ridge_cuml(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    dataset_size: int,
) -> dict:
    from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
    from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner

    print(f'  Benchmarking ridge_cuml vs lr_cpu (n_train={len(X_train)}, n_test={len(X_test)})')

    X_train_f64 = X_train.astype(np.float64)
    y_train_f64 = y_train.astype(np.float64)
    X_test_f64 = X_test.astype(np.float64)

    cpu_learner = LinearRegressionLearner(alpha=0.1)

    def train_cpu():
        cpu_learner.train(X_train_f64, y_train_f64)

    train_cpu()
    train_cpu_median, train_cpu_iqr, _ = _time_fn_ns(train_cpu, N_WARMUP, N_REPS)

    def predict_cpu():
        cpu_learner.predict(X_test_f64)

    predict_cpu()
    predict_cpu_median, predict_cpu_iqr, _ = _time_fn_ns(predict_cpu, N_WARMUP, N_REPS)

    gpu_learner = RidgeCumlLearner(alpha=0.1, random_state=42)

    def train_gpu():
        gpu_learner.train(X_train_f64, y_train_f64)

    _reset_vram_stats()
    train_gpu()
    train_gpu_median, train_gpu_iqr, _ = _time_fn_ns(train_gpu, N_WARMUP, N_REPS, cuda_sync=True)

    def predict_gpu():
        gpu_learner.predict(X_test_f64)

    _reset_vram_stats()
    predict_gpu()
    predict_gpu_median, predict_gpu_iqr, _ = _time_fn_ns(predict_gpu, N_WARMUP, N_REPS, cuda_sync=True)
    vram_peak = _vram_peak_mb()

    _n, p = X_train.shape
    chunked = p > 5000

    speedup = predict_cpu_median / predict_gpu_median if predict_gpu_median > 0 else float('inf')

    return {
        'learner_id': 'ridge_cuml',
        'dataset_size': dataset_size,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'alpha': 0.1,
        'train_time_s': {'cpu': train_cpu_median, 'cpu_iqr': train_cpu_iqr,
                         'gpu': train_gpu_median, 'gpu_iqr': train_gpu_iqr},
        'predict_time_s': {'cpu': predict_cpu_median, 'cpu_iqr': predict_cpu_iqr,
                           'gpu': predict_gpu_median, 'gpu_iqr': predict_gpu_iqr},
        'warmup_time_s': None,
        'gpu_vram_peak_mb': vram_peak,
        'predict_speedup': speedup,
        'chunked': chunked,
        'n_warmup': N_WARMUP,
        'n_reps': N_REPS,
    }


def _benchmark_xgb_cuda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    dataset_size: int,
) -> dict:
    from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner

    print(f'  Benchmarking xgb_cuda vs xgb_cpu (n_train={len(X_train)}, n_test={len(X_test)})')

    X_train_f64 = X_train.astype(np.float64)
    y_train_f64 = y_train.astype(np.float64)
    X_test_f64 = X_test.astype(np.float64)

    cpu_learner = XGBoostLearner(n_estimators=100, random_state=42, device='cpu', n_jobs=-1)

    def train_cpu():
        cpu_learner.train(X_train_f64, y_train_f64)

    train_cpu()
    train_cpu_median, train_cpu_iqr, _ = _time_fn_ns(train_cpu, N_WARMUP, N_REPS)

    def predict_cpu():
        cpu_learner.predict(X_test_f64)

    predict_cpu()
    predict_cpu_median, predict_cpu_iqr, _ = _time_fn_ns(predict_cpu, N_WARMUP, N_REPS)

    gpu_learner = XGBoostLearner(n_estimators=100, random_state=42, device='cuda', n_jobs=-1)

    def train_gpu():
        gpu_learner.train(X_train_f64, y_train_f64)

    _reset_vram_stats()
    train_gpu()
    train_gpu_median, train_gpu_iqr, _ = _time_fn_ns(train_gpu, N_WARMUP, N_REPS, cuda_sync=True)

    def predict_gpu():
        gpu_learner.predict(X_test_f64)

    _reset_vram_stats()
    predict_gpu()
    predict_gpu_median, predict_gpu_iqr, _ = _time_fn_ns(predict_gpu, N_WARMUP, N_REPS, cuda_sync=True)
    vram_peak = _vram_peak_mb()

    speedup = predict_cpu_median / predict_gpu_median if predict_gpu_median > 0 else float('inf')

    return {
        'learner_id': 'xgb_cuda',
        'dataset_size': dataset_size,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_estimators': 100,
        'train_time_s': {'cpu': train_cpu_median, 'cpu_iqr': train_cpu_iqr,
                         'gpu': train_gpu_median, 'gpu_iqr': train_gpu_iqr},
        'predict_time_s': {'cpu': predict_cpu_median, 'cpu_iqr': predict_cpu_iqr,
                           'gpu': predict_gpu_median, 'gpu_iqr': predict_gpu_iqr},
        'warmup_time_s': None,
        'gpu_vram_peak_mb': vram_peak,
        'predict_speedup': speedup,
        'chunked': False,
        'n_warmup': N_WARMUP,
        'n_reps': N_REPS,
    }


def _synthetic_data(n: int, n_features: int = 1024, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    chunk_size = 100_000
    if n <= chunk_size:
        X = rng.random((n, n_features)).astype(np.float32)
    else:
        chunks = []
        remaining = n
        while remaining > 0:
            k = min(remaining, chunk_size)
            chunks.append(rng.random((k, n_features)).astype(np.float32))
            remaining -= k
        X = np.concatenate(chunks, axis=0)
        del chunks
    y = (X[:, 0] * 3.0 + rng.normal(0, 0.1, n)).astype(np.float32)
    return X, y


def _get_hardware_info() -> dict:
    info = {
        'platform': platform.platform(),
        'python': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
    }
    try:
        info['cuda'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info['gpu_vram_total_gb'] = round(props.total_memory / (1024 ** 3), 2)
    except Exception:
        pass
    return info


def _save_result(result: dict, label: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f'gpu_benchmark_{label}.json'
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Saved: {path}')


def _benchmark_batched_prediction(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    dataset_size: int,
) -> list[dict]:
    import math

    from learnm8.core.batching import estimate_batch_size, get_available_memory

    results = []
    learner_configs = []

    try:
        from learnm8.learners.gpu.rf_fil import RfFilLearner
        learner_configs.append(('rf_fil', RfFilLearner(n_estimators=100, random_state=42, n_jobs=-1)))
    except Exception as e:
        print(f'  Skipping rf_fil batched benchmark: {e}')

    try:
        from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
        learner_configs.append(('ridge_cuml', RidgeCumlLearner(alpha=0.1, random_state=42)))
    except Exception as e:
        print(f'  Skipping ridge_cuml batched benchmark: {e}')

    try:
        from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner
        learner_configs.append(('xgb_cuda', XGBoostLearner(n_estimators=100, random_state=42, device='cuda', n_jobs=-1)))
    except Exception as e:
        print(f'  Skipping xgb_cuda batched benchmark: {e}')

    for learner_id, learner in learner_configs:
        print(f'  Batched prediction benchmark: {learner_id} (n_test={len(X_test)})')
        try:
            learner.train(X_train.copy(), y_train.copy())

            n_features = X_test.shape[1]
            from learnm8.core.batching import _get_learner_device
            device = _get_learner_device(learner)
            available_mem = get_available_memory(device)
            profile = learner.memory_profile(n_features)
            batch_size = estimate_batch_size(learner, len(X_test), n_features, device)
            n_batches = math.ceil(len(X_test) / batch_size)
            print(
                f'    memory_profile: {profile}')
            print(
                f'    available={available_mem / (1024**3):.1f} GiB, '
                f'batch_size={batch_size:,}, n_batches={n_batches}'
            )

            _reset_vram_stats()

            def run_batched(_bs=batch_size, _lr=learner):
                all_preds = []
                for i in range(0, len(X_test), _bs):
                    chunk = X_test[i:i + _bs]
                    preds, _ = _lr.predict(chunk)
                    all_preds.append(preds)
                return np.concatenate(all_preds)

            run_batched()
            median_s, iqr_s, _ = _time_fn_ns(run_batched, 1, 3, cuda_sync=True)
            vram_peak = _vram_peak_mb()

            def run_unbatched(_lr=learner):
                return _lr.predict(X_test)

            unbatched_median, unbatched_iqr, _ = _time_fn_ns(
                run_unbatched, 1, 3, cuda_sync=True
            )

            result = {
                'learner_id': f'{learner_id}_batched',
                'dataset_size': dataset_size,
                'n_test': len(X_test),
                'batch_size': batch_size,
                'n_batches': n_batches,
                'memory_profile': profile,
                'predict_time_s': {
                    'batched': median_s, 'batched_iqr': iqr_s,
                    'unbatched': unbatched_median, 'unbatched_iqr': unbatched_iqr,
                },
                'gpu_vram_peak_mb': vram_peak,
                'n_reps': 3,
            }
            _save_result(result, f'{learner_id}_batched_{dataset_size}')
            results.append(result)
            print(
                f'    {learner_id}: batched={median_s:.4f}s unbatched={unbatched_median:.4f}s '
                f'(batch_size={batch_size:,}, {n_batches} batches)'
            )
        except Exception as e:
            print(f'    {learner_id} batched benchmark failed: {e}')

    return results


def main():
    import datetime

    print('=' * 60)
    print('GPU Learner Benchmark Suite')
    print(f'Started: {datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")}')
    hw = _get_hardware_info()
    for k, v in hw.items():
        print(f'  {k}: {v}')
    print('=' * 60)

    all_results = []

    for dataset_size in DATASET_SIZES:
        print(f'\n--- Dataset size: {dataset_size:,} ---')
        data = _load_ampc_features(dataset_size)

        if data is not None:
            features, targets = data
            n_use = min(dataset_size, len(features))
            features, targets = features[:n_use], targets[:n_use]
            source = 'AmpC'
        else:
            print(f'  Using synthetic data (n={dataset_size})')
            features, targets = _synthetic_data(dataset_size)
            source = 'synthetic'

        n_train = int(len(features) * 0.8)
        X_train, X_test = features[:n_train], features[n_train:]
        y_train, _y_test = targets[:n_train], targets[n_train:]

        print(f'  Source: {source}, train={len(X_train)}, test={len(X_test)}, features={features.shape[1]}')

        try:
            result = _benchmark_rf_fil(X_train, y_train, X_test, dataset_size)
            result['data_source'] = source
            result['hardware'] = hw
            _save_result(result, f'rf_fil_{dataset_size}')
            all_results.append(result)
            print(f'  rf_fil predict speedup: {result["predict_speedup"]:.2f}x')
        except Exception as e:
            print(f'  rf_fil benchmark failed: {e}')

        try:
            result = _benchmark_ridge_cuml(X_train, y_train, X_test, dataset_size)
            result['data_source'] = source
            result['hardware'] = hw
            _save_result(result, f'ridge_cuml_{dataset_size}')
            all_results.append(result)
            print(f'  ridge_cuml predict speedup: {result["predict_speedup"]:.2f}x')
        except Exception as e:
            print(f'  ridge_cuml benchmark failed: {e}')

        try:
            result = _benchmark_xgb_cuda(X_train, y_train, X_test, dataset_size)
            result['data_source'] = source
            result['hardware'] = hw
            _save_result(result, f'xgb_cuda_{dataset_size}')
            all_results.append(result)
            print(f'  xgb_cuda predict speedup: {result["predict_speedup"]:.2f}x')
        except Exception as e:
            print(f'  xgb_cuda benchmark failed: {e}')

        print(f'\n  --- Memory-aware batched prediction (dataset_size={dataset_size:,}) ---')
        try:
            batched_results = _benchmark_batched_prediction(X_train, y_train, X_test, dataset_size)
            for br in batched_results:
                br['data_source'] = source
                br['hardware'] = hw
            all_results.extend(batched_results)
        except Exception as e:
            print(f'  Batched prediction benchmark failed: {e}')

    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    for r in all_results:
        if 'batched' in r.get('learner_id', ''):
            pred_batched = r['predict_time_s']['batched']
            print(
                f'  {r["learner_id"]:20s}  size={r["dataset_size"]:>8,}  '
                f'batched={pred_batched:.4f}s  batch_size={r["batch_size"]:,}'
            )
        else:
            pred_cpu = r['predict_time_s']['cpu']
            pred_gpu = r['predict_time_s']['gpu']
            print(
                f'  {r["learner_id"]:20s}  size={r["dataset_size"]:>8,}  '
                f'cpu={pred_cpu:.4f}s  gpu={pred_gpu:.4f}s  '
                f'speedup={r["predict_speedup"]:.2f}x'
            )

    summary_path = RESULTS_DIR / 'gpu_benchmark_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({'results': all_results, 'hardware': hw}, f, indent=2)
    print(f'\nSummary saved: {summary_path}')


if __name__ == '__main__':
    main()

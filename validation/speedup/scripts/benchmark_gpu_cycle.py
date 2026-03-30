from __future__ import annotations

import datetime
import gc
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

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
    from cuml.linear_model import Ridge as _CumlRidge  # noqa: F401
except ImportError:
    print('Skipping benchmark: cuml not available (install RAPIDS for GPU support)')
    sys.exit(0)

from validation.lib.data_loading import load_validation_dataset  # noqa: E402
from validation.lib.dataset_config import get_dataset_info  # noqa: E402

RESULTS_DIR = Path(__file__).parent.parent / 'results'
N_CYCLES = 5
BATCH_FRACTION = 0.01
FEATURIZER = 'morgan'
RANDOM_STATE = 42

CONFIGS = [
    {
        'label': 'rf_fil_gpu',
        'learner': 'rf_fil',
        'device': 'cuda',
        'is_gpu': True,
        'pair': 'rf',
        'dataset': 'ampc_1000k',
    },
    {
        'label': 'rf_cpu',
        'learner': 'rf',
        'device': 'cpu',
        'is_gpu': False,
        'pair': 'rf',
        'dataset': 'ampc_1000k',
    },
    {
        'label': 'ridge_cuml_gpu',
        'learner': 'ridge_cuml',
        'device': 'cuda',
        'is_gpu': True,
        'pair': 'ridge',
        'dataset': 'ampc_100k',
    },
    {
        'label': 'lr_cpu',
        'learner': 'lr_instance',
        'device': 'cpu',
        'is_gpu': False,
        'pair': 'ridge',
        'dataset': 'ampc_100k',
    },
    {
        'label': 'xgb_cuda',
        'learner': 'xgb',
        'device': 'cuda',
        'is_gpu': True,
        'pair': 'xgb',
        'dataset': 'ampc_1000k',
    },
    {
        'label': 'xgb_cpu',
        'learner': 'xgb',
        'device': 'cpu',
        'is_gpu': False,
        'pair': 'xgb',
        'dataset': 'ampc_1000k',
    },
]


def _get_hardware_info() -> dict:
    info = {
        'platform': platform.platform(),
        'python': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
    }
    try:
        info['cuda'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info['gpu_vram_total_gb'] = round(props.total_memory / (1024**3), 2)
    except Exception:
        pass
    return info


def _result_path(label: str) -> Path:
    return RESULTS_DIR / f'gpu_cycle_{label}.json'


def _should_skip(label: str) -> bool:
    return _result_path(label).exists()


def _validate_resume(loaded: dict) -> bool:
    return (
        loaded.get('n_cycles') == N_CYCLES
        and loaded.get('batch_fraction') == BATCH_FRACTION
        and loaded.get('featurizer') == FEATURIZER
    )


def _pre_warm_cache(compound_pool, cache_dir: Path) -> None:
    from learnm8.features.extraction import extract_features

    smiles = compound_pool['SMILES'].to_list()
    print(f'  Pre-warming feature cache for {len(smiles):,} compounds...')
    t0 = time.perf_counter_ns()
    extract_features(smiles_list=smiles, featurizer=FEATURIZER, cache_dir=cache_dir)
    elapsed = (time.perf_counter_ns() - t0) / 1e9
    print(f'  Feature cache ready ({elapsed:.1f}s)')


def _gpu_cleanup() -> None:
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


def _run_config(
    config: dict,
    compound_pool,
    metadata: dict,
    dataset_info: dict,
    cache_dir: Path,
) -> tuple[dict, float]:
    from learnm8 import run_active_learning
    from learnm8.oracles.csv_oracle import CSVOracle

    oracle = CSVOracle(str(dataset_info['path']), id_column=dataset_info['id_column'])

    learner = config['learner']
    if learner == 'lr_instance':
        from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner

        learner = LinearRegressionLearner(alpha=0.1, random_state=RANDOM_STATE)

    output_dir = tempfile.mkdtemp(prefix=f'gpu_cycle_{config["label"]}_')
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter_ns()
        result = run_active_learning(
            compound_pool=compound_pool,
            oracle=oracle,
            learner=learner,
            target_col=metadata['target_column'],
            featurizer=FEATURIZER,
            n_cycles=N_CYCLES,
            batch_fraction=BATCH_FRACTION,
            strategy='greedy',
            initial_strategy='random',
            score_direction=metadata['score_direction'],
            mode='benchmark',
            output_dir=output_dir,
            cache_dir=cache_dir,
            random_state=RANDOM_STATE,
            device=config['device'],
            n_jobs=-1,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        wall_clock_s = (time.perf_counter_ns() - t0) / 1e9
        return result, wall_clock_s
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def _extract_result(
    api_result: dict,
    config: dict,
    wall_clock_s: float,
    hw: dict,
) -> dict:
    cycle_metrics = api_result.get('cycle_metrics', [])
    aggregate_metrics = api_result.get('aggregate_metrics', {})

    per_cycle = []
    total_train_time = 0.0
    total_predict_time = 0.0

    for cm in cycle_metrics:
        train_t = cm.get('training_time') or 0.0
        pred_t = cm.get('prediction_time') or 0.0
        total_train_time += train_t
        total_predict_time += pred_t
        per_cycle.append(
            {
                'cycle': cm.get('cycle'),
                'cumulative_labeled': cm.get('cumulative_labeled'),
                'remaining_unlabeled': cm.get('remaining_unlabeled'),
                'training_time': cm.get('training_time'),
                'prediction_time': cm.get('prediction_time'),
                'acquisition_time': cm.get('acquisition_time'),
                'oracle_time': cm.get('oracle_time'),
                'total_time': cm.get('total_time'),
                'top_10_discovery': cm.get('top_10_discovery'),
                'top_100_discovery': cm.get('top_100_discovery'),
                'top_1000_discovery': cm.get('top_1000_discovery'),
                'top_1_pct_discovery': cm.get('top_1_pct_discovery'),
                'unlabeled_spearman_correlation': cm.get(
                    'unlabeled_spearman_correlation'
                ),
                'unlabeled_ef_1_0': cm.get('unlabeled_ef_1_0'),
            }
        )

    learner_name = config['learner']
    if learner_name == 'lr_instance':
        learner_name = 'LinearRegressionLearner'

    return {
        'label': config['label'],
        'learner': learner_name,
        'device': config['device'],
        'is_gpu': config['is_gpu'],
        'pair': config['pair'],
        'dataset': config['dataset'],
        'n_cycles': N_CYCLES,
        'batch_fraction': BATCH_FRACTION,
        'featurizer': FEATURIZER,
        'random_state': RANDOM_STATE,
        'total_wall_clock_s': round(wall_clock_s, 3),
        'learner_time_s': round(total_train_time + total_predict_time, 3),
        'total_train_time_s': round(total_train_time, 3),
        'total_predict_time_s': round(total_predict_time, 3),
        'per_cycle_metrics': per_cycle,
        'aggregate_metrics': {
            'final_top_10_discovery': aggregate_metrics.get('final_top_10_discovery'),
            'final_top_100_discovery': aggregate_metrics.get('final_top_100_discovery'),
            'final_top_1000_discovery': aggregate_metrics.get(
                'final_top_1000_discovery'
            ),
            'final_unlabeled_spearman': aggregate_metrics.get(
                'final_unlabeled_spearman'
            ),
            'avg_top_10_discovery': aggregate_metrics.get('avg_top_10_discovery'),
        },
        'hardware': hw,
        'timestamp': datetime.datetime.now(datetime.UTC).isoformat(),
    }


def _print_summary(all_results: list[dict]) -> None:
    pairs = {}
    for r in all_results:
        pair = r['pair']
        if pair not in pairs:
            pairs[pair] = {}
        key = 'gpu' if r['is_gpu'] else 'cpu'
        pairs[pair][key] = r

    header = (
        f'{"Pair":<7} {"Learner":<20} {"Device":<6} {"Dataset":<8} '
        f'{"Wall Clock":>10} {"Learner Time":>12} {"Speedup":>8} '
        f'{"Top-10 Disc":>11} {"Spearman":>9}'
    )
    print('\n' + '=' * len(header))
    print('GPU Active Learning Cycle Benchmark Results')
    print('=' * len(header))
    print(header)
    print('-' * len(header))

    for pair_name, pair_data in pairs.items():
        cpu_result = pair_data.get('cpu')
        gpu_result = pair_data.get('gpu')

        for r in [gpu_result, cpu_result]:
            if r is None:
                continue

            learner_time = r.get('learner_time_s', 0)
            wall_clock = r.get('total_wall_clock_s', 0)
            agg = r.get('aggregate_metrics', {})
            disc = agg.get('final_top_10_discovery')
            spearman = agg.get('final_unlabeled_spearman')

            if r['is_gpu'] and cpu_result and cpu_result.get('learner_time_s', 0) > 0:
                speedup = f'{cpu_result["learner_time_s"] / learner_time:.1f}x'
            elif not r['is_gpu']:
                speedup = '(base)'
            else:
                speedup = 'N/A'

            disc_str = f'{disc:.1f}%' if disc is not None else 'N/A'
            spear_str = f'{spearman:.3f}' if spearman is not None else 'N/A'

            print(
                f'{pair_name:<7} {r["learner"]:<20} {r["device"]:<6} '
                f'{r["dataset"]:<8} {wall_clock:>9.1f}s {learner_time:>11.1f}s '
                f'{speedup:>8} {disc_str:>11} {spear_str:>9}'
            )

    print('-' * len(header))
    print(
        'Speedup = CPU learner_time / GPU learner_time '
        '(isolated from feature extraction + oracle overhead)'
    )
    print()


def main() -> None:
    print('=' * 60)
    print('GPU Active Learning Cycle Benchmark')
    print(
        f'Started: {datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")}'
    )
    hw = _get_hardware_info()
    for k, v in hw.items():
        print(f'  {k}: {v}')
    print('=' * 60)

    needed_datasets = {c['dataset'] for c in CONFIGS}
    datasets: dict[str, tuple] = {}

    for dataset_name in sorted(needed_datasets):
        try:
            info = get_dataset_info(dataset_name)
        except KeyError:
            print(f"  Dataset '{dataset_name}' not in registry, skipping configs")
            continue
        if not Path(info['path']).exists():
            print(f'  Dataset not found: {info["path"]}')
            continue
        df, meta = load_validation_dataset(dataset_name)
        datasets[dataset_name] = (df, meta, info)
        print(f'  Loaded {dataset_name}: {len(df):,} compounds')

    if not datasets:
        print('No datasets available. Exiting.')
        sys.exit(0)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = RESULTS_DIR / '.benchmark_feature_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    print('\nPre-warming feature caches...')
    for _dataset_name, (df, _meta, _info) in datasets.items():
        _pre_warm_cache(df, cache_dir)

    all_results: list[dict] = []

    for config in CONFIGS:
        label = config['label']
        dataset_name = config['dataset']

        if dataset_name not in datasets:
            print(f"\n  Skipping {label}: dataset '{dataset_name}' not available")
            continue

        if _should_skip(label):
            with open(_result_path(label)) as f:
                loaded = json.load(f)
            if _validate_resume(loaded):
                print(f'\n  Skipping {label} (result exists)')
                all_results.append(loaded)
                continue
            print(f'\n  Stale result for {label} (params mismatch), re-running')

        df, meta, info = datasets[dataset_name]
        print(f'\n  Running {label} ({dataset_name}, device={config["device"]})...')

        try:
            api_result, wall_clock = _run_config(config, df, meta, info, cache_dir)
            result = _extract_result(api_result, config, wall_clock, hw)

            with open(_result_path(label), 'w') as f:
                json.dump(result, f, indent=2)

            print(
                f'  {label}: wall_clock={wall_clock:.1f}s, '
                f'learner_time={result["learner_time_s"]:.1f}s'
            )
            all_results.append(result)
        except Exception as e:
            print(f'  {label} FAILED: {e}')

        _gpu_cleanup()

    if all_results:
        _print_summary(all_results)
    else:
        print('\nNo results collected.')

    print(f'Results directory: {RESULTS_DIR}')


if __name__ == '__main__':
    main()

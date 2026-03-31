#!/usr/bin/env python
"""Benchmark: GPyTorch GP active learning on molecular screening datasets.

Tests ExactGP (LOVE-enabled) and SVGP (M=256, 512, 1024) in full active
learning loops using run_active_learning().

Configurations:
  - ExactGP-LOVE:  AmpC 100K pool, 1% batch (1000/cycle), 10 cycles, greedy
  - SVGP-M256:     AmpC 1M pool,   1% batch (10K/cycle),  10 cycles, greedy
  - SVGP-M512:     AmpC 1M pool,   1% batch (10K/cycle),  10 cycles, greedy
  - SVGP-M1024:    AmpC 1M pool,   1% batch (10K/cycle),  10 cycles, greedy

Expected runtime: ~10-15 hours total on single GPU.
Results saved incrementally — can be resumed if interrupted.

Usage:
    python validation/scripts/benchmark_gp_active_learning.py
    python validation/scripts/benchmark_gp_active_learning.py --only exactgp
    python validation/scripts/benchmark_gp_active_learning.py --only svgp-m512
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import polars as pl
from rich.console import Console
from rich.table import Table

from learnm8 import run_active_learning
from learnm8.learners.gpytorch.gpu_gp import GPyTorchGPLearner
from learnm8.learners.gpytorch.svgp import SVGPLearner
from learnm8.oracles import CSVOracle
from validation.lib import get_dataset_info, get_dataset_path, load_validation_dataset

logger = logging.getLogger(__name__)
console = Console()

OUTPUT_DIR = Path('validation/reports/performance/gp_active_learning_benchmark')
CACHE_DIR = Path('validation/.shared_cache')
SENTINEL_FILES = ['compounds_final.csv', 'cycle_metrics.csv', 'selection_history.csv']
RANDOM_STATE = 42

COLORS = {
    'ExactGP-LOVE': '#2ca02c',
    'SVGP-M256': '#d62728',
    'SVGP-M512': '#9467bd',
    'SVGP-M1024': '#8c564b',
}
LABELS = {
    'ExactGP-LOVE': 'ExactGP-LOVE (100K pool)',
    'SVGP-M256': 'SVGP M=256 (1M pool)',
    'SVGP-M512': 'SVGP M=512 (1M pool)',
    'SVGP-M1024': 'SVGP M=1024 (1M pool)',
}
MARKERS = {
    'ExactGP-LOVE': 'o',
    'SVGP-M256': 's',
    'SVGP-M512': '^',
    'SVGP-M1024': 'D',
}

BENCHMARK_CONFIGS = [
    {
        'name': 'ExactGP-LOVE',
        'dataset': 'ampc_100k',
        'learner_factory': lambda: GPyTorchGPLearner(
            device='cuda', random_state=RANDOM_STATE
        ),
        'batch_fraction': 0.01,
        'n_cycles': 10,
    },
    {
        'name': 'SVGP-M256',
        'dataset': 'ampc_1000k',
        'learner_factory': lambda: SVGPLearner(
            n_inducing=256, device='cuda', random_state=RANDOM_STATE
        ),
        'batch_fraction': 0.01,
        'n_cycles': 10,
    },
    {
        'name': 'SVGP-M512',
        'dataset': 'ampc_1000k',
        'learner_factory': lambda: SVGPLearner(
            n_inducing=512, device='cuda', random_state=RANDOM_STATE
        ),
        'batch_fraction': 0.01,
        'n_cycles': 10,
    },
    {
        'name': 'SVGP-M1024',
        'dataset': 'ampc_1000k',
        'learner_factory': lambda: SVGPLearner(
            n_inducing=1024, device='cuda', random_state=RANDOM_STATE
        ),
        'batch_fraction': 0.01,
        'n_cycles': 10,
    },
]


def check_existing(config_dir: Path) -> bool:
    return all((config_dir / f).exists() for f in SENTINEL_FILES)


def load_existing_metrics(config_dir: Path) -> list[dict]:
    metrics_path = config_dir / 'cycle_metrics.csv'
    if not metrics_path.exists():
        return []
    df = pl.read_csv(metrics_path, comment_prefix='#')
    return df.to_dicts()


def run_config(config: dict) -> dict:
    name = config['name']
    dataset_name = config['dataset']
    config_dir = OUTPUT_DIR / name

    if check_existing(config_dir):
        console.print(f'  [yellow]Skipping {name} — results exist[/yellow]')
        metrics = load_existing_metrics(config_dir)
        return {
            'name': name,
            'dataset': dataset_name,
            'cycle_metrics': metrics,
            'elapsed_time': None,
            'status': 'skipped',
        }

    console.print(f'  Loading dataset: {dataset_name}')
    compound_pool, metadata = load_validation_dataset(dataset_name)

    dataset_info = get_dataset_info(dataset_name)
    dataset_path = get_dataset_path(dataset_name)
    oracle = CSVOracle(str(dataset_path), id_column=dataset_info['id_column'])

    learner = config['learner_factory']()
    pool_size = len(compound_pool)
    batch_size = int(pool_size * config['batch_fraction'])

    console.print(f'  Learner: {learner.get_name()}')
    console.print(f'  Pool size: {pool_size:,}')
    console.print(f'  Batch size: {batch_size:,} ({config["batch_fraction"]:.0%})')
    console.print(f'  Cycles: {config["n_cycles"]}')

    start = time.perf_counter()
    results = run_active_learning(
        compound_pool=compound_pool,
        oracle=oracle,
        learner=learner,
        target_col=metadata['target_column'],
        featurizer='morgan',
        n_cycles=config['n_cycles'],
        batch_fraction=config['batch_fraction'],
        strategy='greedy',
        score_direction=metadata['score_direction'],
        mode='benchmark',
        output_dir=str(config_dir),
        cache_dir=str(CACHE_DIR),
        random_state=RANDOM_STATE,
    )
    elapsed = time.perf_counter() - start

    console.print(f'  [green]Completed in {elapsed / 60:.1f} min[/green]')

    return {
        'name': name,
        'dataset': dataset_name,
        'cycle_metrics': results.get('cycle_metrics', []),
        'elapsed_time': elapsed,
        'status': 'completed',
    }


def plot_results(all_results: list[dict], output_dir: Path):
    completed = [r for r in all_results if r['cycle_metrics']]
    if not completed:
        console.print('[yellow]No completed results to plot[/yellow]')
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_top1, ax_top01), (ax_train, ax_total) = axes

    for r in completed:
        name = r['name']
        metrics = r['cycle_metrics']
        color = COLORS.get(name, '#888888')
        label = LABELS.get(name, name)
        marker = MARKERS.get(name, 'o')

        cycles = [m['cycle'] for m in metrics]
        kw = dict(color=color, label=label, linewidth=2, marker=marker,
                  markersize=5, zorder=3)

        top1 = [m.get('top_1_pct_discovery') for m in metrics]
        ax_top1.plot(cycles, top1, **kw)

        top01 = [m.get('top_0_1_pct_discovery') for m in metrics]
        ax_top01.plot(cycles, top01, **kw)

        train_t = [m.get('training_time') for m in metrics]
        valid_cycles_t = [c for c, t in zip(cycles, train_t, strict=False) if t is not None]
        valid_train_t = [t for t in train_t if t is not None]
        ax_train.plot(valid_cycles_t, valid_train_t, **kw)

        cum_labeled = [m.get('cumulative_labeled') for m in metrics]
        total_t = [m.get('total_time') for m in metrics]
        valid_labeled = [c for c, t in zip(cum_labeled, total_t, strict=False)
                         if c is not None and t is not None]
        valid_total_t = [t for c, t in zip(cum_labeled, total_t, strict=False)
                         if c is not None and t is not None]
        ax_total.plot(valid_labeled, valid_total_t, **kw)

    ref_metrics = completed[0]['cycle_metrics']
    ref_cycles = [m['cycle'] for m in ref_metrics]
    ref_total = (ref_metrics[0].get('cumulative_labeled', 0)
                 + ref_metrics[0].get('remaining_unlabeled', 0)
                 + ref_metrics[0].get('cumulative_pruned', 0))
    ref_pct = [m.get('cumulative_labeled', 0) / ref_total * 100
               for m in ref_metrics] if ref_total > 0 else ref_cycles

    for ax_main, title_letter, title_metric in [
        (ax_top1, 'A', 'Top 1% Discovery'),
        (ax_top01, 'B', 'Top 0.1% Discovery'),
    ]:
        ax_main.set_xlabel('Cycle', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Discovery (%)', fontsize=12, fontweight='bold')
        ax_main.set_title(f'({title_letter}) {title_metric}', fontsize=13, fontweight='bold')
        ax_main.legend(fontsize=9, framealpha=0.9)
        ax_main.set_ylim(bottom=0)

        ax_pct = ax_main.twiny()
        ax_pct.set_xlim(ax_main.get_xlim())
        ax_pct.set_xticks(ref_cycles)
        ax_pct.set_xticklabels([f'{p:.0f}%' for p in ref_pct], fontsize=8)
        ax_pct.set_xlabel('% Pool Explored', fontsize=10, fontweight='bold')

    ax_train.set_xlabel('Cycle', fontsize=12, fontweight='bold')
    ax_train.set_ylabel('Time (s)', fontsize=12, fontweight='bold')
    ax_train.set_title('(C) Training Time per Cycle', fontsize=13, fontweight='bold')
    ax_train.legend(fontsize=9, framealpha=0.9)

    ax_total.set_xscale('log')
    ax_total.set_xlabel('Cumulative Labeled Compounds', fontsize=12, fontweight='bold')
    ax_total.set_ylabel('Time (s)', fontsize=12, fontweight='bold')
    ax_total.set_title('(D) Total Cycle Time vs Training Set Size', fontsize=13, fontweight='bold')
    ax_total.legend(fontsize=9, framealpha=0.9)

    for ax in axes.flat:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / 'benchmark_active_learning.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    console.print(f'\nPlot saved to {out_path}')


def save_summary(all_results: list[dict], output_dir: Path):
    rows = []
    for r in all_results:
        final = r['cycle_metrics'][-1] if r['cycle_metrics'] else {}
        rows.append(
            {
                'config': r['name'],
                'dataset': r['dataset'],
                'status': r['status'],
                'elapsed_time_s': r.get('elapsed_time'),
                'top_1_pct_discovery': final.get('top_1_pct_discovery'),
                'top_0_1_pct_discovery': final.get('top_0_1_pct_discovery'),
                'unlabeled_spearman_correlation': final.get(
                    'unlabeled_spearman_correlation'
                ),
                'top_10_discovery': final.get('top_10_discovery'),
                'top_100_discovery': final.get('top_100_discovery'),
            }
        )

    summary_df = pl.DataFrame(rows)
    summary_path = output_dir / 'summary.csv'
    summary_df.write_csv(summary_path)
    console.print(f'\nSummary saved to {summary_path}')


def print_summary(all_results: list[dict]):
    table = Table(title='GP Active Learning Benchmark Results')
    table.add_column('Config', style='cyan')
    table.add_column('Dataset')
    table.add_column('Status')
    table.add_column('Time (min)', justify='right')
    table.add_column('Top 1%', justify='right')
    table.add_column('Top 0.1%', justify='right')
    table.add_column('Spearman', justify='right')

    for r in all_results:
        final = r['cycle_metrics'][-1] if r['cycle_metrics'] else {}
        elapsed = f'{r["elapsed_time"] / 60:.1f}' if r.get('elapsed_time') else '\u2014'

        top1 = final.get('top_1_pct_discovery')
        top1_str = f'{top1:.1%}' if top1 is not None else '\u2014'

        top01 = final.get('top_0_1_pct_discovery')
        top01_str = f'{top01:.1%}' if top01 is not None else '\u2014'

        spearman = final.get('unlabeled_spearman_correlation')
        spearman_str = f'{spearman:.4f}' if spearman is not None else '\u2014'

        status = r['status']
        if status == 'completed':
            status_str = '[green]completed[/green]'
        elif status == 'skipped':
            status_str = '[yellow]skipped[/yellow]'
        else:
            status_str = f'[red]{status}[/red]'

        table.add_row(
            r['name'],
            r['dataset'],
            status_str,
            elapsed,
            top1_str,
            top01_str,
            spearman_str,
        )

    console.print()
    console.print(table)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark GPyTorch GP models in active learning'
    )
    parser.add_argument(
        '--only',
        type=str,
        default=None,
        help='Run only the specified config (case-insensitive match on name)',
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    args = parse_args()

    configs = BENCHMARK_CONFIGS
    if args.only:
        target = args.only.lower()
        configs = [c for c in configs if target in c['name'].lower()]
        if not configs:
            available = ', '.join(c['name'] for c in BENCHMARK_CONFIGS)
            console.print(
                f"[red]No config matching '{args.only}'. Available: {available}[/red]"
            )
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    console.print('[bold]GP Active Learning Benchmark[/bold]')
    console.print(f'Output directory: {OUTPUT_DIR}')
    console.print(f'Feature cache: {CACHE_DIR}')
    console.print(f'Configs to run: {len(configs)}')
    console.print()

    all_results = []
    benchmark_start = time.perf_counter()

    for i, config in enumerate(configs, 1):
        console.rule(f'[bold]Config {i}/{len(configs)}: {config["name"]}[/bold]')

        try:
            result = run_config(config)
        except Exception as e:
            logger.error('Config %s failed: %s', config['name'], e, exc_info=True)
            console.print(f'  [red]FAILED: {e}[/red]')
            result = {
                'name': config['name'],
                'dataset': config['dataset'],
                'cycle_metrics': [],
                'elapsed_time': None,
                'status': f'failed: {type(e).__name__}: {e}',
            }

        all_results.append(result)
        console.print()

    total_time = time.perf_counter() - benchmark_start
    console.print(f'[bold]Total benchmark time: {total_time / 60:.1f} min[/bold]')

    save_summary(all_results, OUTPUT_DIR)
    plot_results(all_results, OUTPUT_DIR)
    print_summary(all_results)


if __name__ == '__main__':
    main()

"""Trace host RAM (PSS) and GPU VRAM over a full 1M-compound active-learning run.

Monitor mode (default): launches the active-learning workload as a child
subprocess, samples its whole process tree every ``--interval`` seconds, then
writes a timeline plot and a raw-sample CSV. Running the workload as a child
keeps the sampler's own footprint (psutil, matplotlib) out of the measurement
and captures joblib/multiprocessing forks by walking the process tree.

Child mode (``--child``): runs ``run_active_learning()`` with the
rf_fil / morgan / greedy config used by the prior 1M AmpC run, and emits phase
events (cycle boundaries, train start/end, predict end) via a logging hook.

Memory metric is PSS (Proportional Set Size) summed across the tree, so shared
interpreter/library pages are counted once rather than once per fork. GPU VRAM
is sampled per-process when the driver reports it, else as whole-GPU used minus
a pre-run baseline.

Usage:
    conda run --no-capture-output -n learnm8 \\
        python scripts/benchmarks/ram_timeline_1m.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(
    '/home/tony/Compound_Libraries/LearnM8_datasets/AmpC/subsampled_data/'
    'AmpC_screen_1000K_clean.csv'
)
DEFAULT_OUTPUT = REPO_ROOT / 'output' / 'ram_timeline_1m'


def _nvml_init() -> tuple[Any, list]:
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        return pynvml, [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
    except Exception:
        return None, []


def _gpu_total_used(nvml: Any, handles: list) -> int:
    if nvml is None:
        return 0
    total = 0
    for h in handles:
        with contextlib.suppress(Exception):
            total += int(nvml.nvmlDeviceGetMemoryInfo(h).used)
    return total


def _gpu_proc_used(nvml: Any, handles: list, pids: set[int]) -> int:
    if nvml is None:
        return 0
    total = 0
    for h in handles:
        try:
            procs = nvml.nvmlDeviceGetComputeRunningProcesses(h)
        except Exception:
            continue
        for p in procs:
            mem = getattr(p, 'usedGpuMemory', None)
            if mem and p.pid in pids:
                total += int(mem)
    return total


def sample_tree_pss(root: psutil.Process) -> tuple[int, set[int]]:
    """Sum PSS across ``root`` and all descendants, tolerating mid-walk exits."""
    procs = [root]
    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
        procs.extend(root.children(recursive=True))
    total = 0
    pids: set[int] = set()
    for p in procs:
        try:
            total += p.memory_full_info().pss
            pids.add(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return total, pids


def run_child(args: argparse.Namespace) -> int:
    """Run the active-learning workload and stream phase events to a JSONL file."""
    import logging
    import re

    cycle_re = re.compile(r'Cycle (\d+) complete')
    cycle_start_re = re.compile(r'Cycle (\d+) of \d+')
    train_start_re = re.compile(r'^Training .+ on \d+ compounds')

    with Path(args.events_file).open('a', buffering=1) as events:

        def emit_event(event: str, **extra: object) -> None:
            events.write(json.dumps({'t': time.time(), 'event': event, **extra}) + '\n')

        class _EventHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    msg = record.getMessage()
                except Exception:
                    return
                m = cycle_re.search(msg)
                m_start = cycle_start_re.search(msg)
                if m:
                    emit_event('cycle_complete', cycle=int(m.group(1)))
                elif m_start:
                    emit_event('cycle_start', cycle=int(m_start.group(1)))
                elif train_start_re.match(msg):
                    emit_event('train_start')
                elif msg.startswith('Training complete'):
                    emit_event('train_end')
                elif msg.startswith('Prediction complete'):
                    emit_event('predict_end')

        lm8 = logging.getLogger('learnm8')
        lm8.setLevel(logging.INFO)
        lm8.addHandler(_EventHandler())
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
        )
        lm8.addHandler(stream)

        emit_event('run_start')
        from learnm8 import run_active_learning
        from learnm8.oracles.csv_oracle import CSVOracle

        try:
            run_active_learning(
                compound_pool=str(args.input_csv),
                oracle=CSVOracle(str(args.input_csv), id_column='zincid'),
                learner='rf_fil',
                target_col='dockscore',
                featurizer='morgan',
                smiles_column='smiles',
                id_column='zincid',
                n_cycles=args.n_cycles,
                batch_fraction=0.01,
                strategy='greedy',
                initial_strategy='random',
                score_direction='lower',
                output_dir=str(args.al_output),
                cache_dir=str(args.cache_dir),
                random_state=42,
                large_features_ack=True,
            )
        finally:
            emit_event('run_end')
    return 0


def _write_samples(path: Path, samples: list[tuple[float, int, int, int]]) -> None:
    lines = ['elapsed_s,pss_bytes,vram_total_bytes,vram_proc_bytes']
    for elapsed, pss, vram_total, vram_proc in samples:
        lines.append(f'{elapsed:.3f},{pss},{vram_total},{vram_proc}')
    path.write_text('\n'.join(lines) + '\n')


def _read_events(path: Path) -> list[dict]:
    events: list[dict] = []
    if not path.exists():
        return events
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        with contextlib.suppress(json.JSONDecodeError):
            events.append(json.loads(line))
    return events


def render(
    samples: list[tuple[float, int, int, int]],
    events: list[dict],
    vram_baseline: int,
    plot_png: Path,
    args: argparse.Namespace,
    rc: int,
) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not samples:
        print('[monitor] no samples collected; skipping plot')
        return

    elapsed = [s[0] for s in samples]
    pss = [s[1] / 2**30 for s in samples]
    vram_total = [s[2] / 2**30 for s in samples]
    vram_proc = [s[3] / 2**30 for s in samples]

    if max(vram_proc) > 0.01:
        vram = vram_proc
        vram_label = 'GPU VRAM (per-process, tree)'
    else:
        base = vram_baseline / 2**30
        vram = [max(0.0, v - base) for v in vram_total]
        vram_label = f'GPU VRAM (whole-GPU used - {base:.2f} GiB baseline)'

    fig, ax1 = plt.subplots(figsize=(16, 7))
    ax2 = ax1.twinx()

    starts = sorted(
        (e for e in events if e['event'] == 'cycle_start'),
        key=lambda e: e['cycle'],
    )
    bands: list[tuple[int, float, float]] = []
    if starts:
        bands.append((0, 0.0, starts[0]['t_elapsed']))
        for i, e in enumerate(starts):
            end = starts[i + 1]['t_elapsed'] if i + 1 < len(starts) else elapsed[-1]
            bands.append((e['cycle'], e['t_elapsed'], end))
    for i, (cnum, lo, hi) in enumerate(bands):
        ax1.axvspan(lo, hi, alpha=0.07, color='gray' if i % 2 else 'white')
        ax1.text(
            (lo + hi) / 2, 0.98, f'cyc {cnum}',
            transform=ax1.get_xaxis_transform(), ha='center', va='top',
            fontsize=8, color='dimgray',
        )

    ts_done = pe_done = False
    for e in events:
        if e['event'] == 'train_start':
            ax1.axvline(
                e['t_elapsed'], color='green', ls=':', lw=0.8, alpha=0.7,
                label=None if ts_done else 'train start',
            )
            ts_done = True
        elif e['event'] == 'predict_end':
            ax1.axvline(
                e['t_elapsed'], color='darkorange', ls=':', lw=0.8, alpha=0.7,
                label=None if pe_done else 'predict end',
            )
            pe_done = True

    ax1.plot(elapsed, pss, color='tab:blue', lw=1.3, label='Host RAM (PSS, process tree)')
    l_vram, = ax2.plot(elapsed, vram, color='tab:red', lw=1.3, label=vram_label)

    pk = max(range(len(pss)), key=lambda i: pss[i])
    ax1.scatter([elapsed[pk]], [pss[pk]], color='tab:blue', zorder=5)
    ax1.annotate(
        f'peak {pss[pk]:.2f} GiB', (elapsed[pk], pss[pk]),
        textcoords='offset points', xytext=(6, 6), color='tab:blue', fontsize=9,
    )
    vk = max(range(len(vram)), key=lambda i: vram[i])
    ax2.scatter([elapsed[vk]], [vram[vk]], color='tab:red', zorder=5)
    ax2.annotate(
        f'peak {vram[vk]:.2f} GiB', (elapsed[vk], vram[vk]),
        textcoords='offset points', xytext=(6, -12), color='tab:red', fontsize=9,
    )

    ax1.set_xlabel('Elapsed time (s)')
    ax1.set_ylabel('Host RAM — PSS (GiB)', color='tab:blue')
    ax2.set_ylabel('GPU VRAM (GiB)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.margins(x=0)

    handles_, labels_ = ax1.get_legend_handles_labels()
    handles_.append(l_vram)
    labels_.append(vram_label)
    ax1.legend(handles_, labels_, loc='upper left', fontsize=8)

    ax1.set_title(
        'RAM timeline — LearnM8 active learning, 1M AmpC compounds\n'
        f'rf_fil / morgan / greedy, n_cycles={args.n_cycles}, '
        f'{len(samples)} samples @ {args.interval}s  (child rc={rc})',
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(plot_png, dpi=130)
    plt.close(fig)


def run_monitor(args: argparse.Namespace) -> int:
    """Launch the workload as a child and sample its memory footprint over time."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    events_file = out / 'ram_events.jsonl'
    samples_csv = out / 'ram_samples.csv'
    plot_png = out / 'ram_timeline_1m.png'
    al_output = out / 'al_run'
    cache_dir = out / 'feature_cache'

    for scratch in (al_output, cache_dir):
        if scratch.exists():
            shutil.rmtree(scratch)
    if events_file.exists():
        events_file.unlink()

    nvml, handles = _nvml_init()
    vram_baseline = _gpu_total_used(nvml, handles)
    print(
        f'[monitor] pynvml={"ok" if nvml else "unavailable"}; '
        f'GPU baseline VRAM={vram_baseline / 2**30:.2f} GiB',
        flush=True,
    )

    cmd = [
        sys.executable, str(Path(__file__).resolve()), '--child',
        '--input-csv', str(args.input_csv),
        '--events-file', str(events_file),
        '--al-output', str(al_output),
        '--cache-dir', str(cache_dir),
        '--n-cycles', str(args.n_cycles),
    ]
    env = os.environ.copy()
    if 'CONDA_PREFIX' not in env:
        env['CONDA_PREFIX'] = str(Path(sys.executable).resolve().parents[1])

    print(f'[monitor] launching: {" ".join(cmd)}', flush=True)
    t0 = time.time()
    child = subprocess.Popen(cmd, env=env)
    proc = psutil.Process(child.pid)

    samples: list[tuple[float, int, int, int]] = []
    last_print = t0
    try:
        while child.poll() is None:
            now = time.time()
            pss, pids = sample_tree_pss(proc)
            vram_total = _gpu_total_used(nvml, handles)
            vram_proc = _gpu_proc_used(nvml, handles, pids)
            samples.append((now - t0, pss, vram_total, vram_proc))
            if now - last_print >= 60:
                print(
                    f'[monitor] t={now - t0:.0f}s  pss={pss / 2**30:.2f} GiB  '
                    f'vram_total={vram_total / 2**30:.2f} GiB  procs={len(pids)}',
                    flush=True,
                )
                last_print = now
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('[monitor] interrupted; terminating child...', flush=True)
        child.terminate()
    rc = child.wait()
    print(
        f'[monitor] child exited rc={rc}; {len(samples)} samples over '
        f'{time.time() - t0:.0f}s',
        flush=True,
    )

    events = _read_events(events_file)
    for e in events:
        e['t_elapsed'] = e.get('t', t0) - t0

    _write_samples(samples_csv, samples)
    render(samples, events, vram_baseline, plot_png, args, rc)
    if samples:
        peak_pss = max(s[1] for s in samples) / 2**30
        peak_vram = max(s[2] for s in samples) / 2**30
        print(
            f'[monitor] peak host PSS={peak_pss:.2f} GiB  '
            f'peak GPU used={peak_vram:.2f} GiB',
            flush=True,
        )
    print(f'[monitor] plot:    {plot_png}', flush=True)
    print(f'[monitor] samples: {samples_csv}', flush=True)
    print(f'[monitor] events:  {events_file}', flush=True)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or '').strip().splitlines()[0])
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--input-csv', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--events-file', type=Path)
    parser.add_argument('--al-output', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--n-cycles', type=int, default=10)
    parser.add_argument('--interval', type=float, default=0.5)
    args = parser.parse_args()
    if args.child:
        return run_child(args)
    return run_monitor(args)


if __name__ == '__main__':
    raise SystemExit(main())

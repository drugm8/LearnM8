"""Trace host RAM, CPU/GPU utilization and GPU VRAM over a 1M-compound run.

Monitor mode (default): launches the active-learning workload as a child
subprocess, samples its whole process tree every ``--interval`` seconds, then
writes a timeline plot and a raw-sample CSV. Running the workload as a child
keeps the sampler's own footprint (psutil, matplotlib) out of the measurement
and captures joblib/multiprocessing forks by walking the process tree.

Child mode (``--child``): runs ``run_active_learning()`` with the
rf_fil / morgan / greedy config used by the prior 1M AmpC run. It installs
lightweight monkeypatch wrappers around every active-learning sub-step
(validation, per-cycle train / featurize / predict / persist / prune / select /
evaluate, and end-of-run save) so each phase emits a timestamped span event.

The timeline plot overlays those phase spans as a two-lane Gantt ribbon beneath
the RAM/VRAM and CPU/GPU-utilization curves, so the cycle-1 memory peak and the
later-cycle spikes can be attributed to a specific phase. Memory metric is PSS
(Proportional Set Size) summed across the tree, so shared interpreter/library
pages are counted once. CPU% is the process-tree total normalised by logical
core count; GPU% is whole-device compute utilization.

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

# ---------------------------------------------------------------------------
# Phase taxonomy — colours + ribbon-lane assignment for the timeline plot.
# ---------------------------------------------------------------------------
# Every span emitted by the child instrumentation has one of these names.
# 'validate'/'save' are run-level; the rest are per-cycle sub-steps. 'featurize'
# is nested inside 'train' and 'predict' and is drawn on its own ribbon lane.
PHASE_COLORS: dict[str, str] = {
    'validate': '#4878a8',
    'train': '#2ca02c',
    'featurize': '#f08000',
    'predict': '#d62728',
    'persist': '#e377c2',
    'prune': '#9467bd',
    'select': '#17becf',
    'evaluate': '#8c564b',
    'save': '#7f7f7f',
}
CYCLE_PHASES: frozenset[str] = frozenset(
    {'train', 'featurize', 'predict', 'persist', 'prune', 'select', 'evaluate'}
)
RUN_PHASES: frozenset[str] = frozenset({'validate', 'save'})
# Ribbon lane per phase; anything not listed defaults to lane 0.
RIBBON_LANE: dict[str, int] = {'featurize': 1}


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


def _gpu_util(nvml: Any, handles: list) -> float:
    """Return the busiest GPU's compute-utilization percent (0-100)."""
    if nvml is None:
        return 0.0
    best = 0.0
    for h in handles:
        with contextlib.suppress(Exception):
            best = max(best, float(nvml.nvmlDeviceGetUtilizationRates(h).gpu))
    return best


def sample_tree(
    root: psutil.Process, cache: dict[int, psutil.Process]
) -> tuple[int, float, set[int]]:
    """Sum PSS and CPU% across ``root`` and all descendants.

    ``cache`` holds one persistent ``Process`` per PID so ``cpu_percent()``
    measures the delta since the previous sample — newly forked workers read
    0% on their first appearance, then accurately. PSS counts shared pages
    once; CPU% is the process-tree total normalised by logical core count
    (0-100%). Dead PIDs are evicted from ``cache`` each call.
    """
    procs = [root]
    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
        procs.extend(root.children(recursive=True))
    pss_total = 0
    cpu_total = 0.0
    pids: set[int] = set()
    for p in procs:
        proc = cache.setdefault(p.pid, p)
        try:
            pss_total += proc.memory_full_info().pss
            cpu_total += proc.cpu_percent()
            pids.add(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    for dead in [pid for pid in cache if pid not in pids]:
        del cache[dead]
    return pss_total, cpu_total / (psutil.cpu_count() or 1), pids


# ---------------------------------------------------------------------------
# Child mode — run the workload with granular phase instrumentation.
# ---------------------------------------------------------------------------


def run_child(args: argparse.Namespace) -> int:
    """Run the active-learning workload, emitting a span event per sub-phase.

    Instrumentation is installed by monkeypatching the orchestration-level
    functions (all single-threaded, main-process calls) — no production code is
    modified. Each wrapper writes a ``span_begin``/``span_end`` JSONL record so
    the monitor can attribute memory to a specific phase.
    """
    import functools
    import importlib
    import logging

    # Mutable cell holding the cycle currently executing; sub-phase wrappers
    # read it so their spans are tagged with the right cycle number.
    state: dict[str, Any] = {'cycle': None}

    with Path(args.events_file).open('a', buffering=1) as events:

        def emit(record: dict) -> None:
            events.write(json.dumps(record) + '\n')

        def emit_event(event: str, **extra: object) -> None:
            emit({'t': time.time(), 'event': event, **extra})

        def emit_span(name: str, edge: str, **extra: object) -> None:
            emit(
                {
                    't': time.time(),
                    'event': 'span',
                    'name': name,
                    'edge': edge,
                    'cycle': state['cycle'],
                    **extra,
                }
            )

        installed: list[str] = []

        def _wrap_span(mod_name: str, attr: str, span_name: str) -> None:
            """Wrap ``mod_name.attr`` to bracket it with a begin/end span."""
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                return
            orig = getattr(mod, attr, None)
            if orig is None or getattr(orig, '_ramtl_wrapped', False):
                return

            @functools.wraps(orig)
            def wrapper(*a: object, **kw: object) -> object:
                emit_span(span_name, 'begin')
                try:
                    return orig(*a, **kw)
                finally:
                    emit_span(span_name, 'end')

            wrapper._ramtl_wrapped = True  # type: ignore[attr-defined]
            setattr(mod, attr, wrapper)
            installed.append(f'{span_name} <- {mod_name}.{attr}')

        def _wrap_cycle(
            mod_name: str, attr: str, fixed_cycle: int | None = None
        ) -> None:
            """Wrap a cycle entry-point: set ``state['cycle']`` + emit a cycle span."""
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                return
            orig = getattr(mod, attr, None)
            if orig is None or getattr(orig, '_ramtl_wrapped', False):
                return

            @functools.wraps(orig)
            def wrapper(*a: object, **kw: object) -> object:
                if fixed_cycle is not None:
                    cyc: object = fixed_cycle
                elif 'cycle' in kw:
                    cyc = kw['cycle']
                else:
                    cyc = a[1] if len(a) > 1 else None
                state['cycle'] = cyc
                emit_span('cycle', 'begin', cycle=cyc)
                try:
                    return orig(*a, **kw)
                finally:
                    emit_span('cycle', 'end', cycle=cyc)

            wrapper._ramtl_wrapped = True  # type: ignore[attr-defined]
            setattr(mod, attr, wrapper)
            installed.append(f'cycle <- {mod_name}.{attr}')

        def _wrap_featurize() -> None:
            """Wrap ``extract_features`` at every cycle-relevant call site."""
            try:
                ef_mod = importlib.import_module('learnm8.features.extraction')
            except Exception:
                return
            orig = getattr(ef_mod, 'extract_features', None)
            if orig is None or getattr(orig, '_ramtl_wrapped', False):
                return

            @functools.wraps(orig)
            def wrapper(*a: object, **kw: object) -> object:
                arg0 = a[0] if a else kw.get('smiles_list', [])
                n = len(arg0) if isinstance(arg0, (list, tuple)) else -1
                emit_span('featurize', 'begin', n=n)
                try:
                    return orig(*a, **kw)
                finally:
                    emit_span('featurize', 'end', n=n)

            wrapper._ramtl_wrapped = True  # type: ignore[attr-defined]
            # Patch the definition module AND every importer that bound the
            # name at import time (Python rebinds by value, not by reference).
            ef_attr = 'extract_features'
            for m in (
                'learnm8.features.extraction',
                'learnm8.core.cycle',
                'learnm8.core.batching',
                'learnm8.acquisition.simulated_annealing',
            ):
                try:
                    mod = importlib.import_module(m)
                except Exception:
                    continue
                if getattr(mod, ef_attr, None) is not None:
                    setattr(mod, ef_attr, wrapper)
            installed.append('featurize <- extract_features (all call sites)')

        # Structured stdout logging so the run is still followable live.
        lm8 = logging.getLogger('learnm8')
        lm8.setLevel(logging.INFO)
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
        )
        lm8.addHandler(stream)

        emit_event('run_start')
        from learnm8 import run_active_learning
        from learnm8.oracles.csv_oracle import CSVOracle

        # learnm8 is now imported — install instrumentation on the live modules.
        _wrap_cycle('learnm8.api', 'execute_cycle')
        _wrap_cycle('learnm8.api', 'select_initial_batch', fixed_cycle=0)
        _wrap_span('learnm8.api', 'validate_compound_pool', 'validate')
        _wrap_span('learnm8.api', 'save_results', 'save')
        _wrap_span('learnm8.core.cycle', '_train_learner', 'train')
        _wrap_span('learnm8.core.cycle', '_predict_pool', 'predict')
        # write_cycle_predictions is a lazy import inside execute_cycle — patch
        # the definition module so the in-function `from ... import` binds it.
        _wrap_span('learnm8.core.persistence', 'write_cycle_predictions', 'persist')
        _wrap_span('learnm8.core.cycle', '_apply_pruning', 'prune')
        _wrap_span('learnm8.core.cycle', '_select_and_measure', 'select')
        _wrap_span('learnm8.core.cycle', '_calculate_cycle_metrics', 'evaluate')
        _wrap_featurize()
        print(
            f'[child] phase instrumentation installed ({len(installed)} hooks):',
            flush=True,
        )
        for hook in installed:
            print(f'[child]   {hook}', flush=True)

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
                memory_safety_factor=args.memory_safety_factor,
            )
        finally:
            emit_event('run_end')
    return 0


def _write_samples(
    path: Path, samples: list[tuple[float, int, int, int, float, float]]
) -> None:
    lines = [
        'elapsed_s,pss_bytes,vram_total_bytes,vram_proc_bytes,'
        'cpu_pct,gpu_util_pct'
    ]
    for elapsed, pss, vram_total, vram_proc, cpu, gpu in samples:
        lines.append(
            f'{elapsed:.3f},{pss},{vram_total},{vram_proc},{cpu:.2f},{gpu:.2f}'
        )
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


# ---------------------------------------------------------------------------
# Phase-span analysis
# ---------------------------------------------------------------------------


def _phase_intervals(events: list[dict]) -> list[dict]:
    """Pair ``span`` begin/end events into nested intervals.

    Returns a list of ``{name, cycle, t0, t1, depth, n}`` dicts. ``depth`` is the
    nesting level at the time the span opened (cycle = 0, cycle sub-phase = 1,
    featurize = 2), recovered with a LIFO stack since spans nest cleanly.
    """
    spans = sorted(
        (e for e in events if e.get('event') == 'span' and 't_elapsed' in e),
        key=lambda e: e['t'],
    )
    stack: list[tuple[dict, int]] = []
    intervals: list[dict] = []
    for e in spans:
        if e.get('edge') == 'begin':
            stack.append((e, len(stack)))
            continue
        # 'end' — match the most recent open span of the same name.
        for i in range(len(stack) - 1, -1, -1):
            beg, depth = stack[i]
            if beg.get('name') == e.get('name'):
                intervals.append(
                    {
                        'name': e.get('name'),
                        'cycle': beg.get('cycle'),
                        't0': beg['t_elapsed'],
                        't1': e['t_elapsed'],
                        'depth': depth,
                        'n': beg.get('n'),
                    }
                )
                del stack[i]
                break
    return intervals


def _summarize_phases(
    intervals: list[dict],
    samples: list[tuple[float, int, int, int, float, float]],
) -> str:
    """Per-phase summary: calls, total seconds, peak PSS, mean CPU%/GPU%."""
    elapsed = [s[0] for s in samples]
    pss = [s[1] / 2**30 for s in samples]
    cpu = [s[4] for s in samples]
    gpu = [s[5] for s in samples]
    agg: dict[str, dict[str, float]] = {}
    for iv in intervals:
        name = iv['name']
        if name not in PHASE_COLORS:
            continue
        a = agg.setdefault(
            name,
            {'dur': 0.0, 'peak': 0.0, 'count': 0.0,
             'cpu': 0.0, 'gpu': 0.0, 'samples': 0.0},
        )
        a['dur'] += iv['t1'] - iv['t0']
        a['count'] += 1
        inside = [i for i, t in enumerate(elapsed) if iv['t0'] <= t <= iv['t1']]
        if not inside and elapsed:
            # Phase shorter than the sample interval — use the nearest sample
            # so the summary never reports a spurious 0.00.
            mid = (iv['t0'] + iv['t1']) / 2.0
            inside = [min(range(len(elapsed)), key=lambda i: abs(elapsed[i] - mid))]
        if inside:
            a['peak'] = max(a['peak'], max(pss[i] for i in inside))
            a['cpu'] += sum(cpu[i] for i in inside)
            a['gpu'] += sum(gpu[i] for i in inside)
            a['samples'] += len(inside)

    lines = [
        f'{"phase":<10}{"calls":>6}{"total_s":>10}{"peak_GiB":>10}'
        f'{"cpu%":>8}{"gpu%":>8}'
    ]
    lines.append('-' * 52)
    for name in PHASE_COLORS:
        if name in agg:
            a = agg[name]
            n = a['samples'] or 1.0
            lines.append(
                f'{name:<10}{int(a["count"]):>6}{a["dur"]:>10.1f}'
                f'{a["peak"]:>10.2f}{a["cpu"] / n:>8.1f}{a["gpu"] / n:>8.1f}'
            )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render(
    samples: list[tuple[float, int, int, int, float, float]],
    events: list[dict],
    vram_baseline: int,
    plot_png: Path,
    args: argparse.Namespace,
    rc: int,
) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not samples:
        print('[monitor] no samples collected; skipping plot')
        return

    elapsed = [s[0] for s in samples]
    pss = [s[1] / 2**30 for s in samples]
    vram_total = [s[2] / 2**30 for s in samples]
    vram_proc = [s[3] / 2**30 for s in samples]
    cpu = [s[4] for s in samples]
    gpu = [s[5] for s in samples]

    if max(vram_proc) > 0.01:
        vram = vram_proc
        vram_label = 'GPU VRAM (per-process, tree)'
    else:
        base = vram_baseline / 2**30
        vram = [max(0.0, v - base) for v in vram_total]
        vram_label = f'GPU VRAM (whole-GPU used - {base:.2f} GiB baseline)'

    intervals = _phase_intervals(events)
    cycle_ivs = [iv for iv in intervals if iv['name'] == 'cycle']
    cycle_phase_ivs = [iv for iv in intervals if iv['name'] in CYCLE_PHASES]
    run_ivs = [iv for iv in intervals if iv['name'] in RUN_PHASES]

    fig = plt.figure(figsize=(32, 18))
    gs = GridSpec(3, 1, height_ratios=[3.0, 1.2, 1.0], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axu = fig.add_subplot(gs[1], sharex=ax)
    axr = fig.add_subplot(gs[2], sharex=ax)
    ax2 = ax.twinx()

    span_end = elapsed[-1] if elapsed else 1.0

    # --- background bands: cycles (alternating grey) + validate/save tints ---
    # Drawn on both the memory and utilization panels so the cycle structure
    # lines up vertically; the cycle/phase text labels go on ``ax`` only.
    for iv in sorted(cycle_ivs, key=lambda c: c['t0']):
        cyc = iv['cycle'] if iv['cycle'] is not None else 0
        band = 'gray' if cyc % 2 else 'white'
        for a in (ax, axu):
            a.axvspan(iv['t0'], iv['t1'], alpha=0.07, color=band)
        ax.text(
            (iv['t0'] + iv['t1']) / 2, 0.99, f'cyc {iv["cycle"]}',
            transform=ax.get_xaxis_transform(), ha='center', va='top',
            fontsize=10, color='dimgray', fontweight='bold',
        )
    for iv in run_ivs:
        for a in (ax, axu):
            a.axvspan(iv['t0'], iv['t1'], alpha=0.12, color=PHASE_COLORS[iv['name']])
        ax.text(
            (iv['t0'] + iv['t1']) / 2, 0.99, iv['name'],
            transform=ax.get_xaxis_transform(), ha='center', va='top',
            fontsize=10, color=PHASE_COLORS[iv['name']], fontweight='bold',
        )

    # --- main RAM / VRAM curves ---
    ax.plot(
        elapsed, pss, color='tab:blue', lw=1.7,
        label='Host RAM (PSS, process tree)',
    )
    ax2.plot(elapsed, vram, color='tab:red', lw=1.4, alpha=0.85, label=vram_label)

    # --- peak annotation, named by the deepest phase covering it ---
    pk = max(range(len(pss)), key=lambda i: pss[i])
    pk_t = elapsed[pk]
    covering = [
        iv
        for iv in (cycle_phase_ivs + run_ivs)
        if iv['t0'] <= pk_t <= iv['t1']
    ]
    covering.sort(key=lambda iv: iv['depth'])
    phase_label = ''
    if covering:
        deepest = covering[-1]
        cyc = deepest['cycle']
        phase_label = f' — phase: {deepest["name"]}'
        if cyc is not None:
            phase_label += f' (cycle {cyc})'
    ax.scatter([pk_t], [pss[pk]], color='tab:blue', zorder=6, s=55)
    ax.annotate(
        f'peak {pss[pk]:.2f} GiB{phase_label}',
        (pk_t, pss[pk]),
        textcoords='offset points', xytext=(10, 10),
        color='tab:blue', fontsize=13, fontweight='bold',
    )
    vk = max(range(len(vram)), key=lambda i: vram[i])
    ax2.scatter([elapsed[vk]], [vram[vk]], color='tab:red', zorder=6, s=45)
    ax2.annotate(
        f'peak {vram[vk]:.2f} GiB', (elapsed[vk], vram[vk]),
        textcoords='offset points', xytext=(10, -16),
        color='tab:red', fontsize=11,
    )

    # --- CPU / GPU utilization panel (both normalised 0-100%) ---
    axu.plot(elapsed, cpu, color='tab:green', lw=1.5, label='CPU (process tree)')
    axu.plot(elapsed, gpu, color='tab:orange', lw=1.5, label='GPU (compute)')
    for series, color in ((cpu, 'tab:green'), (gpu, 'tab:orange')):
        upk = max(range(len(series)), key=lambda i: series[i])
        axu.scatter([elapsed[upk]], [series[upk]], color=color, zorder=6, s=40)
        axu.annotate(
            f'peak {series[upk]:.0f}%', (elapsed[upk], series[upk]),
            textcoords='offset points', xytext=(8, 6),
            color=color, fontsize=10, fontweight='bold',
        )
    axu.set_ylabel('Utilization (%)', fontsize=13)
    axu.set_ylim(0, 105)
    axu.tick_params(axis='y', labelsize=11)
    axu.margins(x=0)
    axu.grid(axis='y', alpha=0.2)
    axu.legend(loc='upper left', fontsize=10, ncol=2, framealpha=0.92)
    plt.setp(axu.get_xticklabels(), visible=False)

    # --- phase ribbon (two-lane Gantt) ---
    lane_h = 0.78
    label_min = span_end * 0.012
    for iv in cycle_phase_ivs + run_ivs:
        lane = RIBBON_LANE.get(iv['name'], 0)
        width = iv['t1'] - iv['t0']
        axr.barh(
            lane, max(width, span_end * 0.0006), left=iv['t0'], height=lane_h,
            color=PHASE_COLORS[iv['name']], edgecolor='white', lw=0.5,
        )
        if width > label_min:
            txt = iv['name']
            if iv['name'] == 'featurize' and iv.get('n'):
                txt = f'feat {iv["n"] / 1000:.0f}k'
            axr.text(
                iv['t0'] + width / 2, lane, txt, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold',
            )
    axr.set_ylim(-0.65, 1.65)
    axr.set_yticks([0, 1])
    axr.set_yticklabels(['cycle phase', 'featurization'], fontsize=11)
    axr.set_xlabel('Elapsed time (s)', fontsize=13)
    axr.grid(axis='x', alpha=0.3)
    axr.margins(x=0)

    # --- axes cosmetics ---
    ax.set_ylabel('Host RAM — PSS (GiB)', color='tab:blue', fontsize=13)
    ax2.set_ylabel('GPU VRAM (GiB)', color='tab:red', fontsize=13)
    ax.tick_params(axis='y', labelcolor='tab:blue', labelsize=11)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=11)
    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax.margins(x=0)
    ax.grid(axis='y', alpha=0.2)
    plt.setp(ax.get_xticklabels(), visible=False)

    # --- legend: curves + phase colour key ---
    handles: list[Any] = [
        Line2D([], [], color='tab:blue', lw=2, label='Host RAM (PSS, tree)'),
        Line2D([], [], color='tab:red', lw=2, label=vram_label),
    ]
    handles += [
        Patch(facecolor=c, label=p) for p, c in PHASE_COLORS.items()
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=10, ncol=2, framealpha=0.92)

    # --- per-phase summary table (on-figure + stdout) ---
    summary = _summarize_phases(intervals, samples)
    print('[monitor] per-phase summary:', flush=True)
    for line in summary.splitlines():
        print(f'[monitor]   {line}', flush=True)
    fig.text(
        0.995, 0.988, summary, ha='right', va='top', family='monospace',
        fontsize=10,
        bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.92),
    )

    ax.set_title(
        'RAM & utilization timeline — LearnM8 active learning, 1M AmpC compounds   '
        f'(rf_fil / morgan / greedy, n_cycles={args.n_cycles}, '
        f'{len(samples)} samples @ {args.interval}s, child rc={rc})',
        fontsize=15, fontweight='bold',
    )
    fig.savefig(plot_png, dpi=150, bbox_inches='tight')
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
        '--memory-safety-factor', str(args.memory_safety_factor),
    ]
    env = os.environ.copy()
    if 'CONDA_PREFIX' not in env:
        env['CONDA_PREFIX'] = str(Path(sys.executable).resolve().parents[1])

    print(f'[monitor] launching: {" ".join(cmd)}', flush=True)
    t0 = time.time()
    child = subprocess.Popen(cmd, env=env)
    proc = psutil.Process(child.pid)

    samples: list[tuple[float, int, int, int, float, float]] = []
    proc_cache: dict[int, psutil.Process] = {}
    last_print = t0
    try:
        while child.poll() is None:
            now = time.time()
            pss, cpu_pct, pids = sample_tree(proc, proc_cache)
            vram_total = _gpu_total_used(nvml, handles)
            vram_proc = _gpu_proc_used(nvml, handles, pids)
            gpu_util = _gpu_util(nvml, handles)
            samples.append(
                (now - t0, pss, vram_total, vram_proc, cpu_pct, gpu_util)
            )
            if now - last_print >= 60:
                print(
                    f'[monitor] t={now - t0:.0f}s  pss={pss / 2**30:.2f} GiB  '
                    f'vram_total={vram_total / 2**30:.2f} GiB  '
                    f'cpu={cpu_pct:.0f}%  gpu={gpu_util:.0f}%  procs={len(pids)}',
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
        peak_cpu = max(s[4] for s in samples)
        peak_gpu = max(s[5] for s in samples)
        print(
            f'[monitor] peak host PSS={peak_pss:.2f} GiB  '
            f'peak GPU used={peak_vram:.2f} GiB  '
            f'peak CPU={peak_cpu:.0f}%  peak GPU util={peak_gpu:.0f}%',
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
    parser.add_argument('--memory-safety-factor', type=float, default=0.7)
    args = parser.parse_args()
    if args.child:
        return run_child(args)
    return run_monitor(args)


if __name__ == '__main__':
    raise SystemExit(main())

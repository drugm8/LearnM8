"""Memory / OOM diagnostics for LearnM8 at scale (branch ``diag/oom-memory``).

WHY THIS EXISTS
---------------
LearnM8 sizes its prediction chunk to a fraction of *node-reported* available
RAM (``core/batching.py:get_available_memory`` -> ``psutil.virtual_memory()``)
and resolves ``n_jobs=-1`` through joblib -> ``os.cpu_count()``. Both reads see
the **whole machine**, not the confinement the job actually runs under:

* **HTCondor** confines each job to a *slot* with ``RequestMemory`` /
  ``RequestCpus``. Exceeding ``RequestMemory`` gets the job OOM-killed (cgroup
  enforcement) or held/evicted (procd polling).
* **cgroups** (v2 ``memory.max`` / v1 ``memory.limit_in_bytes``) cap RSS below
  the node total on any managed environment.

Consequence: on a 700 GB machine whose Condor slot grants, say, 128 GB, the
sizer budgets ``0.7 * 700 = 490 GB`` for one feature matrix and the job dies —
while the *same* run fits on a smaller unconfined workstation. "Bigger machine
OOMs, smaller one doesn't" is the signature of this confinement-blind sizing.

WHAT THIS DOES (observe only — no behaviour change)
---------------------------------------------------
On import (auto-activated from ``learnm8/__init__.py`` on this branch) it:

1. Writes a one-shot **environment snapshot**: node RAM, cgroup memory limit +
   current usage (v2/v1, hierarchy-walked), Condor slot ``RequestMemory`` /
   ``RequestCpus`` (from ``_CONDOR_*`` env + the job ClassAd), ``os.cpu_count``
   vs CPU affinity vs Condor CPUs — and DANGER flags when node >> confinement.
2. Wraps ``estimate_batch_size`` to log every sizing decision and the predicted
   working-set bytes, with a loud WARNING when the prediction would exceed the
   confinement headroom (the OOM trigger).
3. Runs a low-overhead **peak sampler** (RSS + cgroup ``memory.current``) tagged
   with the current phase, so the JSONL pinpoints *which* phase OOMs.
4. Wraps validation / cycle entry points to label phases.

Output: JSON Lines to ``$LEARNM8_MEMDIAG_OUT`` if set, else
``<output_dir>/memdiag.jsonl`` once the run's ``output_dir`` is known, else
``./memdiag.jsonl``. Every record is also emitted as a WARNING/INFO log.

Everything is wrapped in try/except: diagnostics must never crash a run.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger('learnm8.memdiag')

GB = 1024**3

_LOCK = threading.Lock()
_ACTIVATED = False
_SINK_PATH: Path | None = None
_PHASE = 'startup'
_SAMPLER: _PeakSampler | None = None
_START = 0.0


# ---------------------------------------------------------------------------
# Low-level readers (never raise)
# ---------------------------------------------------------------------------


def _now() -> float:
    try:
        return time.time() - _START
    except Exception:
        return 0.0


def _read_int_file(path: str) -> int | None:
    try:
        with open(path) as f:
            txt = f.read().strip()
        if txt in ('max', ''):
            return None  # cgroup v2 'max' == unlimited
        return int(txt)
    except (OSError, ValueError):
        return None


def _self_cgroup_paths() -> list[str]:
    """Relative cgroup paths from /proc/self/cgroup (v2 '0::', v1 controllers)."""
    out: list[str] = []
    try:
        with open('/proc/self/cgroup') as f:
            for line in f:
                parts = line.strip().split(':', 2)
                if len(parts) == 3:
                    out.append(parts[2])
    except OSError:
        pass
    return out


def _walk_cgroup_memory() -> dict[str, Any]:
    """Effective cgroup memory limit + current usage (v2 first, v1 fallback).

    Walks the cgroup hierarchy and takes the *minimum* finite limit (the
    effective cap), since a child can be looser than an ancestor only down to
    the tightest ancestor.
    """
    info: dict[str, Any] = {'version': None, 'limit_bytes': None, 'current_bytes': None}
    rel_paths = _self_cgroup_paths()

    # cgroup v2: single unified hierarchy under /sys/fs/cgroup, '0::<path>'.
    if os.path.exists('/sys/fs/cgroup/cgroup.controllers'):
        info['version'] = 2
        rel = rel_paths[0] if rel_paths else '/'
        rel = rel.lstrip('/')
        node = Path('/sys/fs/cgroup') / rel
        limit = None
        # Walk from the leaf up to the root, taking the tightest memory.max.
        cur = node
        root = Path('/sys/fs/cgroup')
        seen_current = None
        while True:
            lim = _read_int_file(str(cur / 'memory.max'))
            if lim is not None:
                limit = lim if limit is None else min(limit, lim)
            if seen_current is None:
                seen_current = _read_int_file(str(cur / 'memory.current'))
            if cur == root:
                break
            cur = cur.parent
        info['limit_bytes'] = limit
        info['current_bytes'] = seen_current
        return info

    # cgroup v1: per-controller hierarchy under /sys/fs/cgroup/memory/<path>.
    mem_rel = None
    try:
        with open('/proc/self/cgroup') as f:
            for line in f:
                parts = line.strip().split(':', 2)
                if len(parts) == 3 and 'memory' in parts[1].split(','):
                    mem_rel = parts[2]
                    break
    except OSError:
        pass
    if mem_rel is not None and os.path.isdir('/sys/fs/cgroup/memory'):
        info['version'] = 1
        node = Path('/sys/fs/cgroup/memory') / mem_rel.lstrip('/')
        limit = _read_int_file(str(node / 'memory.limit_in_bytes'))
        # v1 reports a huge sentinel (~ PAGE_COUNTER_MAX) for "unlimited".
        if limit is not None and limit >= (1 << 62):
            limit = None
        info['limit_bytes'] = limit
        info['current_bytes'] = _read_int_file(str(node / 'memory.usage_in_bytes'))
    return info


def _walk_cgroup_cpu() -> float | None:
    """Effective CPU count from the cgroup CPU quota (v2 cpu.max / v1 cfs)."""
    try:
        if os.path.exists('/sys/fs/cgroup/cgroup.controllers'):
            rel = (_self_cgroup_paths() or ['/'])[0].lstrip('/')
            cur = Path('/sys/fs/cgroup') / rel
            root = Path('/sys/fs/cgroup')
            best = None
            while True:
                try:
                    with open(cur / 'cpu.max') as f:
                        quota, period = f.read().split()
                    if quota != 'max':
                        cpus = int(quota) / int(period)
                        best = cpus if best is None else min(best, cpus)
                except (OSError, ValueError):
                    pass
                if cur == root:
                    break
                cur = cur.parent
            return best
        # v1
        q = _read_int_file('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
        p = _read_int_file('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
        if q and p and q > 0:
            return q / p
    except Exception:
        pass
    return None


def _read_condor() -> dict[str, Any]:
    """HTCondor slot allocation from env + the job ClassAd file.

    Condor exports ``_CONDOR_*`` env vars and writes ``.job.ad`` / ``.machine.ad``
    into the scratch dir (path in ``_CONDOR_JOB_AD`` / ``_CONDOR_MACHINE_AD``).
    ``RequestMemory`` is in MB; ``RequestCpus`` is a count.
    """
    info: dict[str, Any] = {'present': False}
    env_keys = [
        '_CONDOR_SLOT',
        '_CONDOR_SCRATCH_DIR',
        '_CONDOR_JOB_AD',
        '_CONDOR_MACHINE_AD',
        '_CONDOR_AssignedCpus',
        '_CONDOR_JOB_PIDS',
        'OMP_NUM_THREADS',
        'CUDA_VISIBLE_DEVICES',
    ]
    env = {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None}
    if env:
        info['present'] = True
        info['env'] = env

    def _parse_ad(path: str | None) -> dict[str, str]:
        ad: dict[str, str] = {}
        if not path or not os.path.exists(path):
            return ad
        try:
            with open(path) as f:
                for line in f:
                    if '=' in line:
                        k, _, v = line.partition('=')
                        ad[k.strip()] = v.strip().strip('"')
        except OSError:
            pass
        return ad

    job_ad = _parse_ad(os.environ.get('_CONDOR_JOB_AD'))
    machine_ad = _parse_ad(os.environ.get('_CONDOR_MACHINE_AD'))
    if job_ad or machine_ad:
        info['present'] = True

    def _num(ad: dict[str, str], key: str) -> float | None:
        try:
            return float(ad[key])
        except (KeyError, ValueError):
            return None

    req_mem_mb = _num(job_ad, 'RequestMemory') or _num(machine_ad, 'Memory')
    req_cpus = _num(job_ad, 'RequestCpus') or _num(machine_ad, 'Cpus')
    if req_mem_mb is not None:
        info['request_memory_bytes'] = int(req_mem_mb * 1024 * 1024)
    if req_cpus is not None:
        info['request_cpus'] = req_cpus
    return info


# ---------------------------------------------------------------------------
# JSONL sink
# ---------------------------------------------------------------------------


def _resolve_sink() -> Path:
    global _SINK_PATH
    if _SINK_PATH is not None:
        return _SINK_PATH
    env = os.environ.get('LEARNM8_MEMDIAG_OUT')
    _SINK_PATH = Path(env) if env else Path.cwd() / 'memdiag.jsonl'
    return _SINK_PATH


def _set_sink_dir(output_dir: Any) -> None:
    """Redirect the sink to ``<output_dir>/memdiag.jsonl`` (sidecar)."""
    global _SINK_PATH
    if os.environ.get('LEARNM8_MEMDIAG_OUT'):
        return  # explicit override wins
    try:
        if output_dir is None:
            return
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        _SINK_PATH = p / 'memdiag.jsonl'
    except Exception:
        pass


def _emit(record: dict[str, Any], *, warn: bool = False) -> None:
    record.setdefault('t', round(_now(), 2))
    record.setdefault('phase', _PHASE)
    line = json.dumps(record, default=str)
    try:
        with _LOCK, open(_resolve_sink(), 'a') as f:
            f.write(line + '\n')
    except Exception:
        pass
    msg = '[memdiag] ' + line
    if warn:
        logger.warning(msg)
    else:
        logger.info(msg)


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------


def _gb(x: Any) -> Any:
    try:
        return round(x / GB, 2)
    except Exception:
        return None


def snapshot_environment() -> dict[str, Any]:
    """One-shot confinement snapshot with DANGER flags. Returns the record."""
    rec: dict[str, Any] = {'event': 'env_snapshot'}
    # Node memory.
    node_total = node_avail = None
    try:
        import psutil

        vm = psutil.virtual_memory()
        node_total, node_avail = vm.total, vm.available
    except Exception:
        pass
    rec['node_mem_total_gb'] = _gb(node_total)
    rec['node_mem_available_gb'] = _gb(node_avail)

    # cgroup.
    cg = _walk_cgroup_memory()
    rec['cgroup_version'] = cg['version']
    rec['cgroup_limit_gb'] = _gb(cg['limit_bytes'])
    rec['cgroup_current_gb'] = _gb(cg['current_bytes'])
    cg_cpu = _walk_cgroup_cpu()
    rec['cgroup_cpu_quota'] = round(cg_cpu, 2) if cg_cpu else None

    # CPUs.
    rec['os_cpu_count'] = os.cpu_count()
    try:
        rec['cpu_affinity'] = len(os.sched_getaffinity(0))
    except Exception:
        rec['cpu_affinity'] = None
    try:
        from joblib import effective_n_jobs

        rec['joblib_effective_n_jobs_-1'] = effective_n_jobs(-1)
    except Exception:
        rec['joblib_effective_n_jobs_-1'] = None

    # Condor slot.
    condor = _read_condor()
    rec['condor'] = condor

    # The effective memory ceiling the run actually has, and what the sizer sees.
    ceilings = [
        b
        for b in (
            cg['limit_bytes'],
            condor.get('request_memory_bytes'),
        )
        if b is not None
    ]
    effective_ceiling = min(ceilings) if ceilings else None
    rec['effective_mem_ceiling_gb'] = _gb(effective_ceiling)
    rec['sizer_sees_available_gb'] = _gb(node_avail)

    # DANGER: the sizer's view exceeds the real ceiling -> over-allocation risk.
    danger = []
    if effective_ceiling and node_avail and node_avail > 1.5 * effective_ceiling:
        ratio = round(node_avail / effective_ceiling, 1)
        danger.append(
            f'sizer sees {_gb(node_avail)}GB but real ceiling is '
            f'{_gb(effective_ceiling)}GB ({ratio}x over) -> predict chunk will be '
            f'budgeted against node RAM and exceed the limit'
        )
    n_cpu = rec['os_cpu_count'] or 0
    aff = rec['cpu_affinity'] or n_cpu
    req_cpu = condor.get('request_cpus') or rec['cgroup_cpu_quota']
    if req_cpu and n_cpu and n_cpu > 1.5 * req_cpu:
        danger.append(
            f'os.cpu_count()={n_cpu} but allocation is ~{req_cpu} CPUs -> '
            f'n_jobs=-1 over-subscribes workers (~{n_cpu} x per-worker RSS)'
        )
    if aff and n_cpu and aff < n_cpu:
        danger.append(f'cpu affinity {aff} < cpu_count {n_cpu}')
    rec['danger'] = danger
    _emit(rec, warn=bool(danger))
    return rec


# ---------------------------------------------------------------------------
# Phase tracking + peak sampler
# ---------------------------------------------------------------------------


def set_phase(name: str) -> None:
    global _PHASE
    prev = _PHASE
    _PHASE = name
    if _SAMPLER is not None:
        peak = _SAMPLER.take_phase_peak()
        _emit(
            {
                'event': 'phase_change',
                'from': prev,
                'to': name,
                'prev_phase_peak_rss_gb': _gb(peak),
            }
        )


def _proc_rss() -> int | None:
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class _PeakSampler:
    """Background RSS + cgroup.current sampler, peak tracked per phase."""

    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._phase_peak = 0
        self._global_peak = 0
        self._mem_current_path = self._resolve_current_path()

    @staticmethod
    def _resolve_current_path() -> str | None:
        if os.path.exists('/sys/fs/cgroup/cgroup.controllers'):
            rel = (_self_cgroup_paths() or ['/'])[0].lstrip('/')
            p = Path('/sys/fs/cgroup') / rel / 'memory.current'
            return str(p) if p.exists() else None
        return None

    def take_phase_peak(self) -> int:
        peak = self._phase_peak
        self._phase_peak = 0
        return peak

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = _proc_rss() or 0
            self._phase_peak = max(self._phase_peak, rss)
            if rss > self._global_peak:
                self._global_peak = rss
                cur = (
                    _read_int_file(self._mem_current_path)
                    if self._mem_current_path
                    else None
                )
                _emit(
                    {
                        'event': 'rss_high_water',
                        'rss_gb': _gb(rss),
                        'cgroup_current_gb': _gb(cur),
                    }
                )
            self._stop.wait(self._interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        _emit({'event': 'final', 'global_peak_rss_gb': _gb(self._global_peak)})


# ---------------------------------------------------------------------------
# Monkeypatch hooks (observe only)
# ---------------------------------------------------------------------------


def _repatch_everywhere(name: str, orig: Any, wrapped: Any) -> int:
    """Replace every ``name`` attribute in loaded modules that IS ``orig``.

    Callers that did ``from mod import name`` hold their own binding; patching
    only the definition module misses them. In particular ``streaming_cycle``
    imports ``estimate_batch_size`` by value — the very call that OOMs at scale.
    Returns the number of bindings patched.
    """
    n = 0
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, name, None) is orig:
                setattr(mod, name, wrapped)
                n += 1
        except Exception:
            pass
    return n


def _wrap_estimate_batch_size() -> None:
    import learnm8.core.batching as b

    orig = b.estimate_batch_size

    def wrapped(
        learner,
        n_samples,
        n_features,
        device='cpu',
        memory_safety_factor=b.DEFAULT_SAFETY_FACTOR,
        n_jobs=1,
    ):
        batch = orig(
            learner, n_samples, n_features, device, memory_safety_factor, n_jobs
        )
        try:
            prof = learner.memory_profile(n_features)
            cost = prof['bytes_per_sample'] * prof['working_multiplier']
            est = batch * cost
            avail = b.get_available_memory(device)
            cg = _walk_cgroup_memory()
            condor = _read_condor()
            ceilings = [
                c for c in (cg['limit_bytes'], condor.get('request_memory_bytes')) if c
            ]
            ceiling = min(ceilings) if ceilings else None
            over = bool(ceiling and est > 0.9 * ceiling)
            _emit(
                {
                    'event': 'batch_size_decision',
                    'n_samples': n_samples,
                    'batch_size': batch,
                    'safety_factor': memory_safety_factor,
                    'sizer_available_gb': _gb(avail),
                    'predicted_working_set_gb': _gb(est),
                    'effective_ceiling_gb': _gb(ceiling),
                    'EXCEEDS_CEILING': over,
                },
                warn=over,
            )
        except Exception:
            pass
        return batch

    wrapped.__wrapped__ = orig  # type: ignore[attr-defined]
    count = _repatch_everywhere('estimate_batch_size', orig, wrapped)
    logger.debug('[memdiag] patched estimate_batch_size in %d module(s)', count)


def _wrap_phase(module_name: str, func_name: str, phase: str) -> None:
    try:
        import importlib

        mod = importlib.import_module(module_name)
        orig = getattr(mod, func_name)
    except (ImportError, AttributeError):
        return

    def wrapped(*args, **kwargs):
        set_phase(phase)
        return orig(*args, **kwargs)

    wrapped.__wrapped__ = orig  # type: ignore[attr-defined]
    _repatch_everywhere(func_name, orig, wrapped)


def _wrap_run_active_learning() -> None:
    """Capture output_dir to redirect the sidecar, and bracket the whole run."""
    import learnm8.api as api

    orig = api.run_active_learning

    def wrapped(*args, **kwargs):
        with contextlib.suppress(Exception):
            _set_sink_dir(kwargs.get('output_dir'))
        snapshot_environment()
        global _SAMPLER
        if _SAMPLER is None:
            _SAMPLER = _PeakSampler()
            _SAMPLER.start()
        try:
            return orig(*args, **kwargs)
        finally:
            if _SAMPLER is not None:
                _SAMPLER.stop()

    wrapped.__wrapped__ = orig  # type: ignore[attr-defined]
    api.run_active_learning = wrapped


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def activate() -> None:
    """Idempotently install all hooks. Safe to call at package import."""
    global _ACTIVATED, _START
    if _ACTIVATED:
        return
    _ACTIVATED = True
    _START = time.time()
    for fn in (
        lambda: _wrap_run_active_learning(),
        lambda: _wrap_estimate_batch_size(),
        lambda: _wrap_phase(
            'learnm8.core.validation', 'validate_compound_pool', 'validation'
        ),
        lambda: _wrap_phase('learnm8.core.cycle', 'execute_cycle', 'cycle_execute'),
    ):
        try:
            fn()
        except Exception as exc:  # diagnostics must never break import
            logger.debug('memdiag hook failed: %s', exc)
    logger.info('[memdiag] active — sink=%s', _resolve_sink())

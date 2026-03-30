from __future__ import annotations

import contextlib
import datetime
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

from rich.console import Console
from rich.table import Table

from validation.speedup.lib.config import get_results_dir

_console = Console()


def _compute_speedup_metrics(
    baseline_time: float | None,
    optimized_time: float | None,
) -> tuple[float | None, str | None]:
    if baseline_time is None or optimized_time is None:
        return None, None

    speedup_ratio = baseline_time / optimized_time if optimized_time > 0 else float('inf')
    if speedup_ratio > 1.05:
        speedup_label = 'SPEEDUP'
    elif speedup_ratio < 0.95:
        speedup_label = 'REGRESSION'
    else:
        speedup_label = 'NEUTRAL'
    return speedup_ratio, speedup_label


def _get_hardware_info() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        ram_gb = round(ram_bytes / (1024**3), 1)
    except (ValueError, OSError):
        ram_gb = 0.0
    gpu = "N/A"
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {"cpu": cpu, "ram_gb": ram_gb, "gpu": gpu}


def _get_software_info() -> dict:
    info = {"python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"}
    packages = {
        "numpy": "numpy",
        "torch": "torch",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
    }
    for key, dist_name in packages.items():
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            info[key] = importlib.metadata.version(dist_name)
    return info


def create_result(
    optimization_id: int,
    optimization_name: str,
    learner_name: str,
    tier: int,
    baseline_time: float,
    optimized_time: float,
    prediction_match: bool,
    optimization_backend: str | None = None,
    prediction_spearman_rho: float | None = None,
    uncertainty_spearman_rho: float | None = None,
    spearman_rho: float | None = None,
    setup_time: float | None = None,
    training_baseline_time: float | None = None,
    training_optimized_time: float | None = None,
    training_match: bool | None = None,
    training_spearman_rho: float | None = None,
    baseline_times: list[float] | None = None,
    optimized_times: list[float] | None = None,
    training_baseline_times: list[float] | None = None,
    training_optimized_times: list[float] | None = None,
    n_samples: int = 0,
    n_features: int = 0,
    n_warmup_runs: int = 0,
    n_timed_runs: int = 10,
) -> dict:
    speedup_ratio, speedup_label = _compute_speedup_metrics(
        baseline_time,
        optimized_time,
    )
    training_speedup_ratio, training_speedup_label = _compute_speedup_metrics(
        training_baseline_time,
        training_optimized_time,
    )

    prediction_spearman_rho = (
        prediction_spearman_rho
        if prediction_spearman_rho is not None
        else spearman_rho
    )

    result = {
        "optimization_id": optimization_id,
        "optimization_name": optimization_name,
        "learner_name": learner_name,
        "tier": tier,
        "optimization_backend": optimization_backend,
        "baseline_time": baseline_time,
        "optimized_time": optimized_time,
        "speedup_ratio": speedup_ratio,
        "speedup_label": speedup_label,
        "prediction_match": prediction_match,
        "prediction_spearman_rho": prediction_spearman_rho,
        "uncertainty_spearman_rho": uncertainty_spearman_rho,
        "spearman_rho": prediction_spearman_rho,
        "uncertainty_spearman": uncertainty_spearman_rho,
        "setup_time": setup_time,
        "training_baseline_time": training_baseline_time,
        "training_optimized_time": training_optimized_time,
        "training_speedup_ratio": training_speedup_ratio,
        "training_speedup_label": training_speedup_label,
        "training_match": training_match,
        "training_spearman_rho": training_spearman_rho,
        "baseline_times": baseline_times,
        "optimized_times": optimized_times,
        "training_baseline_times": training_baseline_times,
        "training_optimized_times": training_optimized_times,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_warmup_runs": n_warmup_runs,
        "n_timed_runs": n_timed_runs,
        "scope": "mechanism_validation",
        "hardware_info": _get_hardware_info(),
        "software_info": _get_software_info(),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    return result


def normalize_result_metrics(result: dict) -> dict:
    prediction_rho = result.get("prediction_spearman_rho")
    if prediction_rho is None:
        prediction_rho = result.get("spearman_rho")

    uncertainty_rho = result.get("uncertainty_spearman_rho")
    if uncertainty_rho is None:
        uncertainty_rho = result.get("uncertainty_spearman")

    result["prediction_spearman_rho"] = prediction_rho
    result["uncertainty_spearman_rho"] = uncertainty_rho
    result["spearman_rho"] = prediction_rho
    result["uncertainty_spearman"] = uncertainty_rho
    return result


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_result(result: dict, output_dir: Path) -> Path:
    normalize_result_metrics(result)
    output_dir = get_results_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result['optimization_name']}_{result['learner_name']}.json"
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(result, f, indent=2, cls=_NumpyEncoder)
    return path


def _speedup_color(label: str) -> str:
    if label == "SPEEDUP":
        return "green"
    if label == "REGRESSION":
        return "red"
    return "yellow"


def _fmt_time(t: float | None) -> str:
    if t is None:
        return "N/A"
    return f"{t:.3f}s"


def print_result_table(results: list[dict]) -> None:
    table = Table(title="Benchmark Results")
    table.add_column("Optimization")
    table.add_column("Learner")
    table.add_column("Backend")
    show_training = any(r.get('training_baseline_time') is not None for r in results)
    if show_training:
        table.add_column("Train (B)")
        table.add_column("Train (O)")
    show_setup = any(r.get('setup_time') is not None for r in results)
    if show_setup:
        table.add_column("Setup")
    table.add_column("Pred (B)")
    table.add_column("Pred (O)")
    table.add_column("Speedup")
    table.add_column("Match")
    table.add_column("Pred rho")
    table.add_column("Unc rho")

    for r in results:
        normalize_result_metrics(r)
        label = r.get("speedup_label", "NEUTRAL")
        color = _speedup_color(label)
        ratio = r.get("speedup_ratio", 0.0)
        match_str = "PASS" if r.get("prediction_match") else "FAIL"
        prediction_rho = r.get("prediction_spearman_rho")
        prediction_rho_str = (
            f"{prediction_rho:.4f}" if prediction_rho is not None else "N/A"
        )
        uncertainty_rho = r.get("uncertainty_spearman_rho")
        uncertainty_rho_str = (
            f"{uncertainty_rho:.4f}" if uncertainty_rho is not None else "N/A"
        )
        row = [
            r.get("optimization_name", ""),
            r.get("learner_name", ""),
            r.get("optimization_backend", "") or "N/A",
        ]
        if show_training:
            row.append(_fmt_time(r.get('training_baseline_time')))
            row.append(_fmt_time(r.get('training_optimized_time')))
        if show_setup:
            row.append(_fmt_time(r.get('setup_time')))
        row.extend([
            _fmt_time(r.get('baseline_time')),
            _fmt_time(r.get('optimized_time')),
            f"[{color}]{ratio:.2f}x ({label})[/{color}]",
            match_str,
            prediction_rho_str,
            uncertainty_rho_str,
        ])
        table.add_row(*row)

    _console.print(table)


def print_tiered_summary(results: list[dict]) -> None:
    table = Table(title="Tiered Summary")
    table.add_column("Tier")
    table.add_column("Optimization")
    table.add_column("Learner")
    table.add_column("Backend")
    show_training = any(r.get('training_baseline_time') is not None for r in results)
    if show_training:
        table.add_column("Train (B)")
        table.add_column("Train (O)")
    show_setup = any(r.get('setup_time') is not None for r in results)
    if show_setup:
        table.add_column("Setup")
    table.add_column("Pred (B)")
    table.add_column("Pred (O)")
    table.add_column("Speedup")
    table.add_column("Label")
    table.add_column("Match")
    table.add_column("Pred rho")
    table.add_column("Unc rho")

    has_breakeven = False
    by_tier: dict[int, list[dict]] = {}
    for r in results:
        tier = r.get("tier", 0)
        by_tier.setdefault(tier, []).append(r)

    for tier in sorted(by_tier):
        for r in by_tier[tier]:
            normalize_result_metrics(r)
            label = r.get("speedup_label", "NEUTRAL")
            color = _speedup_color(label)
            ratio = r.get("speedup_ratio", 0.0)
            match_str = "PASS" if r.get("prediction_match") else "FAIL"

            prediction_rho = r.get("prediction_spearman_rho")
            prediction_rho_str = (
                f"{prediction_rho:.4f}" if prediction_rho is not None else ""
            )
            uncertainty_rho = r.get("uncertainty_spearman_rho")
            uncertainty_rho_str = (
                f"{uncertainty_rho:.4f}" if uncertainty_rho is not None else ""
            )

            speedup_str = f"[{color}]{ratio:.2f}x[/{color}]"
            setup_time = r.get("setup_time")
            if setup_time is not None and setup_time > 0:
                optimized_time = r.get("optimized_time", 0.0)
                baseline_time = r.get("baseline_time", 0.0)
                if optimized_time > 0 and baseline_time > 0:
                    saving = baseline_time - optimized_time
                    if saving > 0:
                        breakeven = int(setup_time / saving) + 1
                        speedup_str += f" (breakeven: {breakeven})"
                        has_breakeven = True

            row = [
                str(tier),
                r.get("optimization_name", ""),
                r.get("learner_name", ""),
                r.get("optimization_backend", "") or "N/A",
            ]
            if show_training:
                row.append(_fmt_time(r.get('training_baseline_time')))
                row.append(_fmt_time(r.get('training_optimized_time')))
            if show_setup:
                row.append(_fmt_time(r.get('setup_time')))
            row.extend([
                _fmt_time(r.get('baseline_time')),
                _fmt_time(r.get('optimized_time')),
                speedup_str,
                f"[{color}]{label}[/{color}]",
                match_str,
                prediction_rho_str,
                uncertainty_rho_str,
            ])
            table.add_row(*row)

    _console.print(table)
    if has_breakeven:
        _console.print(
            "[dim]Breakeven = number of predict() calls needed to recoup "
            "the one-time setup cost of the optimization.[/dim]"
        )

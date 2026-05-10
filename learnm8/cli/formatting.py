"""Rich-formatted metrics table display for the LearnM8 CLI.

Moved from learnm8.utils.logging_formatters per simplification §3.2.20 —
API code now uses plain-text formatting.
"""

from typing import Any


def format_cycle_metrics_table(
    metrics: dict[str, Any],
    oracle_type: str = 'auto',
    previous_metrics: dict[str, Any] | None = None,
    score_direction: str = 'higher'
) -> str:
    lines = []

    cycle = metrics.get('cycle', '?')
    batch_size = metrics.get('batch_size', '?')
    cumulative_labeled = metrics.get('cumulative_labeled', '?')
    is_benchmark = (oracle_type == 'benchmark')

    bad_metrics = {'rmse', 'mae', 'mse'}
    if score_direction == 'lower':
        bad_metrics = bad_metrics | {'avg_score_selected'}

    def get_change(key: str, current_val: float, is_pct: bool = False) -> str:
        if previous_metrics is None or key not in previous_metrics:
            return ""

        prev_val = previous_metrics.get(key)
        if prev_val is None or current_val is None:
            return ""

        try:
            diff = float(current_val) - float(prev_val)
        except (ValueError, TypeError):
            return ""

        if is_pct or 'discovery' in key or 'overlap' in key:
            stagnation_threshold = 1.0
        else:
            stagnation_threshold = 0.01

        if abs(diff) < stagnation_threshold:
            return "[yellow]→[/yellow]"

        is_higher_better = key not in bad_metrics
        is_improvement = (diff > 0) if is_higher_better else (diff < 0)

        return "[green]↑[/green]" if is_improvement else "[red]↓[/red]"

    def format_metric(key: str, label: str, digits: int = 1, is_pct: bool = False) -> str:
        val = metrics.get(key)
        if val is None:
            return f"{label}:N/A"

        change = get_change(key, val, is_pct)
        if is_pct:
            return f"{label}:{val:.{digits}f}%{change}"
        else:
            return f"{label}:{val:.{digits}f}{change}"

    lines.append(f"Cycle {cycle} | {batch_size} selected | {cumulative_labeled} labeled")
    lines.append("─" * 60)

    if is_benchmark:
        discovery_row1 = [
            format_metric('top_10_discovery', 'Top10', 1, True),
            format_metric('top_100_discovery', 'Top100', 1, True),
            format_metric('top_1000_discovery', 'Top1K', 1, True),
        ]
        discovery_row2 = [
            format_metric('top_0_1_pct_discovery', 'Top0.1%', 1, True),
            format_metric('top_1_pct_discovery', 'Top1%', 1, True),
        ]
        lines.append(f"Discovery │ {' '.join(discovery_row1)}")
        lines.append(f"          │ {' '.join(discovery_row2)}")

        ranking_parts = [
            format_metric('unlabeled_spearman_correlation', 'Spearman', 3, False),
            format_metric('unlabeled_ef_1_0', 'EF@1%', 2, False),
            format_metric('unlabeled_ef_5_0', 'EF@5%', 2, False),
        ]
        lines.append(f"Ranking   │ {' '.join(ranking_parts)}")

    selection_parts = [format_metric('avg_score_selected', 'Avg', 2, False)]

    if metrics.get('uncertainty_mean') is not None:
        selection_parts.append(format_metric('uncertainty_mean', 'Unc', 3, False))

    lines.append(f"Selection │ {' '.join(selection_parts)}")

    return "\n".join(lines)

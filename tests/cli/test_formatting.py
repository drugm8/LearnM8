"""Tests for CLI cycle metrics formatting (moved from utils per §3.2.20)."""

import pytest
from learnm8.cli.formatting import format_cycle_metrics_table


@pytest.mark.unit
class TestFormatCycleMetricsTable:

    @staticmethod
    def strip_ansi(text):
        import re
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        return ansi_escape.sub('', text)

    def test_benchmark_mode_table(self):
        metrics = {
            'cycle': 1,
            'batch_size': 50,
            'avg_score_selected': 0.823,
            'avg_score_ground_truth': 0.756,
            'cumulative_labeled': 150,
            'diversity_score': 0.432,
            'top_10_discovery': 23.5,
            'enrichment_factor_10': 4.52,
            'unlabeled_top_10_discovery': 45.2,
            'spearman_correlation': 0.678,
        }

        result = format_cycle_metrics_table(metrics, oracle_type='benchmark')
        clean_result = self.strip_ansi(result)

        assert 'Cycle 1' in clean_result
        assert 'Selection' in clean_result
        assert 'Discovery' in clean_result
        assert 'Ranking' in clean_result

    def test_run_mode_table(self):
        metrics = {
            'cycle': 2,
            'batch_size': 100,
            'avg_score_selected': 0.891,
            'cumulative_labeled': 250,
            'diversity_score': 0.389,
        }

        result = format_cycle_metrics_table(metrics, oracle_type='run')
        clean_result = self.strip_ansi(result)

        assert 'Cycle 2' in clean_result
        assert 'Selection' in clean_result

    def test_with_previous_metrics(self):
        current_metrics = {
            'cycle': 2,
            'batch_size': 50,
            'avg_score_selected': 0.850,
            'cumulative_labeled': 100,
        }

        previous_metrics = {
            'cycle': 1,
            'batch_size': 50,
            'avg_score_selected': 0.800,
            'cumulative_labeled': 50,
        }

        result = format_cycle_metrics_table(
            current_metrics,
            oracle_type='run',
            previous_metrics=previous_metrics,
        )
        clean_result = self.strip_ansi(result)

        assert 'Cycle 2' in clean_result

    def test_handles_missing_optional_metrics(self):
        metrics = {
            'cycle': 1,
            'batch_size': 50,
        }

        result = format_cycle_metrics_table(metrics, oracle_type='run')
        assert result is not None

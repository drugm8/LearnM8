"""Tests for logging formatters module."""

import pytest
import polars as pl
import numpy as np
from learnm8.utils.logging_formatters import (
    format_cycle_schedule,
    format_duration,
    format_experiment_summary
)
from learnm8.core.config import CycleConfig


@pytest.mark.unit
class TestFormatCycleSchedule:
    """Test cycle schedule formatting."""

    def test_single_cycle_no_pruning(self):
        """Format single cycle without pruning."""
        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.01)
        result = format_cycle_schedule(1, config, 10000)

        assert 'Cycle 1' in result
        assert 'GREEDY acquisition' in result
        assert '100 compounds' in result
        assert '1.0% batch size' in result
        assert 'pruned' not in result

    def test_multiple_cycles_with_pruning(self):
        """Format multiple cycles with pruning."""
        config = CycleConfig(
            'ucb',
            n_cycles=5,
            batch_fraction=0.005,
            pruning_strategy='score_based',
            pruning_params={'pruning_fraction': 0.3}
        )
        result = format_cycle_schedule(1, config, 10000)

        assert 'Cycles 1-5' in result
        assert 'UCB acquisition' in result
        assert '50 compounds per cycle' in result
        assert '30% pruned per cycle' in result

    def test_random_strategy_formatting(self):
        """Format random strategy."""
        config = CycleConfig('random', n_cycles=2, batch_fraction=0.02)
        result = format_cycle_schedule(3, config, 5000)

        assert 'Cycles 3-4' in result
        assert 'RANDOM acquisition' in result
        assert '100 compounds per cycle' in result

    def test_pruning_without_params(self):
        """Handle pruning strategy without params (uses default)."""
        config = CycleConfig(
            'greedy',
            n_cycles=1,
            batch_fraction=0.01,
            pruning_strategy='score_based',
            pruning_params=None
        )
        result = format_cycle_schedule(1, config, 10000)

        assert '30% pruned per cycle' in result


@pytest.mark.unit
class TestFormatDuration:
    """Test duration formatting."""

    def test_seconds(self):
        """Format duration under 60 seconds."""
        result = format_duration(45.7)
        assert '45.7 seconds' in result

    def test_minutes(self):
        """Format duration in minutes."""
        result = format_duration(154)
        assert '2 minutes 34 seconds' in result

    def test_hours(self):
        """Format duration in hours."""
        result = format_duration(7384)
        assert '2 hours 3 minutes' in result

    def test_exact_minute(self):
        """Format exact minute boundary."""
        result = format_duration(120)
        assert '2 minutes 0 seconds' in result

    def test_exact_hour(self):
        """Format exact hour boundary."""
        result = format_duration(3600)
        assert '1 hours 0 minutes' in result




@pytest.mark.unit
class TestFormatExperimentSummary:
    """Test experiment summary formatting."""

    def test_summary_with_pruning(self):
        """Format summary with labeled, pruned, and unlabeled compounds."""
        df = pl.DataFrame({
            'ID': range(1000),
            'status': ['labeled'] * 300 + ['pruned'] * 200 + ['unlabeled'] * 500
        })

        lines = format_experiment_summary(df, 3723.5, 10)

        assert len(lines) >= 4

        summary = '\n'.join(lines)

        assert '300' in summary
        assert '200' in summary
        assert '500' in summary
        assert '30.0%' in summary

    def test_summary_no_pruning(self):
        """Format summary without pruning."""
        df = pl.DataFrame({
            'ID': range(1000),
            'status': ['labeled'] * 400 + ['unlabeled'] * 600
        })

        lines = format_experiment_summary(df, 154.2, 5)
        summary = '\n'.join(lines)

        assert '400' in summary
        assert '600' in summary
        assert len([line for line in lines if 'pruned' in line.lower()]) == 0

    def test_summary_all_labeled(self):
        """Format summary when all compounds are labeled."""
        df = pl.DataFrame({
            'ID': range(100),
            'status': ['labeled'] * 100
        })

        lines = format_experiment_summary(df, 60.5, 3)
        summary = '\n'.join(lines)

        assert '100.0%' in summary
        assert '100 of 100' in summary

    def test_duration_formats_correctly(self):
        """Verify duration is formatted in summary."""
        df = pl.DataFrame({
            'ID': range(100),
            'status': ['labeled'] * 50 + ['unlabeled'] * 50
        })

        lines_short = format_experiment_summary(df, 45.0, 2)
        assert any('seconds' in line for line in lines_short)

        lines_long = format_experiment_summary(df, 7200, 10)
        assert any('hours' in line or 'hour' in line for line in lines_long)

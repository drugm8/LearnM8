"""Comprehensive CLI tests for LearnM8.

Tests all subcommands (run, list, validate) and major CLI options.
Uses subprocess.run to invoke CLI in isolation.
"""

import subprocess
import sys
import json
from pathlib import Path
import pandas as pd
import polars as pl
import pytest


@pytest.fixture
def minimal_compounds(tmp_path):
    """Create minimal valid compound pool CSV."""
    csv_path = tmp_path / "compounds.csv"
    df = pd.DataFrame({
        'ID': ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004', 'COMP_005'],
        'SMILES': ['CCO', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC'],
        'Activity': [0.1, 0.3, 0.5, 0.7, 0.9]
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def oracle_csv(tmp_path):
    """Create oracle CSV file."""
    csv_path = tmp_path / "oracle.csv"
    df = pd.DataFrame({
        'ID': ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004', 'COMP_005'],
        'SMILES': ['CCO', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC'],
        'Activity': [0.15, 0.35, 0.55, 0.65, 0.85]
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def config_yaml(tmp_path):
    """Create YAML config file."""
    config_path = tmp_path / "config.yaml"
    config_content = """
target_col: Activity
featurizer: morgan
learner: rf
n_cycles: 3
batch_fraction: 0.2
strategy: greedy
random_state: 42
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def config_json(tmp_path):
    """Create JSON config file."""
    config_path = tmp_path / "config.json"
    config = {
        "target_col": "Activity",
        "featurizer": "morgan",
        "learner": "rf",
        "n_cycles": 3,
        "batch_fraction": 0.2,
        "strategy": "greedy",
        "random_state": 42
    }
    config_path.write_text(json.dumps(config))
    return config_path


def run_cli(*args, timeout=60):
    """Helper function to run CLI command with optional timeout."""
    cmd = [sys.executable, '-m', 'learnm8.cli.main'] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result


@pytest.mark.slow
class TestRunSubcommand:
    """Test 'run' subcommand functionality."""

    def test_minimal_invocation(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / "output"

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--n-cycles', '2',
            '--batch-fraction', '0.4',
            '-o', str(output_dir)
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_dir.exists()
        assert (output_dir / 'compounds_final.csv').exists()
        assert (output_dir / 'cycle_metrics.csv').exists()

    def test_with_explicit_oracle(self, minimal_compounds, oracle_csv, tmp_path):
        output_dir = tmp_path / "output"

        result = run_cli(
            'run',
            str(minimal_compounds),
            str(oracle_csv),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--n-cycles', '2',
            '--batch-fraction', '0.4',
            '-o', str(output_dir)
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_dir.exists()

    def test_cycles_spec_parsing(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / "output"

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--cycles', 'random:0.2 greedy:0.2*2',
            '-o', str(output_dir)
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_dir.exists()

        metrics = pd.read_csv(output_dir / 'cycle_metrics.csv')
        assert len(metrics) == 3

    def test_output_dir_creation(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / "nested" / "output" / "dir"

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--n-cycles', '1',
            '--batch-fraction', '0.4',
            '-o', str(output_dir)
        )

        assert result.returncode == 0
        assert output_dir.exists()
        assert (output_dir / 'compounds_final.csv').exists()

    def test_pruning_flags(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / "output"

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--n-cycles', '2',
            '--batch-fraction', '0.4',
            '--pruning-fraction', '0.3',
            '-o', str(output_dir)
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_learner_selection(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / "output"

        learners = ['rf', 'gp', 'xgb']

        for learner in learners:
            result = run_cli(
                'run',
                str(minimal_compounds),
                '--target', 'Activity',
                '--featurizer', 'morgan',
                '--learner', learner,
                '--n-cycles', '1',
                '--batch-fraction', '0.4',
                '-o', str(output_dir / learner)
            )

            assert result.returncode == 0, f"Learner {learner} failed: {result.stderr}"

    def test_acquisition_selection(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / "output"

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'gp',
            '--strategy', 'ucb',
            '--n-cycles', '2',
            '--batch-fraction', '0.4',
            '-o', str(output_dir)
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_predefined_schedules(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / "output"

        schedules = ['quick', 'standard', 'intensive']

        for schedule in schedules:
            result = run_cli(
                'run',
                str(minimal_compounds),
                '--target', 'Activity',
                '--featurizer', 'morgan',
                '--learner', 'rf',
                '--schedule', schedule,
                '-o', str(output_dir / schedule)
            )

            assert result.returncode == 0, f"Schedule {schedule} failed: {result.stderr}"

    def test_config_yaml(self, minimal_compounds, config_yaml, tmp_path):
        pytest.importorskip('yaml', reason="PyYAML not installed")

        output_dir = tmp_path / "output"

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--config', str(config_yaml),
            '-o', str(output_dir)
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_dir.exists()

    def test_config_json(self, minimal_compounds, config_json, tmp_path):
        output_dir = tmp_path / "output"

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--config', str(config_json),
            '-o', str(output_dir)
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_dir.exists()

    def test_missing_required_args(self, minimal_compounds):
        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--learner', 'rf'
        )

        assert result.returncode != 0

    def test_invalid_file(self, tmp_path):
        result = run_cli(
            'run',
            str(tmp_path / "nonexistent.csv"),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf'
        )

        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_invalid_learner(self, minimal_compounds, tmp_path):
        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'invalid_learner',
            '-o', str(tmp_path / "output")
        )

        assert result.returncode != 0

    def test_featurizer_options(self, minimal_compounds, tmp_path):
        featurizers = ['morgan', 'maccs', 'ecfp6']

        for feat in featurizers:
            output_dir = tmp_path / feat
            result = run_cli(
                'run',
                str(minimal_compounds),
                '--target', 'Activity',
                '--featurizer', feat,
                '--learner', 'rf',
                '--n-cycles', '1',
                '--batch-fraction', '0.4',
                '-o', str(output_dir)
            )

            assert result.returncode == 0, f"Featurizer {feat} failed: {result.stderr}"

    def test_score_direction(self, minimal_compounds, tmp_path):
        for direction in ['higher', 'lower']:
            output_dir = tmp_path / direction
            result = run_cli(
                'run',
                str(minimal_compounds),
                '--target', 'Activity',
                '--featurizer', 'morgan',
                '--learner', 'rf',
                '--score-direction', direction,
                '--n-cycles', '1',
                '--batch-fraction', '0.4',
                '-o', str(output_dir)
            )

            assert result.returncode == 0, f"Direction {direction} failed"

    def test_mode_parameter(self, minimal_compounds, tmp_path):
        for mode in ['run', 'benchmark']:
            output_dir = tmp_path / mode
            result = run_cli(
                'run',
                str(minimal_compounds),
                '--target', 'Activity',
                '--featurizer', 'morgan',
                '--learner', 'rf',
                '--mode', mode,
                '--n-cycles', '1',
                '--batch-fraction', '0.4',
                '-o', str(output_dir)
            )

            assert result.returncode == 0, f"Mode {mode} failed"

    def test_random_state(self, minimal_compounds, tmp_path):
        output_dir1 = tmp_path / "run1"
        output_dir2 = tmp_path / "run2"

        result1 = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--random-state', '42',
            '--n-cycles', '2',
            '--batch-fraction', '0.4',
            '-o', str(output_dir1)
        )

        result2 = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--random-state', '42',
            '--n-cycles', '2',
            '--batch-fraction', '0.4',
            '-o', str(output_dir2)
        )

        assert result1.returncode == 0
        assert result2.returncode == 0

        df1 = pd.read_csv(output_dir1 / 'compounds_final.csv')
        df2 = pd.read_csv(output_dir2 / 'compounds_final.csv')

        pd.testing.assert_frame_equal(
            df1.sort_values('ID').reset_index(drop=True),
            df2.sort_values('ID').reset_index(drop=True)
        )

    def test_quiet_flag(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / "output"

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--n-cycles', '1',
            '--batch-fraction', '0.4',
            '--quiet',
            '-o', str(output_dir)
        )

        assert result.returncode == 0


@pytest.mark.slow
class TestListSubcommand:
    """Test 'list' subcommand functionality."""

    def test_list_learners(self):
        result = run_cli('list', 'learners')

        assert result.returncode == 0
        assert 'rf' in result.stdout or 'gp' in result.stdout

    def test_list_acquisition(self):
        result = run_cli('list', 'acquisition')

        assert result.returncode == 0
        assert 'greedy' in result.stdout or 'random' in result.stdout

    def test_list_featurizers(self):
        result = run_cli('list', 'featurizers')

        assert result.returncode == 0
        assert 'morgan' in result.stdout
        assert 'maccs' in result.stdout

    def test_list_schedules(self):
        result = run_cli('list', 'schedules')

        assert result.returncode == 0
        assert 'quick' in result.stdout
        assert 'standard' in result.stdout
        assert 'intensive' in result.stdout

    def test_list_invalid_component(self):
        result = run_cli('list', 'invalid')

        assert result.returncode != 0


@pytest.mark.slow
class TestValidateSubcommand:
    """Test 'validate' subcommand functionality."""

    def test_validate_valid_compounds(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / "validation"

        result = run_cli(
            'validate',
            str(minimal_compounds),
            '-o', str(output_dir)
        )

        assert result.returncode == 0
        assert 'Valid compounds:' in result.stdout

    def test_validate_with_output(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / "validation"

        result = run_cli(
            'validate',
            str(minimal_compounds),
            '-o', str(output_dir)
        )

        assert result.returncode == 0
        assert output_dir.exists()
        assert (output_dir / 'validation_report.csv').exists()

    def test_validate_invalid_file(self, tmp_path):
        result = run_cli(
            'validate',
            str(tmp_path / "nonexistent.csv")
        )

        assert result.returncode != 0


@pytest.mark.slow
class TestHelpAndErrors:
    """Test help messages and error handling."""

    def test_no_args_shows_help(self):
        result = run_cli()

        assert result.returncode == 0
        assert 'learnm8' in result.stdout.lower() or 'usage' in result.stdout.lower()

    def test_run_help(self):
        result = run_cli('run', '-h')

        assert result.returncode == 0
        assert 'compound_pool' in result.stdout

    def test_list_help(self):
        result = run_cli('list', '-h')

        assert result.returncode == 0
        assert 'Component type' in result.stdout or 'component' in result.stdout.lower()

    def test_validate_help(self):
        result = run_cli('validate', '-h')

        assert result.returncode == 0
        assert 'compound_pool' in result.stdout


@pytest.mark.slow
class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_empty_compound_pool(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        df = pd.DataFrame(columns=['ID', 'SMILES', 'Activity'])
        df.to_csv(csv_path, index=False)

        result = run_cli(
            'run',
            str(csv_path),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '-o', str(tmp_path / "output")
        )

        assert result.returncode != 0

    def test_missing_required_columns(self, tmp_path):
        csv_path = tmp_path / "bad_columns.csv"
        df = pd.DataFrame({
            'compound_id': ['C1', 'C2'],
            'structure': ['CCO', 'CCC']
        })
        df.to_csv(csv_path, index=False)

        result = run_cli(
            'run',
            str(csv_path),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '-o', str(tmp_path / "output")
        )

        assert result.returncode != 0

    def test_invalid_cycles_spec(self, minimal_compounds, tmp_path):
        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--cycles', 'invalid_format',
            '-o', str(tmp_path / "output")
        )

        assert result.returncode != 0

    def test_invalid_schedule(self, minimal_compounds, tmp_path):
        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--schedule', 'nonexistent',
            '-o', str(tmp_path / "output")
        )

        assert result.returncode != 0

    @pytest.mark.slow
    def test_invalid_pruning_fraction(self, minimal_compounds, tmp_path):
        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--pruning-fraction', '1.5',
            '-o', str(tmp_path / "output")
        )

        assert result.returncode != 0

    @pytest.mark.slow
    def test_acquisition_params_json(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / "output"

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'gp',
            '--strategy', 'ucb',
            '--acquisition-params', '{"beta": 2.0}',
            '--n-cycles', '1',
            '--batch-fraction', '0.4',
            '-o', str(output_dir)
        )

        assert result.returncode == 0

    def test_invalid_acquisition_params_json(self, minimal_compounds, tmp_path):
        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--acquisition-params', '{invalid json}',
            '-o', str(tmp_path / "output")
        )

        assert result.returncode != 0


@pytest.mark.slow
class TestIntegration:
    """Integration tests across CLI features."""

    def test_full_workflow_with_all_options(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / "full_workflow"

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'gp',
            '--cycles', 'random:0.2 ucb:0.2*2',
            '--score-direction', 'higher',
            '--pruning-fraction', '0.2',
            '--random-state', '123',
            '-o', str(output_dir)
        )

        assert result.returncode == 0
        assert (output_dir / 'compounds_final.csv').exists()
        assert (output_dir / 'cycle_metrics.csv').exists()
        assert (output_dir / 'selection_history.csv').exists()

        final_df = pd.read_csv(output_dir / 'compounds_final.csv')
        assert 'ID' in final_df.columns
        assert 'SMILES' in final_df.columns
        assert 'status' in final_df.columns

    def test_benchmark_mode_auto_detection(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / "benchmark"

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--n-cycles', '1',
            '--batch-fraction', '0.4',
            '-o', str(output_dir)
        )

        assert result.returncode == 0

    @pytest.mark.xfail(reason="Config loading overwrites CLI args instead of providing defaults")
    def test_cli_flags_override_config_file(self, minimal_compounds, tmp_path):
        config_path = tmp_path / "override_config.yaml"
        config_content = """
target_col: Activity
featurizer: morgan
learner: rf
n_cycles: 1
batch_fraction: 0.2
strategy: greedy
random_state: 42
"""
        config_path.write_text(config_content)

        output_dir = tmp_path / "override"

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--config', str(config_path),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'gp',
            '--n-cycles', '2',
            '-o', str(output_dir),
            timeout=120
        )

        assert result.returncode == 0

        compounds_final = pd.read_csv(output_dir / 'compounds_final.csv')
        assert len(compounds_final) > 0

        config_used = json.loads((output_dir / 'config.json').read_text())
        assert config_used['learner'] == 'gp'
        assert config_used['n_cycles'] == 2

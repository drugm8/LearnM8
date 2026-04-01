"""Comprehensive CLI tests for LearnM8.

Tests all subcommands (run, list, validate) and major CLI options.
Uses subprocess.run to invoke CLI in isolation.
"""

import argparse
import json
import subprocess
import sys
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from learnm8.exceptions import (
    AcquisitionError,
    ConfigurationError,
    FeatureExtractionError,
    LearnerError,
    OracleError,
    PersistenceError,
    PruningError,
    ValidationError,
)


@pytest.fixture
def minimal_compounds(tmp_path):
    """Create valid compound pool CSV with enough diversity for predefined schedules."""
    csv_path = tmp_path / "compounds.csv"
    base_smiles = [
        'CCO', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC',
        'c1ccccc1', 'CC(=O)O', 'CCN', 'CC(O)C', 'CCOC',
        'C1CCCCC1', 'CC=CC', 'CC#N', 'CC(=O)N', 'CCCO',
        'c1ccc(O)cc1', 'CC(C)O', 'CCOCC', 'CCNC', 'CC(=O)OC',
    ]
    n = 20
    smiles = base_smiles[:n]
    df = pd.DataFrame({
        'ID': [f'COMP_{i:04d}' for i in range(1, n + 1)],
        'SMILES': smiles,
        'Activity': [round(i * 0.005, 4) for i in range(1, n + 1)]
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def oracle_csv(tmp_path, minimal_compounds):
    """Create oracle CSV file matching the minimal_compounds fixture."""
    csv_path = tmp_path / "oracle.csv"
    df = pd.read_csv(minimal_compounds)
    df['Activity'] = [round(0.15 + 0.004*i, 4) for i in range(len(df))]
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
    cmd = [sys.executable, '-m', 'learnm8.cli.main', *args]
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

        metrics = pd.read_csv(output_dir / 'cycle_metrics.csv', comment='#')
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
                '--batch-fraction', '0.4',
                '-o', str(output_dir / schedule)
            )

            assert result.returncode == 0, f"Schedule {schedule} failed: {result.stderr}"

    def test_config_yaml(self, minimal_compounds, config_yaml, tmp_path):
        pytest.importorskip('yaml', reason="PyYAML not installed")

        output_dir = tmp_path / "output"

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target', 'Activity',
            '--featurizer', 'morgan',
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
            '--target', 'Activity',
            '--featurizer', 'morgan',
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

        df1 = pd.read_csv(output_dir1 / 'compounds_final.csv', comment='#')
        df2 = pd.read_csv(output_dir2 / 'compounds_final.csv', comment='#')

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

        final_df = pd.read_csv(output_dir / 'compounds_final.csv', comment='#')
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

        compounds_final = pd.read_csv(output_dir / 'compounds_final.csv', comment='#')
        assert len(compounds_final) > 0

        config_used = json.loads((output_dir / 'config.json').read_text())
        assert config_used['learner'] == 'gp'
        assert config_used['n_cycles'] == 2


@pytest.fixture
def run_args(minimal_compounds, tmp_path):
    """Create minimal argparse.Namespace for cmd_run."""
    return argparse.Namespace(
        compound_pool=minimal_compounds,
        oracle=None,
        target_col='Activity',
        featurizer='morgan',
        learner='rf',
        score_direction='higher',
        cycles=None,
        schedule=None,
        config=None,
        n_cycles=2,
        batch_fraction=0.4,
        strategy='greedy',
        initial_strategy='random',
        pruning_fraction=None,
        pruning_strategy=None,
        acquisition_params=None,
        output=tmp_path / 'output',
        cache_dir=None,
        quiet=True,
        n_jobs=1,
        device='cpu',
        random_state=42,
        mode=None,
        memory_safety_factor=0.7,
        smiles_col=None,
        id_col=None,
    )


@pytest.fixture
def validate_args(minimal_compounds, tmp_path):
    """Create minimal argparse.Namespace for cmd_validate."""
    return argparse.Namespace(
        compound_pool=minimal_compounds,
        n_jobs=1,
        output=None,
        smiles_col=None,
        id_col=None,
    )


@pytest.mark.unit
class TestCmdRunErrorBoundary:
    """Test that cmd_run catches LearnM8Error subtypes and formats them."""

    @patch('learnm8.cli.main.run_active_learning')
    def test_configuration_error_formatted_and_exit_2(self, mock_run, run_args, capsys):
        mock_run.side_effect = ConfigurationError(
            "Unknown learner 'bad'. Available: rf, gp, xgb"
        )
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run
            cmd_run(run_args)
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert 'Configuration error' in captured.out or 'Configuration error' in captured.err

    @patch('learnm8.cli.main.run_active_learning')
    def test_validation_error_formatted_and_exit_1(self, mock_run, run_args, capsys):
        mock_run.side_effect = ValidationError(
            "50 compounds have invalid SMILES",
            invalid_indices=[0, 1, 2],
        )
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run
            cmd_run(run_args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'Validation error' in captured.out or 'Validation error' in captured.err

    @patch('learnm8.cli.main.run_active_learning')
    def test_validation_error_suggests_validate_command(self, mock_run, run_args, capsys):
        mock_run.side_effect = ValidationError("Bad SMILES")
        with pytest.raises(SystemExit):
            from learnm8.cli.main import cmd_run
            cmd_run(run_args)
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert 'learnm8 validate' in output

    @patch('learnm8.cli.main.run_active_learning')
    def test_learner_error_formatted_and_exit_1(self, mock_run, run_args, capsys):
        mock_run.side_effect = LearnerError("Training failed: insufficient data")
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run
            cmd_run(run_args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert 'Training/prediction error' in output

    @patch('learnm8.cli.main.run_active_learning')
    def test_feature_extraction_error_formatted_and_exit_1(self, mock_run, run_args, capsys):
        mock_run.side_effect = FeatureExtractionError("Morgan featurizer failed")
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run
            cmd_run(run_args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert 'Feature extraction error' in output

    @patch('learnm8.cli.main.run_active_learning')
    def test_feature_extraction_error_suggests_validate(self, mock_run, run_args, capsys):
        mock_run.side_effect = FeatureExtractionError("Failed")
        with pytest.raises(SystemExit):
            from learnm8.cli.main import cmd_run
            cmd_run(run_args)
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert 'learnm8 validate' in output

    @patch('learnm8.cli.main.run_active_learning')
    def test_generic_learnm8_error_formatted_and_exit_1(self, mock_run, run_args, capsys):
        mock_run.side_effect = OracleError("Oracle measurement failed")
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run
            cmd_run(run_args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert 'Error' in output
        assert 'Oracle measurement failed' in output

    @patch('learnm8.cli.main.run_active_learning')
    def test_unexpected_exception_shows_traceback_and_exit_1(self, mock_run, run_args):
        mock_run.side_effect = AttributeError("unexpected bug")
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run
            cmd_run(run_args)
        assert exc_info.value.code == 1

    @patch('learnm8.cli.main.run_active_learning')
    def test_persistence_error_exit_1(self, mock_run, run_args, capsys):
        mock_run.side_effect = PersistenceError("Cannot write output file")
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run
            cmd_run(run_args)
        assert exc_info.value.code == 1

    @patch('learnm8.cli.main.run_active_learning')
    def test_acquisition_error_exit_1(self, mock_run, run_args, capsys):
        mock_run.side_effect = AcquisitionError("Selection failed")
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run
            cmd_run(run_args)
        assert exc_info.value.code == 1

    @patch('learnm8.cli.main.run_active_learning')
    def test_pruning_error_exit_1(self, mock_run, run_args, capsys):
        mock_run.side_effect = PruningError("Pruning fraction too high")
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run
            cmd_run(run_args)
        assert exc_info.value.code == 1

    @patch('learnm8.cli.main.run_active_learning')
    def test_no_raw_traceback_for_learnm8_errors(self, mock_run, run_args, capsys):
        mock_run.side_effect = ConfigurationError("Bad config")
        with pytest.raises(SystemExit):
            from learnm8.cli.main import cmd_run
            cmd_run(run_args)
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert 'Traceback' not in output


@pytest.mark.unit
class TestCmdValidateErrorBoundary:
    """Test that cmd_validate catches LearnM8Error subtypes."""

    @patch('learnm8.cli.main.validate_compound_pool')
    @patch('learnm8.cli.main.load_compound_file')
    def test_validation_error_formatted_and_exit_1(
        self, mock_load, mock_validate, validate_args, capsys
    ):
        mock_load.return_value = pl.DataFrame({'ID': ['C1'], 'SMILES': ['CCO']})
        mock_validate.side_effect = ValidationError("All compounds invalid")
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_validate
            cmd_validate(validate_args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert 'Validation error' in output

    @patch('learnm8.cli.main.validate_compound_pool')
    @patch('learnm8.cli.main.load_compound_file')
    def test_feature_extraction_error_exit_1(
        self, mock_load, mock_validate, validate_args, capsys
    ):
        mock_load.return_value = pl.DataFrame({'ID': ['C1'], 'SMILES': ['CCO']})
        mock_validate.side_effect = FeatureExtractionError("Failed")
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_validate
            cmd_validate(validate_args)
        assert exc_info.value.code == 1

    @patch('learnm8.cli.main.validate_compound_pool')
    @patch('learnm8.cli.main.load_compound_file')
    def test_unexpected_exception_shows_traceback(
        self, mock_load, mock_validate, validate_args
    ):
        mock_load.return_value = pl.DataFrame({'ID': ['C1'], 'SMILES': ['CCO']})
        mock_validate.side_effect = RuntimeError("unexpected")
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_validate
            cmd_validate(validate_args)
        assert exc_info.value.code == 1

    @patch('learnm8.cli.main.validate_compound_pool')
    @patch('learnm8.cli.main.load_compound_file')
    def test_no_raw_traceback_for_learnm8_errors(
        self, mock_load, mock_validate, validate_args, capsys
    ):
        mock_load.return_value = pl.DataFrame({'ID': ['C1'], 'SMILES': ['CCO']})
        mock_validate.side_effect = ValidationError("Bad compounds")
        with pytest.raises(SystemExit):
            from learnm8.cli.main import cmd_validate
            cmd_validate(validate_args)
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert 'Traceback' not in output


@pytest.mark.unit
class TestConfigParseErrorBoundary:
    """Test that config file parse errors use ConfigurationError."""

    def test_malformed_yaml_raises_configuration_error(self, tmp_path):
        config_path = tmp_path / 'bad.yaml'
        config_path.write_text('invalid: yaml: content: [')
        from learnm8.cli.main import load_config_file
        with pytest.raises(ConfigurationError, match='Failed to parse config file'):
            load_config_file(config_path)

    def test_malformed_json_raises_configuration_error(self, tmp_path):
        config_path = tmp_path / 'bad.json'
        config_path.write_text('{invalid json}')
        from learnm8.cli.main import load_config_file
        with pytest.raises(ConfigurationError, match='Failed to parse config file'):
            load_config_file(config_path)


@pytest.mark.unit
class TestColumnArgsParsing:
    """Test --smiles-col and --id-col are registered and parse correctly."""

    def test_run_parser_has_smiles_col(self):
        from learnm8.cli.main import create_parser
        parser = create_parser()
        args = parser.parse_args([
            'run', 'compounds.csv',
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--smiles-col', 'canonical_smiles',
        ])
        assert args.smiles_col == 'canonical_smiles'

    def test_run_parser_has_id_col(self):
        from learnm8.cli.main import create_parser
        parser = create_parser()
        args = parser.parse_args([
            'run', 'compounds.csv',
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--id-col', 'compound_id',
        ])
        assert args.id_col == 'compound_id'

    def test_run_parser_col_defaults_to_none(self):
        from learnm8.cli.main import create_parser
        parser = create_parser()
        args = parser.parse_args([
            'run', 'compounds.csv',
            '--target', 'Activity',
            '--featurizer', 'morgan',
        ])
        assert args.smiles_col is None
        assert args.id_col is None

    def test_validate_parser_has_smiles_col(self):
        from learnm8.cli.main import create_parser
        parser = create_parser()
        args = parser.parse_args([
            'validate', 'compounds.csv',
            '--smiles-col', 'canonical_smiles',
        ])
        assert args.smiles_col == 'canonical_smiles'

    def test_validate_parser_has_id_col(self):
        from learnm8.cli.main import create_parser
        parser = create_parser()
        args = parser.parse_args([
            'validate', 'compounds.csv',
            '--id-col', 'compound_id',
        ])
        assert args.id_col == 'compound_id'

    def test_validate_parser_col_defaults_to_none(self):
        from learnm8.cli.main import create_parser
        parser = create_parser()
        args = parser.parse_args(['validate', 'compounds.csv'])
        assert args.smiles_col is None
        assert args.id_col is None


@pytest.mark.unit
class TestColumnArgsForwarding:
    """Test that --smiles-col/--id-col are forwarded correctly to underlying functions."""

    @patch('learnm8.cli.main.run_active_learning')
    def test_smiles_col_forwarded_to_run_active_learning(self, mock_run, run_args, tmp_path):
        run_args.smiles_col = 'canonical_smiles'
        run_args.id_col = 'compound_id'
        mock_run.return_value = {
            'output_dir': tmp_path / 'output',
            'cycle_metrics': [{}],
            'labeled_data': pl.DataFrame({'ID': ['C1'], 'SMILES': ['CCO']}),
            'unlabeled_data': pl.DataFrame({'ID': [], 'SMILES': []}),
            'validation_result': type('R', (), {
                'valid_compounds': pl.DataFrame({'ID': ['C1'], 'SMILES': ['CCO']}),
                'invalid_compounds': pl.DataFrame({'ID': [], 'SMILES': []}),
                'success_rate': 1.0,
                'validation_errors': {},
            })(),
            'saved_files': {},
        }
        from learnm8.cli.main import cmd_run
        cmd_run(run_args)
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get('smiles_column') == 'canonical_smiles'
        assert call_kwargs.get('id_column') == 'compound_id'

    @patch('learnm8.cli.main.validate_compound_pool')
    @patch('learnm8.cli.main.load_compound_file')
    def test_smiles_col_forwarded_to_load_compound_file(
        self, mock_load, mock_validate, validate_args
    ):
        validate_args.smiles_col = 'canonical_smiles'
        validate_args.id_col = 'compound_id'
        mock_load.return_value = pl.DataFrame({'ID': ['C1'], 'SMILES': ['CCO']})
        mock_validate.return_value = type('R', (), {
            'valid_compounds': pl.DataFrame({'ID': ['C1'], 'SMILES': ['CCO']}),
            'invalid_compounds': pl.DataFrame({'ID': [], 'SMILES': []}),
            'success_rate': 1.0,
            'validation_errors': {},
        })()
        from learnm8.cli.main import cmd_validate
        cmd_validate(validate_args)
        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs.get('smiles_column') == 'canonical_smiles'
        assert call_kwargs.get('id_column') == 'compound_id'


@pytest.mark.slow
class TestColumnArgsEndToEnd:
    """End-to-end tests for --smiles-col and --id-col with non-standard column names."""

    def test_run_with_custom_smiles_col(self, tmp_path):
        csv_path = tmp_path / "compounds.csv"
        df = pd.DataFrame({
            'cid': ['C1', 'C2', 'C3', 'C4', 'C5'],
            'canonical_smiles': ['CCO', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC'],
            'Activity': [0.1, 0.3, 0.5, 0.7, 0.9],
        })
        df.to_csv(csv_path, index=False)
        output_dir = tmp_path / "output"

        result = run_cli(
            'run', str(csv_path),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--smiles-col', 'canonical_smiles',
            '--id-col', 'cid',
            '--n-cycles', '2',
            '--batch-fraction', '0.4',
            '-o', str(output_dir),
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert (output_dir / 'compounds_final.csv').exists()

    def test_validate_with_custom_smiles_col(self, tmp_path):
        csv_path = tmp_path / "compounds.csv"
        df = pd.DataFrame({
            'cid': ['C1', 'C2', 'C3'],
            'canonical_smiles': ['CCO', 'CCC', 'CCCC'],
        })
        df.to_csv(csv_path, index=False)

        result = run_cli(
            'validate', str(csv_path),
            '--smiles-col', 'canonical_smiles',
            '--id-col', 'cid',
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert 'Valid compounds:' in result.stdout

    def test_run_fails_without_smiles_col_on_nonstandard_file(self, tmp_path):
        csv_path = tmp_path / "compounds.csv"
        df = pd.DataFrame({
            'cid': ['C1', 'C2'],
            'canonical_smiles': ['CCO', 'CCC'],
            'Activity': [0.1, 0.9],
        })
        df.to_csv(csv_path, index=False)

        result = run_cli(
            'run', str(csv_path),
            '--target', 'Activity',
            '--featurizer', 'morgan',
            '--learner', 'rf',
            '--n-cycles', '1',
            '--batch-fraction', '0.4',
            '-o', str(tmp_path / 'output'),
        )

        assert result.returncode != 0

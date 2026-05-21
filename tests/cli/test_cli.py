"""Comprehensive CLI tests for LearnM8.

Tests all subcommands (run, list, validate) and major CLI options.
Uses in-process invocation where possible, subprocess for select integration tests.
"""

import argparse
import io
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class CLIResult:
    returncode: int
    stdout: str
    stderr: str


def _get_cli_module():
    """Get the actual learnm8.cli.main module object (not the main function)."""
    import importlib

    return importlib.import_module('learnm8.cli.main')


def run_cmd_inprocess(cmd_func, args, monkeypatch_obj):
    """Run a CLI command in-process, capturing Rich console output."""
    from rich.console import Console

    cli_module = _get_cli_module()
    buf = io.StringIO()
    err_buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False, stderr=False)
    monkeypatch_obj.setattr(cli_module, 'console', test_console)

    try:
        cmd_func(args)
        return CLIResult(returncode=0, stdout=buf.getvalue(), stderr='')
    except SystemExit as e:
        return CLIResult(
            returncode=e.code or 0, stdout=buf.getvalue(), stderr=err_buf.getvalue()
        )


def make_run_namespace(compound_pool, tmp_path, **overrides):
    """Build argparse.Namespace with sensible defaults for cmd_run."""
    defaults = dict(
        compound_pool=compound_pool
        if isinstance(compound_pool, Path)
        else Path(str(compound_pool)),
        oracle=None,
        target_col='Activity',
        featurizer='morgan',
        learner='rf',
        score_direction='higher',
        cycles=None,
        config=None,
        n_cycles=2,
        batch_fraction=0.4,
        strategy='greedy',
        initial_strategy='random',
        pruning_fraction=None,
        pruning_strategy=None,
        acquisition_params=None,
        output=str(tmp_path / 'output'),
        cache_dir=str(tmp_path / 'cache'),
        quiet=False,
        n_jobs=-1,
        device='cpu',
        random_state=42,
        memory_safety_factor=0.7,
        smiles_col=None,
        id_col=None,
        force_uncertainty=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_validate_namespace(compound_pool, **overrides):
    """Build argparse.Namespace for cmd_validate."""
    defaults = dict(
        compound_pool=compound_pool
        if isinstance(compound_pool, Path)
        else Path(str(compound_pool)),
        n_jobs=-1,
        output=None,
        smiles_col=None,
        id_col=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def minimal_compounds(tmp_path):
    """Create valid compound pool CSV."""
    csv_path = tmp_path / 'compounds.csv'
    base_smiles = [
        'CCO',
        'CCC',
        'CCCC',
        'CCCCC',
        'CCCCCC',
        'c1ccccc1',
        'CC(=O)O',
        'CCN',
        'CC(O)C',
        'CCOC',
        'C1CCCCC1',
        'CC=CC',
        'CC#N',
        'CC(=O)N',
        'CCCO',
        'c1ccc(O)cc1',
        'CC(C)O',
        'CCOCC',
        'CCNC',
        'CC(=O)OC',
    ]
    n = 20
    smiles = base_smiles[:n]
    df = pd.DataFrame(
        {
            'ID': [f'COMP_{i:04d}' for i in range(1, n + 1)],
            'SMILES': smiles,
            'Activity': [round(i * 0.005, 4) for i in range(1, n + 1)],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def oracle_csv(tmp_path, minimal_compounds):
    """Create oracle CSV file matching the minimal_compounds fixture."""
    csv_path = tmp_path / 'oracle.csv'
    df = pd.read_csv(minimal_compounds)
    df['Activity'] = [round(0.15 + 0.004 * i, 4) for i in range(len(df))]
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def config_yaml(tmp_path):
    """Create YAML config file."""
    config_path = tmp_path / 'config.yaml'
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
def mock_run_success(monkeypatch):
    """Patch learnm8.cli.main.run_active_learning to skip the pipeline.

    Writes stub compounds_final.csv and cycle_metrics.csv into ``output_dir``
    so tests asserting on output-file existence still pass, while skipping
    the multi-second model train + featurize cost. Tests of pipeline
    behavior live elsewhere; CLI tests should only verify the CLI layer.
    """

    def _stub(**kwargs):
        output_dir = Path(kwargs.get('output_dir', '.'))
        output_dir.mkdir(parents=True, exist_ok=True)
        n_cycles = kwargs.get('n_cycles')
        cycles_spec = kwargs.get('cycles')
        if cycles_spec:
            n_cycles = 0
            for token in str(cycles_spec).split():
                _, rest = token.split(':', 1)
                _, _, mult = rest.partition('*')
                n_cycles += int(mult) if mult else 1
        n_cycles = int(n_cycles or 1)
        (output_dir / 'compounds_final.csv').write_text(
            'ID,SMILES,Activity,status\nCOMP_0001,CCO,0.5,labeled\n'
        )
        (output_dir / 'cycle_metrics.csv').write_text(
            'cycle,strategy,selected_count\n'
            + ''.join(f'{i},random,1\n' for i in range(n_cycles))
        )

        class _StubValidation:
            valid_compounds = pl.DataFrame({'ID': ['COMP_0001'], 'SMILES': ['CCO']})
            invalid_compounds = pl.DataFrame({'ID': [], 'SMILES': []})
            success_rate = 1.0

        return {
            'compounds_df': pl.DataFrame(
                {
                    'ID': ['COMP_0001'],
                    'SMILES': ['CCO'],
                    'Activity': [0.5],
                }
            ),
            'cycle_metrics': [
                {'cycle': i, 'strategy': 'random', 'selected_count': 1}
                for i in range(n_cycles)
            ],
            'output_dir': output_dir,
            'labeled_count': n_cycles,
            'unlabeled_count': 0,
            'saved_files': {
                'compounds_final': output_dir / 'compounds_final.csv',
                'cycle_metrics': output_dir / 'cycle_metrics.csv',
            },
            'validation_result': _StubValidation(),
        }

    cli_module = _get_cli_module()
    monkeypatch.setattr(cli_module, 'run_active_learning', _stub)
    return _stub


@pytest.fixture
def config_json(tmp_path):
    """Create JSON config file."""
    config_path = tmp_path / 'config.json'
    config = {
        'target_col': 'Activity',
        'featurizer': 'morgan',
        'learner': 'rf',
        'n_cycles': 3,
        'batch_fraction': 0.2,
        'strategy': 'greedy',
        'random_state': 42,
    }
    config_path.write_text(json.dumps(config))
    return config_path


def run_cli(*args, timeout=120):
    """Helper function to run CLI command with optional timeout."""
    cmd = [sys.executable, '-m', 'learnm8.cli.main', *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result


@pytest.mark.slow
class TestRunSubcommand:
    """Test 'run' subcommand functionality."""

    def test_minimal_invocation(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / 'output'

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target',
            'Activity',
            '--featurizer',
            'morgan',
            '--learner',
            'rf',
            '--n-cycles',
            '2',
            '--batch-fraction',
            '0.4',
            '-o',
            str(output_dir),
        )

        assert result.returncode == 0, f'CLI failed: {result.stderr}'
        assert output_dir.exists()
        assert (output_dir / 'compounds_final.csv').exists()
        assert (output_dir / 'cycle_metrics.csv').exists()

    def test_with_explicit_oracle(
        self, minimal_compounds, oracle_csv, tmp_path, monkeypatch, mock_run_success
    ):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            oracle=str(oracle_csv),
            n_cycles=2,
            batch_fraction=0.4,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0, f'Failed: {result.stdout}'
        output_dir = Path(args.output)
        assert output_dir.exists()

    def test_cycles_spec_parsing(self, minimal_compounds, tmp_path, monkeypatch):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            cycles='random:0.2 greedy:0.2*2',
            n_cycles=None,
            batch_fraction=None,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0, f'Failed: {result.stdout}'
        output_dir = Path(args.output)
        assert output_dir.exists()

        metrics = pd.read_csv(output_dir / 'cycle_metrics.csv', comment='#')
        assert len(metrics) == 3

    def test_output_dir_creation(
        self, minimal_compounds, tmp_path, monkeypatch, mock_run_success
    ):
        from learnm8.cli.main import cmd_run

        output_dir = tmp_path / 'nested' / 'output' / 'dir'
        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            output=str(output_dir),
            n_cycles=1,
            batch_fraction=0.4,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0, f'Failed: {result.stdout}'
        assert output_dir.exists()
        assert (output_dir / 'compounds_final.csv').exists()

    def test_pruning_flags(
        self, minimal_compounds, tmp_path, monkeypatch, mock_run_success
    ):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            n_cycles=2,
            batch_fraction=0.4,
            pruning_fraction=0.3,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0, f'Failed: {result.stdout}'

    @pytest.mark.parametrize('learner', ['rf', 'gp', 'xgb'])
    def test_learner_selection(
        self, minimal_compounds, tmp_path, monkeypatch, mock_run_success, learner
    ):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            learner=learner,
            output=str(tmp_path / 'output' / learner),
            n_cycles=1,
            batch_fraction=0.4,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0, f'Learner {learner} failed: {result.stdout}'

    def test_acquisition_selection(
        self, minimal_compounds, tmp_path, monkeypatch, mock_run_success
    ):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            learner='gp',
            strategy='ucb',
            n_cycles=2,
            batch_fraction=0.4,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0, f'Failed: {result.stdout}'

    def test_config_yaml(
        self, minimal_compounds, config_yaml, tmp_path, monkeypatch, mock_run_success
    ):
        pytest.importorskip('yaml', reason='PyYAML not installed')
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            config=config_yaml,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0, f'Failed: {result.stdout}'
        output_dir = Path(args.output)
        assert output_dir.exists()

    def test_config_json(
        self, minimal_compounds, config_json, tmp_path, monkeypatch, mock_run_success
    ):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            config=config_json,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0, f'Failed: {result.stdout}'
        output_dir = Path(args.output)
        assert output_dir.exists()

    def test_missing_required_args(self, minimal_compounds):
        from learnm8.cli.main import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(
                [
                    'run',
                    str(minimal_compounds),
                    '--target',
                    'Activity',
                    '--learner',
                    'rf',
                ]
            )
        assert exc_info.value.code != 0

    def test_invalid_file(self, tmp_path, monkeypatch):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            tmp_path / 'nonexistent.csv',
            tmp_path,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode != 0
        assert 'not found' in result.stdout.lower() or 'error' in result.stdout.lower()

    def test_invalid_learner(self, minimal_compounds, tmp_path, monkeypatch):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            learner='invalid_learner',
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode != 0

    @pytest.mark.parametrize('featurizer', ['morgan', 'maccs', 'ecfp6'])
    def test_featurizer_options(
        self, minimal_compounds, tmp_path, monkeypatch, mock_run_success, featurizer
    ):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            featurizer=featurizer,
            output=str(tmp_path / featurizer),
            n_cycles=1,
            batch_fraction=0.4,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0, (
            f'Featurizer {featurizer} failed: {result.stdout}'
        )

    @pytest.mark.parametrize('direction', ['higher', 'lower'])
    def test_score_direction(
        self, minimal_compounds, tmp_path, monkeypatch, mock_run_success, direction
    ):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            score_direction=direction,
            output=str(tmp_path / direction),
            n_cycles=1,
            batch_fraction=0.4,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0, f'Direction {direction} failed'

    def test_random_state(self, minimal_compounds, tmp_path, monkeypatch):
        from learnm8.cli.main import cmd_run

        output_dir1 = tmp_path / 'run1'
        output_dir2 = tmp_path / 'run2'

        args1 = make_run_namespace(
            minimal_compounds,
            tmp_path,
            output=str(output_dir1),
            random_state=42,
            n_cycles=2,
            batch_fraction=0.4,
        )
        result1 = run_cmd_inprocess(cmd_run, args1, monkeypatch)

        args2 = make_run_namespace(
            minimal_compounds,
            tmp_path,
            output=str(output_dir2),
            random_state=42,
            n_cycles=2,
            batch_fraction=0.4,
        )
        result2 = run_cmd_inprocess(cmd_run, args2, monkeypatch)

        assert result1.returncode == 0
        assert result2.returncode == 0

        df1 = pd.read_csv(output_dir1 / 'compounds_final.csv', comment='#')
        df2 = pd.read_csv(output_dir2 / 'compounds_final.csv', comment='#')

        pd.testing.assert_frame_equal(
            df1.sort_values('ID').reset_index(drop=True),
            df2.sort_values('ID').reset_index(drop=True),
        )

    def test_quiet_flag(
        self, minimal_compounds, tmp_path, monkeypatch, mock_run_success
    ):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            quiet=True,
            n_cycles=1,
            batch_fraction=0.4,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0

    def test_force_uncertainty_flag_forwarding(
        self, minimal_compounds, tmp_path, monkeypatch
    ):
        """FR-014: --force-uncertainty must be forwarded to run_active_learning()."""
        import learnm8.cli.main as cli_main_mod
        from learnm8.cli.main import cmd_run

        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return {
                'compounds_df': None,
                'cycle_metrics': [],
                'aggregate_metrics': {},
                'validation_result': None,
                'output_dir': kwargs.get('output_dir', tmp_path),
                'saved_files': {},
                'prediction_files': [],
                'labeled_count': 0,
                'unlabeled_count': 0,
            }

        monkeypatch.setattr(cli_main_mod, 'run_active_learning', fake_run)
        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            n_cycles=1,
            batch_fraction=0.4,
            force_uncertainty=True,
        )
        run_cmd_inprocess(cmd_run, args, monkeypatch)

        assert captured.get('force_uncertainty') is True, (
            'cmd_run did not forward --force-uncertainty to '
            'run_active_learning() (FR-014).'
        )

    def test_force_uncertainty_default_false(
        self, minimal_compounds, tmp_path, monkeypatch
    ):
        """Default for --force-uncertainty must be False (FR-013)."""
        import learnm8.cli.main as cli_main_mod
        from learnm8.cli.main import cmd_run

        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return {
                'compounds_df': None,
                'cycle_metrics': [],
                'aggregate_metrics': {},
                'validation_result': None,
                'output_dir': kwargs.get('output_dir', tmp_path),
                'saved_files': {},
                'prediction_files': [],
                'labeled_count': 0,
                'unlabeled_count': 0,
            }

        monkeypatch.setattr(cli_main_mod, 'run_active_learning', fake_run)
        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            n_cycles=1,
            batch_fraction=0.4,
        )
        run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert captured.get('force_uncertainty') is False


@pytest.mark.slow
class TestValidateSubcommand:
    """Test 'validate' subcommand functionality."""

    def test_validate_valid_compounds(self, minimal_compounds, tmp_path, monkeypatch):
        from learnm8.cli.main import cmd_validate

        args = make_validate_namespace(
            minimal_compounds,
            output=str(tmp_path / 'validation'),
        )
        result = run_cmd_inprocess(cmd_validate, args, monkeypatch)
        assert result.returncode == 0
        assert 'Valid compounds:' in result.stdout

    def test_validate_with_output(self, minimal_compounds, tmp_path, monkeypatch):
        from learnm8.cli.main import cmd_validate

        output_dir = tmp_path / 'validation'
        args = make_validate_namespace(
            minimal_compounds,
            output=str(output_dir),
        )
        result = run_cmd_inprocess(cmd_validate, args, monkeypatch)
        assert result.returncode == 0
        assert output_dir.exists()
        assert (output_dir / 'validation_report.csv').exists()

    def test_validate_invalid_file(self, tmp_path, monkeypatch):
        from learnm8.cli.main import cmd_validate

        args = make_validate_namespace(tmp_path / 'nonexistent.csv')
        result = run_cmd_inprocess(cmd_validate, args, monkeypatch)
        assert result.returncode != 0


@pytest.mark.unit
class TestHelpAndErrors:
    """Test help messages and error handling — pure parser, no pipeline."""

    def test_no_args_shows_help(self):
        from learnm8.cli.main import create_parser

        parser = create_parser()
        buf = io.StringIO()
        parser.print_help(buf)
        output = buf.getvalue()
        assert 'learnm8' in output.lower() or 'usage' in output.lower()

    def test_run_help(self):
        from learnm8.cli.main import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['run', '-h'])
        assert exc_info.value.code == 0

    def test_validate_help(self):
        from learnm8.cli.main import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['validate', '-h'])
        assert exc_info.value.code == 0


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error scenarios — most fail fast before pipeline runs."""

    def test_empty_compound_pool(self, tmp_path, monkeypatch):
        from learnm8.cli.main import cmd_run

        csv_path = tmp_path / 'empty.csv'
        df = pd.DataFrame(columns=['ID', 'SMILES', 'Activity'])
        df.to_csv(csv_path, index=False)

        args = make_run_namespace(csv_path, tmp_path)
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode != 0

    def test_missing_required_columns(self, tmp_path, monkeypatch):
        from learnm8.cli.main import cmd_run

        csv_path = tmp_path / 'bad_columns.csv'
        df = pd.DataFrame({'compound_id': ['C1', 'C2'], 'structure': ['CCO', 'CCC']})
        df.to_csv(csv_path, index=False)

        args = make_run_namespace(csv_path, tmp_path)
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode != 0

    def test_invalid_cycles_spec(self, minimal_compounds, tmp_path, monkeypatch):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            cycles='invalid_format',
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode != 0

    @pytest.mark.slow
    def test_invalid_pruning_fraction(self, minimal_compounds, tmp_path, monkeypatch):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            pruning_fraction=1.5,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode != 0

    @pytest.mark.slow
    def test_acquisition_params_json(self, minimal_compounds, tmp_path, monkeypatch):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            learner='gp',
            strategy='ucb',
            acquisition_params='{"beta": 2.0}',
            n_cycles=1,
            batch_fraction=0.4,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0

    def test_invalid_acquisition_params_json(
        self, minimal_compounds, tmp_path, monkeypatch
    ):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            acquisition_params='{invalid json}',
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode != 0


@pytest.mark.slow
class TestIntegration:
    """Integration tests across CLI features."""

    def test_full_workflow_with_all_options(self, minimal_compounds, tmp_path):
        output_dir = tmp_path / 'full_workflow'

        result = run_cli(
            'run',
            str(minimal_compounds),
            '--target',
            'Activity',
            '--featurizer',
            'morgan',
            '--learner',
            'gp',
            '--cycles',
            'random:0.2 ucb:0.2*2',
            '--score-direction',
            'higher',
            '--pruning-fraction',
            '0.2',
            '--random-state',
            '123',
            '-o',
            str(output_dir),
        )

        assert result.returncode == 0
        assert (output_dir / 'compounds_final.csv').exists()
        assert (output_dir / 'cycle_metrics.csv').exists()
        assert (output_dir / 'selection_history.csv').exists()

        final_df = pd.read_csv(output_dir / 'compounds_final.csv', comment='#')
        assert 'ID' in final_df.columns
        assert 'SMILES' in final_df.columns
        assert 'status' in final_df.columns

    def test_benchmark_mode_auto_detection(
        self, minimal_compounds, tmp_path, monkeypatch
    ):
        from learnm8.cli.main import cmd_run

        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            output=str(tmp_path / 'benchmark'),
            n_cycles=1,
            batch_fraction=0.4,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0

    @pytest.mark.xfail(
        reason='Config loading overwrites CLI args instead of providing defaults'
    )
    def test_cli_flags_override_config_file(
        self, minimal_compounds, tmp_path, monkeypatch
    ):
        from learnm8.cli.main import cmd_run

        config_path = tmp_path / 'override_config.yaml'
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

        output_dir = tmp_path / 'override'
        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            config=str(config_path),
            learner='gp',
            n_cycles=2,
            output=str(output_dir),
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
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
        assert (
            'Configuration error' in captured.out
            or 'Configuration error' in captured.err
        )

    @patch('learnm8.cli.main.run_active_learning')
    def test_validation_error_formatted_and_exit_1(self, mock_run, run_args, capsys):
        mock_run.side_effect = ValidationError(
            '50 compounds have invalid SMILES',
            invalid_indices=[0, 1, 2],
        )
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run

            cmd_run(run_args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'Validation error' in captured.out or 'Validation error' in captured.err

    @patch('learnm8.cli.main.run_active_learning')
    def test_validation_error_suggests_validate_command(
        self, mock_run, run_args, capsys
    ):
        mock_run.side_effect = ValidationError('Bad SMILES')
        with pytest.raises(SystemExit):
            from learnm8.cli.main import cmd_run

            cmd_run(run_args)
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert 'learnm8 validate' in output

    @patch('learnm8.cli.main.run_active_learning')
    def test_learner_error_formatted_and_exit_1(self, mock_run, run_args, capsys):
        mock_run.side_effect = LearnerError('Training failed: insufficient data')
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run

            cmd_run(run_args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert 'Training/prediction error' in output

    @patch('learnm8.cli.main.run_active_learning')
    def test_feature_extraction_error_formatted_and_exit_1(
        self, mock_run, run_args, capsys
    ):
        mock_run.side_effect = FeatureExtractionError('Morgan featurizer failed')
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run

            cmd_run(run_args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert 'Feature extraction error' in output

    @patch('learnm8.cli.main.run_active_learning')
    def test_feature_extraction_error_suggests_validate(
        self, mock_run, run_args, capsys
    ):
        mock_run.side_effect = FeatureExtractionError('Failed')
        with pytest.raises(SystemExit):
            from learnm8.cli.main import cmd_run

            cmd_run(run_args)
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert 'learnm8 validate' in output

    @patch('learnm8.cli.main.run_active_learning')
    def test_generic_learnm8_error_formatted_and_exit_1(
        self, mock_run, run_args, capsys
    ):
        mock_run.side_effect = OracleError('Oracle measurement failed')
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
        mock_run.side_effect = AttributeError('unexpected bug')
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run

            cmd_run(run_args)
        assert exc_info.value.code == 1

    @patch('learnm8.cli.main.run_active_learning')
    def test_persistence_error_exit_1(self, mock_run, run_args, capsys):
        mock_run.side_effect = PersistenceError('Cannot write output file')
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run

            cmd_run(run_args)
        assert exc_info.value.code == 1

    @patch('learnm8.cli.main.run_active_learning')
    def test_acquisition_error_exit_1(self, mock_run, run_args, capsys):
        mock_run.side_effect = AcquisitionError('Selection failed')
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run

            cmd_run(run_args)
        assert exc_info.value.code == 1

    @patch('learnm8.cli.main.run_active_learning')
    def test_pruning_error_exit_1(self, mock_run, run_args, capsys):
        mock_run.side_effect = PruningError('Pruning fraction too high')
        with pytest.raises(SystemExit) as exc_info:
            from learnm8.cli.main import cmd_run

            cmd_run(run_args)
        assert exc_info.value.code == 1

    @patch('learnm8.cli.main.run_active_learning')
    def test_no_raw_traceback_for_learnm8_errors(self, mock_run, run_args, capsys):
        mock_run.side_effect = ConfigurationError('Bad config')
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
        mock_validate.side_effect = ValidationError('All compounds invalid')
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
        mock_validate.side_effect = FeatureExtractionError('Failed')
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
        mock_validate.side_effect = RuntimeError('unexpected')
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
        mock_validate.side_effect = ValidationError('Bad compounds')
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
class TestConfigOnlyRun:
    """BUG 7b: `learnm8 run --config experiment.yaml` must be possible without
    --target/--featurizer on the command line (they may live in the config)."""

    def test_config_only_run_parses_without_target_or_featurizer(self, tmp_path):
        from learnm8.cli.main import create_parser

        config_path = tmp_path / 'experiment.yaml'
        config_path.write_text(
            'target_col: Activity\nfeaturizer: morgan\nlearner: rf\n'
        )
        parser = create_parser()
        args = parser.parse_args(['run', '--config', str(config_path)])
        assert args.config == config_path

    def test_cmd_run_errors_when_target_missing_everywhere(
        self, minimal_compounds, tmp_path, monkeypatch
    ):
        from learnm8.cli.main import cmd_run

        config_path = tmp_path / 'no_target.yaml'
        config_path.write_text('featurizer: morgan\nlearner: rf\n')
        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            config=config_path,
            target_col=None,
            featurizer=None,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode != 0
        assert 'target' in result.stdout.lower()

    def test_cmd_run_errors_when_featurizer_missing_everywhere(
        self, minimal_compounds, tmp_path, monkeypatch
    ):
        from learnm8.cli.main import cmd_run

        config_path = tmp_path / 'no_featurizer.yaml'
        config_path.write_text('target_col: Activity\nlearner: rf\n')
        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            config=config_path,
            target_col=None,
            featurizer=None,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode != 0
        assert 'featurizer' in result.stdout.lower()

    def test_config_only_run_succeeds_with_target_and_featurizer_in_config(
        self, minimal_compounds, tmp_path, monkeypatch, mock_run_success
    ):
        from learnm8.cli.main import cmd_run

        config_path = tmp_path / 'full.yaml'
        config_path.write_text(
            f'compound_pool: {minimal_compounds}\n'
            'target_col: Activity\nfeaturizer: morgan\nlearner: rf\n'
            'n_cycles: 1\n'
        )
        args = make_run_namespace(
            minimal_compounds,
            tmp_path,
            config=config_path,
            target_col=None,
            featurizer=None,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0, f'Failed: {result.stdout}'


@pytest.mark.unit
class TestColumnArgsParsing:
    """Test --smiles-col and --id-col are registered and parse correctly."""

    def test_run_parser_has_smiles_col(self):
        from learnm8.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                'run',
                'compounds.csv',
                '--target',
                'Activity',
                '--featurizer',
                'morgan',
                '--smiles-col',
                'canonical_smiles',
            ]
        )
        assert args.smiles_col == 'canonical_smiles'

    def test_run_parser_has_id_col(self):
        from learnm8.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                'run',
                'compounds.csv',
                '--target',
                'Activity',
                '--featurizer',
                'morgan',
                '--id-col',
                'compound_id',
            ]
        )
        assert args.id_col == 'compound_id'

    def test_run_parser_col_defaults_to_none(self):
        from learnm8.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                'run',
                'compounds.csv',
                '--target',
                'Activity',
                '--featurizer',
                'morgan',
            ]
        )
        assert args.smiles_col is None
        assert args.id_col is None

    def test_validate_parser_has_smiles_col(self):
        from learnm8.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                'validate',
                'compounds.csv',
                '--smiles-col',
                'canonical_smiles',
            ]
        )
        assert args.smiles_col == 'canonical_smiles'

    def test_validate_parser_has_id_col(self):
        from learnm8.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                'validate',
                'compounds.csv',
                '--id-col',
                'compound_id',
            ]
        )
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
    def test_smiles_col_forwarded_to_run_active_learning(
        self, mock_run, run_args, tmp_path
    ):
        run_args.smiles_col = 'canonical_smiles'
        run_args.id_col = 'compound_id'
        mock_run.return_value = {
            'output_dir': tmp_path / 'output',
            'cycle_metrics': [{}],
            'labeled_count': 1,
            'unlabeled_count': 0,
            'prediction_files': [],
            'validation_result': type(
                'R',
                (),
                {
                    'valid_compounds': pl.DataFrame({'ID': ['C1'], 'SMILES': ['CCO']}),
                    'invalid_compounds': pl.DataFrame({'ID': [], 'SMILES': []}),
                    'success_rate': 1.0,
                    'validation_errors': {},
                },
            )(),
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
        mock_validate.return_value = type(
            'R',
            (),
            {
                'valid_compounds': pl.DataFrame({'ID': ['C1'], 'SMILES': ['CCO']}),
                'invalid_compounds': pl.DataFrame({'ID': [], 'SMILES': []}),
                'success_rate': 1.0,
                'validation_errors': {},
            },
        )()
        from learnm8.cli.main import cmd_validate

        cmd_validate(validate_args)
        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs.get('smiles_column') == 'canonical_smiles'
        assert call_kwargs.get('id_column') == 'compound_id'


@pytest.mark.slow
class TestColumnArgsEndToEnd:
    """End-to-end tests for --smiles-col and --id-col with non-standard column names."""

    def test_run_with_custom_smiles_col(self, tmp_path, monkeypatch):
        from learnm8.cli.main import cmd_run

        csv_path = tmp_path / 'compounds.csv'
        df = pd.DataFrame(
            {
                'cid': ['C1', 'C2', 'C3', 'C4', 'C5'],
                'canonical_smiles': ['CCO', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC'],
                'Activity': [0.1, 0.3, 0.5, 0.7, 0.9],
            }
        )
        df.to_csv(csv_path, index=False)

        args = make_run_namespace(
            csv_path,
            tmp_path,
            smiles_col='canonical_smiles',
            id_col='cid',
            n_cycles=2,
            batch_fraction=0.4,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode == 0, f'Failed: {result.stdout}'
        output_dir = Path(args.output)
        assert (output_dir / 'compounds_final.csv').exists()

    def test_validate_with_custom_smiles_col(self, tmp_path, monkeypatch):
        from learnm8.cli.main import cmd_validate

        csv_path = tmp_path / 'compounds.csv'
        df = pd.DataFrame(
            {
                'cid': ['C1', 'C2', 'C3'],
                'canonical_smiles': ['CCO', 'CCC', 'CCCC'],
            }
        )
        df.to_csv(csv_path, index=False)

        args = make_validate_namespace(
            csv_path,
            smiles_col='canonical_smiles',
            id_col='cid',
        )
        result = run_cmd_inprocess(cmd_validate, args, monkeypatch)
        assert result.returncode == 0, f'Failed: {result.stdout}'
        assert 'Valid compounds:' in result.stdout

    def test_run_fails_without_smiles_col_on_nonstandard_file(
        self, tmp_path, monkeypatch
    ):
        from learnm8.cli.main import cmd_run

        csv_path = tmp_path / 'compounds.csv'
        df = pd.DataFrame(
            {
                'cid': ['C1', 'C2'],
                'canonical_smiles': ['CCO', 'CCC'],
                'Activity': [0.1, 0.9],
            }
        )
        df.to_csv(csv_path, index=False)

        args = make_run_namespace(
            csv_path,
            tmp_path,
            n_cycles=1,
            batch_fraction=0.4,
        )
        result = run_cmd_inprocess(cmd_run, args, monkeypatch)
        assert result.returncode != 0

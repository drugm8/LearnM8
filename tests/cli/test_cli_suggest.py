"""Tests for the ``learnm8 suggest`` subcommand."""

import polars as pl
import pytest

from learnm8.cli.main import cmd_suggest, create_parser
from tests.cli.test_cli import run_cmd_inprocess

pytestmark = pytest.mark.unit


def _argv(tmp_path):
    return [
        'suggest',
        'tests/data/small_molecules.csv',
        str(tmp_path),
        '--target',
        'Activity',
        '--featurizer',
        'morgan',
        '--batch-size',
        '4',
        '--n-jobs',
        '1',
    ]


def test_suggest_cli_writes_batch(tmp_path, monkeypatch):
    args = create_parser().parse_args(_argv(tmp_path))
    r = run_cmd_inprocess(cmd_suggest, args, monkeypatch)
    assert r.returncode == 0 and 'batch_cycle_0.csv' in r.stdout
    assert pl.read_csv(tmp_path / 'batch_cycle_0.csv').height == 4


def test_suggest_cli_rerun_exits_2_configuration_error(tmp_path, monkeypatch):
    argv = _argv(tmp_path)
    run_cmd_inprocess(cmd_suggest, create_parser().parse_args(argv), monkeypatch)
    r = run_cmd_inprocess(cmd_suggest, create_parser().parse_args(argv), monkeypatch)
    assert r.returncode == 2 and 'no results' in r.stdout.lower()

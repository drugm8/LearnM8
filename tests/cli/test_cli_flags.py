"""CLI flags and config-file handling for `learnm8 run` (parsing layer only)."""

from __future__ import annotations

import importlib
import io
from pathlib import Path

import pytest
from rich.console import Console

import learnm8
from learnm8.cli.main import _cli_provided_dests, cmd_run, create_parser

cli = importlib.import_module('learnm8.cli.main')

pytestmark = [pytest.mark.unit]


@pytest.fixture
def pool(tmp_path) -> Path:
    path = tmp_path / 'pool.csv'
    path.write_text('ID,SMILES,V\nC0,CCO,1.0\nC1,CCC,2.0\n')
    return path


@pytest.fixture
def captured(monkeypatch):
    """Capture run_active_learning kwargs and console output of cmd_run."""
    calls: list[dict] = []
    buf = io.StringIO()

    def _stub(**kwargs):
        calls.append(kwargs)
        return {
            'output_dir': Path('.'),
            'cycle_metrics': [],
            'labeled_count': 0,
            'unlabeled_count': 0,
            'validation_result': type(
                'V',
                (),
                {
                    'valid_compounds': [],
                    'invalid_compounds': [],
                    'success_rate': 1.0,
                },
            )(),
            'saved_files': {},
        }

    monkeypatch.setattr(cli, 'run_active_learning', _stub)
    monkeypatch.setattr(cli, 'console', Console(file=buf, width=200))
    return calls, buf


def _run(argv: list[str]) -> None:
    args = create_parser().parse_args(argv)
    cmd_run(args, _cli_provided_dests(argv))


class TestFlags:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            create_parser().parse_args(['--version'])
        assert exc.value.code == 0
        assert learnm8.__version__ in capsys.readouterr().out

    def test_initial_strategy_flag_removed(self):
        with pytest.raises(SystemExit):
            create_parser().parse_args(['run', 'p.csv', '--initial-strategy', 'random'])

    def test_strategy_choices_from_registry(self):
        with pytest.raises(SystemExit):
            create_parser().parse_args(['run', 'p.csv', '--strategy', 'bogus'])
        args = create_parser().parse_args(['run', 'p.csv', '--strategy', 'ucb'])
        assert args.strategy == 'ucb'

    def test_pruning_strategy_choices(self):
        with pytest.raises(SystemExit):
            create_parser().parse_args(['run', 'p.csv', '--pruning-strategy', 'x'])

    def test_validate_epilog_has_no_featurizer(self):
        epilog = create_parser().epilog
        assert 'validate compounds.csv --featurizer' not in epilog

    def test_fine_tuning_and_similarity_defaults(self, pool, captured):
        calls, _ = captured
        _run(['run', str(pool), '--target', 'V', '--featurizer', 'morgan'])
        kw = calls[0]
        assert kw['enable_chemprop_fine_tuning'] is False
        assert kw['disable_molecular_similarity'] is False
        assert 'initial_strategy' not in kw

    def test_fine_tuning_flag(self, pool, captured):
        calls, _ = captured
        _run(
            [
                'run',
                str(pool),
                '--target',
                'V',
                '--learner',
                'chemprop',
                '--enable-chemprop-fine-tuning',
            ]
        )
        assert calls[0]['enable_chemprop_fine_tuning'] is True

    def test_disable_similarity_bare_flag_means_all(self, pool, captured):
        calls, _ = captured
        _run(
            [
                'run',
                str(pool),
                '--target',
                'V',
                '--featurizer',
                'morgan',
                '--disable-molecular-similarity',
            ]
        )
        assert calls[0]['disable_molecular_similarity'] is True

    def test_disable_similarity_with_names(self, pool, captured):
        calls, _ = captured
        _run(
            [
                'run',
                str(pool),
                '--target',
                'V',
                '--featurizer',
                'morgan',
                '--disable-molecular-similarity',
                'scaffold_diversity_index_batch',
            ]
        )
        assert calls[0]['disable_molecular_similarity'] == [
            'scaffold_diversity_index_batch'
        ]


class TestConfigFile:
    def _config(self, tmp_path, body: str) -> Path:
        path = tmp_path / 'cfg.yaml'
        path.write_text(body)
        return path

    def test_new_keys_from_config(self, pool, tmp_path, captured):
        calls, _ = captured
        cfg = self._config(
            tmp_path,
            'target_col: V\nlearner: chemprop\nenable_chemprop_fine_tuning: true\n'
            'disable_molecular_similarity: [scaffold_diversity_index_batch]\n',
        )
        _run(['run', str(pool), '--config', str(cfg)])
        assert calls[0]['enable_chemprop_fine_tuning'] is True
        assert calls[0]['disable_molecular_similarity'] == [
            'scaffold_diversity_index_batch'
        ]

    def test_similarity_true_from_config(self, pool, tmp_path, captured):
        calls, _ = captured
        cfg = self._config(
            tmp_path,
            'target_col: V\nfeaturizer: morgan\ndisable_molecular_similarity: true\n',
        )
        _run(['run', str(pool), '--config', str(cfg)])
        assert calls[0]['disable_molecular_similarity'] is True

    @pytest.mark.parametrize('key', ['lerner', 'initial_strategy'])
    def test_unknown_key_warns_and_is_not_applied(self, pool, tmp_path, captured, key):
        calls, buf = captured
        cfg = self._config(
            tmp_path, f'target_col: V\nfeaturizer: morgan\n{key}: whatever\n'
        )
        _run(['run', str(pool), '--config', str(cfg)])
        out = buf.getvalue()
        assert f'Config key "{key}" is not a known option' in out
        assert len(calls) == 1
        assert key not in calls[0]

    def test_aliases_do_not_warn(self, pool, tmp_path, captured):
        _, buf = captured
        cfg = self._config(
            tmp_path,
            'target_col: V\nfeaturizer: morgan\noutput_dir: out\nsmiles_column: SMILES\n'
            'pruning_params: {pruning_fraction: 0.1}\n',
        )
        _run(['run', str(pool), '--config', str(cfg)])
        assert 'not a known option' not in buf.getvalue()


class TestExitCodes:
    def test_missing_target_exits_2(self, pool, captured):
        with pytest.raises(SystemExit) as exc:
            _run(['run', str(pool), '--featurizer', 'morgan'])
        assert exc.value.code == 2

    def test_missing_featurizer_exits_2(self, pool, captured):
        with pytest.raises(SystemExit) as exc:
            _run(['run', str(pool), '--target', 'V'])
        assert exc.value.code == 2

    def test_configuration_error_exits_2_single_line(self, pool, captured, monkeypatch):
        from learnm8.exceptions import ConfigurationError

        def _boom(**kwargs):
            raise ConfigurationError('bad thing')

        monkeypatch.setattr(cli, 'run_active_learning', _boom)
        _, buf = captured
        with pytest.raises(SystemExit) as exc:
            _run(['run', str(pool), '--target', 'V', '--featurizer', 'morgan'])
        assert exc.value.code == 2
        assert 'Configuration error: bad thing' in buf.getvalue()
        assert 'Traceback' not in buf.getvalue()

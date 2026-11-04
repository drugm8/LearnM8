"""Entry-point validation of run_active_learning: error types and early checks."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from learnm8.api import (
    NO_UNCERTAINTY_LEARNERS,
    _build_cycle_schedule,
    _create_learner,
    run_active_learning,
)
from learnm8.core.interfaces import Oracle
from learnm8.exceptions import ConfigurationError, ValidationError

pytestmark = [pytest.mark.integration]


def _write_pool(path: Path, target: list[str] | None = None) -> Path:
    smiles = ['CCO', 'CCC', 'CCN', 'CCCC', 'CCCl', 'CCBr', 'c1ccccc1', 'CCOC']
    target = target or [str(float(i + 1)) for i in range(len(smiles))]
    lines = ['ID,SMILES,V'] + [
        f'C{i},{s},{t}' for i, (s, t) in enumerate(zip(smiles, target, strict=True))
    ]
    path.write_text('\n'.join(lines) + '\n')
    return path


class _CountingOracle(Oracle):
    def __init__(self) -> None:
        self.calls = 0

    def measure(self, compounds: pl.DataFrame, properties: list[str]) -> pl.DataFrame:
        self.calls += 1
        return compounds.select('ID').with_columns(pl.lit(1.0).alias(properties[0]))


def _run(tmp_path: Path, **kwargs):
    params = dict(
        compound_pool=_write_pool(tmp_path / 'default_pool.csv'),
        oracle=None,
        learner='rf',
        target_col='V',
        featurizer='morgan',
        n_cycles=2,
        batch_fraction=0.25,
        output_dir=tmp_path / 'out',
    )
    params.update(kwargs)
    return run_active_learning(**params)


class TestErrorTypes:
    def test_unknown_learner(self, tmp_path):
        with pytest.raises(ConfigurationError, match='Unknown learner'):
            _run(tmp_path, learner='nope')

    def test_unknown_learner_message_lists_uncertainty_learners(self):
        with pytest.raises(ConfigurationError) as exc:
            _create_learner('nope', random_state=0)
        text = str(exc.value)
        assert 'learnm8 list' not in text
        listed = text.split('Uncertainty-capable: ')[1].split('.')[0].split(', ')
        assert 'rf' in listed
        for name in NO_UNCERTAINTY_LEARNERS:
            assert name not in listed

    def test_missing_featurizer(self, tmp_path):
        with pytest.raises(ConfigurationError, match='requires a featurizer'):
            _run(tmp_path, featurizer=None)

    def test_unknown_featurizer(self, tmp_path):
        with pytest.raises(ConfigurationError, match='Unknown featurizer') as exc:
            _run(tmp_path, featurizer='nope')
        assert 'learnm8 list featurizers' not in str(exc.value)

    def test_pruning_fraction_range(self, tmp_path):
        with pytest.raises(ConfigurationError, match='pruning_fraction'):
            _run(tmp_path, pruning_fraction=1.5)

    @pytest.mark.parametrize('n_jobs', [0, -2])
    def test_bad_n_jobs(self, tmp_path, n_jobs):
        with pytest.raises(ConfigurationError, match='n_jobs'):
            _run(tmp_path, n_jobs=n_jobs)

    def test_bad_device(self, tmp_path):
        with pytest.raises(ConfigurationError, match='device'):
            _run(tmp_path, device='tpu')

    def test_dataframe_pool_without_oracle(self, tmp_path):
        df = pl.DataFrame({'ID': ['a'], 'SMILES': ['CCO']})
        with pytest.raises(ConfigurationError, match='Oracle is required'):
            _run(tmp_path, compound_pool=df)

    def test_missing_pool_file(self, tmp_path):
        with pytest.raises(ConfigurationError, match='not found'):
            _run(tmp_path, compound_pool=tmp_path / 'missing.csv')

    def test_wrong_pool_type(self, tmp_path):
        with pytest.raises(ConfigurationError, match='compound_pool must be'):
            _run(tmp_path, compound_pool=123)

    def test_missing_smiles_column(self, tmp_path):
        df = pl.DataFrame({'ID': ['a'], 'structure': ['CCO']})
        with pytest.raises(ValidationError, match='missing required'):
            _run(tmp_path, compound_pool=df, oracle=_CountingOracle())

    def test_no_valid_compounds(self, tmp_path):
        df = pl.DataFrame({'ID': ['a', 'b'], 'SMILES': ['not-a-smiles', 'also bad']})
        with pytest.raises(ValidationError, match='No valid compounds'):
            _run(tmp_path, compound_pool=df, oracle=_CountingOracle())


class TestStrategyChecks:
    def test_unknown_strategy_before_any_oracle_call(self, tmp_path):
        oracle = _CountingOracle()
        df = pl.read_csv(_write_pool(tmp_path / 'p.csv'))
        with pytest.raises(ConfigurationError, match='Unknown acquisition strategy'):
            _run(tmp_path, compound_pool=df, oracle=oracle, strategy='bogus')
        assert oracle.calls == 0

    def test_unknown_strategy_in_cycles_list_names_valid_ones(self, tmp_path):
        from learnm8.core.config import CycleConfig

        with pytest.raises(ConfigurationError, match='greedy'):
            _run(
                tmp_path,
                cycles=[CycleConfig('random', 1, 0.2), CycleConfig('bogus', 2, 0.2)],
            )

    @pytest.mark.parametrize('learner', NO_UNCERTAINTY_LEARNERS[:2])
    def test_uncertainty_strategy_with_non_uncertainty_learner(self, tmp_path, learner):
        oracle = _CountingOracle()
        df = pl.read_csv(_write_pool(tmp_path / 'p.csv'))
        with pytest.raises(ConfigurationError, match='uncertainty') as exc:
            _run(
                tmp_path,
                compound_pool=df,
                oracle=oracle,
                learner=learner,
                strategy='ucb',
            )
        assert 'ucb' in str(exc.value)
        assert oracle.calls == 0

    def test_no_uncertainty_learner_list_matches_learners(self):
        from learnm8.api import LEARNER_REGISTRY

        for name in LEARNER_REGISTRY:
            try:
                instance = _create_learner(name, random_state=0)
            except ConfigurationError:
                continue  # optional dependency (GPU stack) not installed
            assert instance.supports_uncertainty() == (
                name not in NO_UNCERTAINTY_LEARNERS
            ), name


class TestOptionValidation:
    @pytest.mark.parametrize('value', [False, True, 'off', 'sometimes'])
    def test_bad_streaming(self, tmp_path, value):
        with pytest.raises(ConfigurationError, match='streaming'):
            _run(tmp_path, streaming=value)

    def test_bad_output_format(self, tmp_path):
        with pytest.raises(ConfigurationError, match='output_format'):
            _run(tmp_path, output_format='xlsx')

    def test_bad_diversity_metric_name(self, tmp_path):
        with pytest.raises(ConfigurationError, match='diversity metric'):
            _run(tmp_path, disable_molecular_similarity=['nope'])

    def test_diversity_metric_string_rejected(self, tmp_path):
        with pytest.raises(ConfigurationError, match='disable_molecular_similarity'):
            _run(tmp_path, disable_molecular_similarity='scaffold_diversity_index')


def test_build_cycle_schedule_does_not_mutate_pruning_params():
    params = {'extra': 1}
    _build_cycle_schedule(
        cycles=None,
        strategy='greedy',
        n_cycles=2,
        batch_fraction=0.1,
        acquisition_params=None,
        pruning_strategy='score',
        pruning_params=params,
        pruning_fraction=0.3,
    )
    assert params == {'extra': 1}


class TestChempropEnsembleSeed:
    def test_random_state_reaches_members(self):
        from learnm8.learners.ensemble.ensemble import _derive_random_states

        learner = _create_learner('chemprop_ensemble', random_state=7)
        assert learner.random_states == _derive_random_states(7, 3)  # type: ignore[attr-defined]
        assert learner.random_states == [7, 88, 321]  # type: ignore[attr-defined]
        xgb = _create_learner('xgb_ensemble', random_state=7)
        assert learner.random_states == xgb.random_states  # type: ignore[attr-defined]

    def test_random_state_with_fine_tuning(self, tmp_path):
        learner = _create_learner(
            'chemprop_ensemble',
            random_state=7,
            enable_chemprop_fine_tuning=True,
            checkpoint_dir=tmp_path,
        )
        assert learner.random_states == [7, 88, 321]  # type: ignore[attr-defined]


class TestCsvOracleTargets:
    @pytest.mark.parametrize('bad', ['', 'nan', 'inf', '-inf'])
    def test_bad_target_rejected_before_cycle_0(self, tmp_path, bad):
        target = ['1.0', '2.0', bad, '4.0', '5.0', '6.0', '7.0', '8.0']
        pool = _write_pool(tmp_path / 'pool.csv', target)
        with pytest.raises(ValidationError, match=r'IDs: C2\b') as exc:
            _run(tmp_path, compound_pool=pool, oracle=None)
        assert 'target column' in str(exc.value)

    def test_explicit_oracle_csv_rejected_too(self, tmp_path):
        pool = _write_pool(tmp_path / 'pool.csv')
        oracle = _write_pool(
            tmp_path / 'oracle.csv', ['1', '2', '3', '', '5', '6', '7', 'nan']
        )
        with pytest.raises(ValidationError, match=r'2 compound\(s\)'):
            _run(tmp_path, compound_pool=pool, oracle=oracle)

    def test_ids_absent_from_oracle_still_just_dropped(self, tmp_path):
        pool = _write_pool(tmp_path / 'pool.csv')
        oracle = tmp_path / 'oracle.csv'
        oracle.write_text(
            'ID,V\n' + '\n'.join(f'C{i},{i + 1.0}' for i in range(7)) + '\n'
        )
        results = _run(tmp_path, compound_pool=pool, oracle=oracle, n_cycles=2)
        assert results['compounds_df'].height == 7

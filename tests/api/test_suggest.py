"""Tests for suggest_next_batch (lab-in-the-loop one-shot suggestion)."""

from pathlib import Path

import polars as pl
import pytest

from learnm8 import suggest_next_batch
from learnm8.core.lab_batches import batch_path
from learnm8.exceptions import ConfigurationError, ValidationError

pytestmark = pytest.mark.integration

POOL_CSV = Path(__file__).parent.parent / 'data' / 'small_molecules.csv'
KW = dict(
    target_col='Activity', learner='rf', featurizer='morgan', batch_size=5, n_jobs=1
)


@pytest.fixture
def pool():
    return POOL_CSV


@pytest.fixture
def truth():
    df = pl.read_csv(POOL_CSV)
    return dict(zip(df['ID'].to_list(), df['Activity'].to_list(), strict=True))


def fill(path, truth, overrides=None):
    """Rewrite the Activity column of a batch file from truth (+ cell overrides)."""
    overrides = overrides or {}
    df = pl.read_csv(path, infer_schema=False, skip_rows=1)
    cells = [
        overrides[i] if i in overrides else str(truth[i]) for i in df['ID'].to_list()
    ]
    path.write_text(
        'sep=,\n' + df.with_columns(pl.Series('Activity', cells)).write_csv()
    )


def test_cycle0_random_batch(tmp_path, pool):
    b = suggest_next_batch(pool, tmp_path, **KW)
    assert (tmp_path / 'batch_cycle_0.csv').exists()
    assert b.columns == ['rank', 'ID', 'Activity', 'SMILES']
    assert b['rank'].to_list() == [1, 2, 3, 4, 5] and b['Activity'].null_count() == 5


def test_three_round_campaign_disjoint(tmp_path, pool, truth):
    seen = []
    for cycle in range(3):
        b = suggest_next_batch(pool, tmp_path, strategy='ucb', **KW)
        seen += b['ID'].to_list()
        fill(batch_path(tmp_path, cycle), truth)
    assert len(seen) == len(set(seen)) == 15
    cols = set(pl.read_csv(batch_path(tmp_path, 2), skip_rows=1).columns)
    assert {'prediction', 'uncertainty'} <= cols


def test_pending_and_failed_never_reselected(tmp_path, pool, truth):
    b0 = suggest_next_batch(pool, tmp_path, **KW)
    ids = b0['ID'].to_list()
    fill(batch_path(tmp_path, 0), truth, overrides={ids[0]: '', ids[1]: 'precipitated'})
    kw = {k: v for k, v in KW.items() if k != 'batch_size'}
    b1 = suggest_next_batch(pool, tmp_path, batch_size=45, **kw)
    assert not set(ids) & set(b1['ID'])
    assert b1.height == 45


def test_rerun_before_results_refused(tmp_path, pool):
    suggest_next_batch(pool, tmp_path, **KW)
    with pytest.raises(ConfigurationError, match=r'batch_cycle_0\.csv') as exc:
        suggest_next_batch(pool, tmp_path, **KW)
    assert 'no results entered yet' in str(exc.value)
    assert not batch_path(tmp_path, 1).exists()


def test_all_failed_cycle0_draws_random_batch(tmp_path, pool):
    b0 = suggest_next_batch(pool, tmp_path, **KW)
    fill(batch_path(tmp_path, 0), {}, overrides={i: 'failed' for i in b0['ID']})
    b1 = suggest_next_batch(pool, tmp_path, **KW)
    assert batch_path(tmp_path, 1).exists()
    assert b1.height == 5 and not set(b0['ID']) & set(b1['ID'])
    assert b1.columns == ['rank', 'ID', 'Activity', 'SMILES']


def test_partial_return_proceeds(tmp_path, pool, truth):
    b0 = suggest_next_batch(pool, tmp_path, **KW)
    fill(batch_path(tmp_path, 0), truth, overrides={b0['ID'][0]: ''})
    b1 = suggest_next_batch(pool, tmp_path, **KW)
    assert b1.height == 5 and b0['ID'][0] not in b1['ID'].to_list()


def test_newest_all_failed_with_earlier_measured_proceeds(tmp_path, pool, truth):
    suggest_next_batch(pool, tmp_path, **KW)
    fill(batch_path(tmp_path, 0), truth)
    b1 = suggest_next_batch(pool, tmp_path, **KW)
    fill(batch_path(tmp_path, 1), {}, overrides={i: 'failed' for i in b1['ID']})
    b2 = suggest_next_batch(pool, tmp_path, **KW)
    assert b2.height == 5 and not set(b1['ID']) & set(b2['ID'])
    assert b2.columns == ['rank', 'ID', 'Activity', 'SMILES', 'prediction']


def test_semicolon_resave_end_to_end(tmp_path, pool, truth):
    b0 = suggest_next_batch(pool, tmp_path, **KW)
    rows = '\r\n'.join(
        f'{r};{i};{str(truth[i]).replace(".", ",")};{s}'
        for r, i, s in zip(b0['rank'], b0['ID'], b0['SMILES'], strict=True)
    )
    batch_path(tmp_path, 0).write_bytes(
        ('\ufeffrank;ID;Activity;SMILES\r\n' + rows + '\r\n').encode('utf-8')
    )
    b1 = suggest_next_batch(pool, tmp_path, **KW)
    assert b1.height == 5 and not set(b0['ID']) & set(b1['ID'])


def test_smiles_mismatch_raises(tmp_path, pool):
    b0 = suggest_next_batch(pool, tmp_path, **KW)
    path = batch_path(tmp_path, 0)
    text = path.read_text().replace(b0['SMILES'][0], 'CCO', 1)
    path.write_text(text)
    with pytest.raises(ValidationError, match=r'batch_cycle_0\.csv'):
        suggest_next_batch(pool, tmp_path, **KW)


def test_unknown_strategy_fails_at_cycle0(tmp_path, pool):
    with pytest.raises(ConfigurationError, match='nonsense'):
        suggest_next_batch(pool, tmp_path, strategy='nonsense', **KW)
    assert not batch_path(tmp_path, 0).exists()


def test_unknown_id_raises(tmp_path, pool):
    batch_path(tmp_path, 0).write_text('rank,ID,Activity,SMILES\n1,NOT_IN_POOL,1.0,C\n')
    with pytest.raises(ValidationError, match=r'NOT_IN_POOL.*batch_cycle_0\.csv'):
        suggest_next_batch(pool, tmp_path, **KW)


def test_deterministic_rerun(tmp_path, pool, truth):
    suggest_next_batch(pool, tmp_path, **KW)
    fill(batch_path(tmp_path, 0), truth)
    first = suggest_next_batch(pool, tmp_path, strategy='thompson', **KW)
    batch_path(tmp_path, 1).unlink()
    again = suggest_next_batch(pool, tmp_path, strategy='thompson', **KW)
    assert first.equals(again)


def test_pool_exhausted_refused(tmp_path, pool, truth):
    kw = {k: v for k, v in KW.items() if k != 'batch_size'}
    suggest_next_batch(pool, tmp_path, batch_size=50, **kw)
    fill(batch_path(tmp_path, 0), truth)
    with pytest.raises(ConfigurationError, match='no selectable compounds'):
        suggest_next_batch(pool, tmp_path, **KW)


def test_feature_learner_without_featurizer_refused(tmp_path, pool):
    with pytest.raises(ConfigurationError, match='featurizer'):
        suggest_next_batch(
            pool, tmp_path, target_col='Activity', learner='rf', featurizer=None
        )


def test_invalid_score_direction_fails_at_cycle0(tmp_path, pool):
    with pytest.raises(ConfigurationError, match='score_direction'):
        suggest_next_batch(pool, tmp_path, score_direction='up', **KW)
    assert not batch_path(tmp_path, 0).exists()


def test_non_dict_acquisition_params_rejected(tmp_path, pool):
    with pytest.raises(ConfigurationError, match='acquisition_params'):
        suggest_next_batch(pool, tmp_path, acquisition_params=[1, 2], **KW)
    assert not batch_path(tmp_path, 0).exists()


def test_missing_pool_raises_without_creating_work_dir(tmp_path):
    batch_dir = tmp_path / 'batches'
    with pytest.raises(ConfigurationError, match='not found'):
        suggest_next_batch(tmp_path / 'nope.csv', batch_dir, **KW)
    assert not batch_dir.exists()


def test_leading_zero_ids_round_trip(tmp_path):
    smiles = ['CCO', 'CCN', 'CCC', 'CCCl', 'CCBr', 'CCF', 'CCCC', 'CCCO', 'CCCN']
    ids = [f'{i:03d}' for i in range(1, 10)]
    pool_csv = tmp_path / 'pool.csv'
    pl.DataFrame({'ID': ids, 'SMILES': smiles}).write_csv(pool_csv)
    b = suggest_next_batch(
        pool_csv, tmp_path / 'b', 'Activity', 'rf', 'morgan', batch_size=9, n_jobs=1
    )
    assert sorted(b['ID'].to_list()) == ids
    written = pl.read_csv(
        batch_path(tmp_path / 'b', 0), infer_schema=False, skip_rows=1
    )
    assert sorted(written['ID'].to_list()) == ids


def test_noncanonical_batch_smiles_accepted(tmp_path, pool):
    from rdkit import Chem

    b0 = suggest_next_batch(pool, tmp_path, **KW)
    path = batch_path(tmp_path, 0)
    mol = Chem.MolFromSmiles(b0['SMILES'][0])
    other = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
    path.write_text(path.read_text().replace(b0['SMILES'][0], other, 1))
    fill(path, {i: 1.0 for i in b0['ID']})
    b1 = suggest_next_batch(pool, tmp_path, **KW)
    assert b1.height == 5


def _zero_pool(tmp_path, ids):
    smiles = ['CCO', 'CCN', 'CCC', 'CCCl', 'CCBr', 'CCF', 'CCCC', 'CCCO', 'CCCN']
    pool_csv = tmp_path / 'pool.csv'
    pl.DataFrame({'ID': ids, 'SMILES': smiles[: len(ids)]}).write_csv(pool_csv)
    return pool_csv


def test_stripped_leading_zero_id_is_recovered(tmp_path, monkeypatch):
    import learnm8.suggest as suggest_mod

    captured = {}
    real = suggest_mod.execute_cycle

    def spy(master, *a, **k):
        captured['master'] = master
        return real(master, *a, **k)

    monkeypatch.setattr(suggest_mod, 'execute_cycle', spy)
    pool_csv = _zero_pool(tmp_path, ['007', '008', '009', '010', '011', '012'])
    d = tmp_path / 'b'
    d.mkdir()
    batch_path(d, 0).write_text('rank,ID,Activity,SMILES\n1,7,1.5,CCO\n2,8,2.5,\n')
    b = suggest_next_batch(
        pool_csv, d, 'Activity', 'rf', 'morgan', batch_size=2, n_jobs=1
    )
    assert not {'007', '008'} & set(b['ID'].to_list())
    labeled = captured['master'].filter(pl.col('status') == 'labeled')
    assert dict(zip(labeled['ID'], labeled['Activity'], strict=True)) == {
        '007': 1.5,
        '008': 2.5,
    }


def test_ambiguous_stripped_id_raises(tmp_path):
    pool_csv = _zero_pool(tmp_path, ['07', '007', '008', '009'])
    d = tmp_path / 'b'
    d.mkdir()
    batch_path(d, 0).write_text('rank,ID,Activity,SMILES\n1,7,1.5,\n')
    with pytest.raises(ValidationError, match=r"'?7'? \(batch_cycle_0\.csv\)"):
        suggest_next_batch(
            pool_csv, d, 'Activity', 'rf', 'morgan', batch_size=2, n_jobs=1
        )


def test_stripped_id_with_different_smiles_raises(tmp_path):
    pool_csv = _zero_pool(tmp_path, ['007', '008', '009', '010'])
    d = tmp_path / 'b'
    d.mkdir()
    batch_path(d, 0).write_text('rank,ID,Activity,SMILES\n1,7,1.5,CCCCCC\n')
    with pytest.raises(ValidationError, match=r'not found.*7 \(batch_cycle_0\.csv\)'):
        suggest_next_batch(
            pool_csv, d, 'Activity', 'rf', 'morgan', batch_size=2, n_jobs=1
        )


def test_recovery_creating_duplicate_raises(tmp_path):
    pool_csv = _zero_pool(tmp_path, ['007', '008', '009', '010'])
    d = tmp_path / 'b'
    d.mkdir()
    batch_path(d, 0).write_text('rank,ID,Activity,SMILES\n1,007,1.5,\n')
    batch_path(d, 1).write_text('rank,ID,Activity,SMILES\n1,7,2.5,\n')
    with pytest.raises(
        ValidationError, match=r'batch_cycle_0\.csv.*batch_cycle_1\.csv'
    ):
        suggest_next_batch(
            pool_csv, d, 'Activity', 'rf', 'morgan', batch_size=2, n_jobs=1
        )


def test_non_digit_id_not_remapped(tmp_path):
    pool_csv = _zero_pool(tmp_path, ['0A12', 'B1', 'C1', 'D1'])
    d = tmp_path / 'b'
    d.mkdir()
    batch_path(d, 0).write_text('rank,ID,Activity,SMILES\n1,A12,1.5,\n')
    with pytest.raises(ValidationError, match=r'not found.*A12 \(batch_cycle_0\.csv\)'):
        suggest_next_batch(
            pool_csv, d, 'Activity', 'rf', 'morgan', batch_size=2, n_jobs=1
        )


def test_present_id_with_zero_variant_and_empty_smiles_is_ambiguous(tmp_path):
    pool_csv = _zero_pool(tmp_path, ['7', '0007', '008', '009'])
    d = tmp_path / 'b'
    d.mkdir()
    batch_path(d, 0).write_text('rank,ID,Activity,SMILES\n1,7,1.5,\n')
    with pytest.raises(ValidationError, match=r'Ambiguous.*7 \(batch_cycle_0\.csv\)'):
        suggest_next_batch(
            pool_csv, d, 'Activity', 'rf', 'morgan', batch_size=2, n_jobs=1
        )

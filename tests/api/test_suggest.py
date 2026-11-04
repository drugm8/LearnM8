"""Tests for suggest_next_batch (lab-in-the-loop one-shot suggestion)."""

import logging
from contextlib import contextmanager
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


def test_empty_smiles_cell_warns_unchecked(tmp_path, pool, truth):
    suggest_next_batch(pool, tmp_path, **KW)
    path = batch_path(tmp_path, 0)
    fill(path, truth)
    df = pl.read_csv(path, infer_schema=False, skip_rows=1)
    blanked = df['ID'][0]
    path.write_text(
        'sep=,\n'
        + df.with_columns(
            pl.when(pl.col('ID') == blanked)
            .then(None)
            .otherwise(pl.col('SMILES'))
            .alias('SMILES')
        ).write_csv()
    )
    suggest_next_batch(pool, tmp_path, **KW)
    log = (tmp_path / '.learnm8' / 'learnm8.log').read_text()
    assert 'could not be cross-checked' in log
    assert f'{blanked} (batch_cycle_0.csv)' in log


PRUNE = dict(pruning_fraction=0.3, pruning_start=2)


def _rounds(pool, batch_dir, truth, n, **kw):
    """Run n suggest rounds, filling each batch from truth; return the batches."""
    batches = []
    for cycle in range(n):
        batches.append(suggest_next_batch(pool, batch_dir, **KW, **kw))
        fill(batch_path(batch_dir, cycle), truth)
    return batches


def _pruned(batch_dir, cycle):
    return pl.read_parquet(batch_dir / f'pruned_cycle_{cycle}.parquet')


@pytest.fixture(params=['streaming', 'legacy'])
def selection_path(request, monkeypatch):
    """Force execute_cycle onto one selection path; yield (path, paths seen)."""
    import learnm8.suggest as suggest_mod

    real = suggest_mod.execute_cycle
    mode = 'always' if request.param == 'streaming' else 'never'
    seen = []

    def run(*a, **k):
        master, metrics = real(*a, streaming=mode, **k)
        seen.append(metrics['selection_path'])
        return master, metrics

    monkeypatch.setattr(suggest_mod, 'execute_cycle', run)
    return request.param, seen


def test_pruning_off_by_default(tmp_path, pool, truth):
    _rounds(pool, tmp_path / 'a', truth, 4)
    _rounds(pool, tmp_path / 'b', truth, 4, pruning_fraction=0.0, pruning_start=1)
    for d in ('a', 'b'):
        assert not list((tmp_path / d).glob('pruned_cycle_*'))
    for c in range(4):
        assert (
            batch_path(tmp_path / 'a', c).read_bytes()
            == batch_path(tmp_path / 'b', c).read_bytes()
        )


def test_pruning_campaign(tmp_path, pool, truth, selection_path):
    path, seen = selection_path
    batches = _rounds(pool, tmp_path, truth, 4, **PRUNE)
    assert not list(tmp_path.glob('pruned_cycle_[01].parquet'))
    p2, p3 = _pruned(tmp_path, 2), _pruned(tmp_path, 3)
    assert p2.schema == pl.Schema({'ID': pl.Utf8, 'SMILES': pl.Utf8})
    # Unlabeled pool when pruning: 50 - 10 batched = 40, then 50 - 15 - 12 = 23.
    assert p2['ID'].n_unique() == p2.height == int(0.3 * 40)
    assert p3['ID'].n_unique() == p3.height == int(0.3 * 23)
    assert not set(p2['ID']) & set(p3['ID'])
    batched = {i for b in batches for i in b['ID']}
    assert len(batched) == 20 and not batched & (set(p2['ID']) | set(p3['ID']))
    log = (tmp_path / '.learnm8' / 'learnm8.log').read_text()
    assert 'Pruned 6 compounds in cycle 3 (18 pruned so far)' in log
    assert set(seen) == {path}


def test_no_pruning_without_measurements(tmp_path, pool):
    kw = dict(pruning_fraction=0.3, pruning_start=1)
    b0 = suggest_next_batch(pool, tmp_path, **KW, **kw)
    fill(batch_path(tmp_path, 0), {}, overrides={i: 'failed' for i in b0['ID']})
    suggest_next_batch(pool, tmp_path, **KW, **kw)
    assert batch_path(tmp_path, 1).exists()
    assert not list(tmp_path.glob('pruned_cycle_*'))


def test_rerun_of_pruned_round_reproduces_it(tmp_path, pool, truth, selection_path):
    _rounds(pool, tmp_path, truth, 2, **PRUNE)
    first = suggest_next_batch(pool, tmp_path, **KW, **PRUNE)
    written = batch_path(tmp_path, 2).read_bytes()
    pruned = _pruned(tmp_path, 2)
    batch_path(tmp_path, 2).unlink()
    # A stale record for the round being redone is ignored, then replaced.
    pl.DataFrame({'ID': first['ID'], 'SMILES': first['SMILES']}).write_parquet(
        tmp_path / 'pruned_cycle_2.parquet'
    )
    again = suggest_next_batch(pool, tmp_path, **KW, **PRUNE)
    assert first.equals(again)
    assert batch_path(tmp_path, 2).read_bytes() == written
    assert _pruned(tmp_path, 2).equals(pruned)
    log = (tmp_path / '.learnm8' / 'learnm8.log').read_text()
    assert 'pruned_cycle_2.parquet' in log and 'Ignoring' in log


def test_stale_prune_records_ignored(tmp_path, pool, truth):
    suggest_next_batch(pool, tmp_path, **KW)
    fill(batch_path(tmp_path, 0), truth)
    first = suggest_next_batch(pool, tmp_path, **KW)
    batch_path(tmp_path, 1).unlink()
    for k in (1, 4):
        pl.DataFrame({'ID': first['ID'], 'SMILES': first['SMILES']}).write_parquet(
            tmp_path / f'pruned_cycle_{k}.parquet'
        )
    again = suggest_next_batch(pool, tmp_path, **KW)
    assert first.equals(again)
    # Round 1 pruned nothing, so its stale record goes; round 4's is left alone.
    assert not (tmp_path / 'pruned_cycle_1.parquet').exists()
    assert (tmp_path / 'pruned_cycle_4.parquet').exists()


@pytest.mark.parametrize(
    'kw',
    [
        {'pruning_fraction': -0.1},
        {'pruning_fraction': 0.95},
        {'pruning_fraction': float('nan')},
        {'pruning_start': 0},
        {'pruning_start': 2.5},
        {'pruning_start': True},
        {'pruning_fraction': '0.3'},
        {'pruning_fraction': True},
    ],
)
def test_invalid_pruning_settings_rejected(tmp_path, pool, kw):
    with pytest.raises(ConfigurationError, match=next(iter(kw))):
        suggest_next_batch(pool, tmp_path / 'b', **KW, **kw)
    assert not (tmp_path / 'b').exists()


@contextmanager
def _capture(caplog, level):
    """Route the non-propagating 'learnm8' logger into caplog (call after round 0)."""
    lg = logging.getLogger('learnm8')
    lg.addHandler(caplog.handler)
    caplog.set_level(level, logger='learnm8')
    try:
        yield
    finally:
        lg.removeHandler(caplog.handler)


def _prune_rows(master_pool, ids):
    df = pl.read_csv(master_pool, infer_schema_length=0)
    return df.filter(pl.col('ID').is_in(ids)).select('ID', 'SMILES')


def test_measured_value_wins_over_earlier_prune_record(
    tmp_path, pool, truth, selection_path, caplog
):
    import learnm8.suggest as suggest_mod

    labeled_seen = []
    real = suggest_mod.execute_cycle

    def spy(master, *a, **k):
        labeled_seen.append(master.filter(pl.col('status') == 'labeled'))
        return real(master, *a, **k)

    b = _rounds(pool, tmp_path, truth, 2)
    measured_id = b[0]['ID'][0]
    _prune_rows(pool, [measured_id]).write_parquet(tmp_path / 'pruned_cycle_1.parquet')
    import pytest as _pt

    with _pt.MonkeyPatch.context() as mp, _capture(caplog, logging.WARNING):
        mp.setattr(suggest_mod, 'execute_cycle', spy)
        b2 = suggest_next_batch(pool, tmp_path, **KW)
    labeled = labeled_seen[-1]
    assert labeled.height == 10 and measured_id in labeled['ID'].to_list()
    assert measured_id not in b2['ID'].to_list()
    assert any(
        r.levelno == logging.WARNING and measured_id in r.getMessage()
        for r in caplog.records
    )


def test_pruning_score_direction_lower_prunes_highest(
    tmp_path, pool, truth, selection_path
):
    kw = dict(pruning_fraction=0.3, pruning_start=2, score_direction='lower')
    _rounds(pool, tmp_path, truth, 2, **kw)
    suggest_next_batch(pool, tmp_path, **KW, **kw)
    pruned = set(_pruned(tmp_path, 2)['ID'])
    preds = pl.read_parquet(tmp_path / '.learnm8' / 'prediction_cycle_2.parquet')
    in_pruned = pl.col('ID').is_in(list(pruned))
    assert pruned
    assert (
        preds.filter(in_pruned)['prediction'].min()
        >= preds.filter(~in_pruned)['prediction'].max()
    )


def test_crash_between_prune_and_batch_write_recovers(
    tmp_path, pool, truth, selection_path
):
    import learnm8.suggest as suggest_mod

    ref = tmp_path / 'ref'
    run = tmp_path / 'run'
    _rounds(pool, ref, truth, 2, **PRUNE)
    suggest_next_batch(pool, ref, **KW, **PRUNE)
    _rounds(pool, run, truth, 2, **PRUNE)

    def boom(*a, **k):
        raise OSError('disk full')

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(suggest_mod, 'write_batch', boom)
        with pytest.raises(OSError, match='disk full'):
            suggest_next_batch(pool, run, **KW, **PRUNE)
    assert not batch_path(run, 2).exists()
    assert (run / 'pruned_cycle_2.parquet').exists()
    suggest_next_batch(pool, run, **KW, **PRUNE)
    assert batch_path(run, 2).read_bytes() == batch_path(ref, 2).read_bytes()
    assert _pruned(run, 2).equals(_pruned(ref, 2))


def test_turning_pruning_off_keeps_earlier_pruned_excluded(
    tmp_path, pool, truth, selection_path
):
    _rounds(pool, tmp_path, truth, 4, **PRUNE)
    pruned = set(_pruned(tmp_path, 2)['ID']) | set(_pruned(tmp_path, 3)['ID'])
    b4 = suggest_next_batch(pool, tmp_path, **KW)
    assert not pruned & set(b4['ID'])
    assert not (tmp_path / 'pruned_cycle_4.parquet').exists()


def test_leading_zero_ids_round_trip_through_prune_file(tmp_path, selection_path):
    ids = [f'{i:05d}' for i in range(41, 50)]
    pool_csv = _zero_pool(tmp_path, ids)
    kw = dict(
        target_col='Activity',
        learner='rf',
        featurizer='morgan',
        batch_size=2,
        n_jobs=1,
        pruning_fraction=0.5,
        pruning_start=1,
    )
    d = tmp_path / 'b'
    batches = []
    for c in range(3):
        batches.append(suggest_next_batch(pool_csv, d, **kw))
        fill(batch_path(d, c), {i: float(k) for k, i in enumerate(ids)})
    # Round 2 has too few compounds left to prune; it re-reads round 1's record.
    p1 = _pruned(d, 1)
    assert p1.schema['ID'] == pl.Utf8 and p1.height > 0
    assert all(i.startswith('000') for i in p1['ID'])
    assert not set(p1['ID']) & set(batches[2]['ID'])


def test_pruning_capped_to_keep_a_full_batch(
    tmp_path, pool, truth, selection_path, caplog
):
    kw = {**KW, 'batch_size': 20}
    prune = dict(pruning_fraction=0.9, pruning_start=1)
    suggest_next_batch(pool, tmp_path, **kw, **prune)
    fill(batch_path(tmp_path, 0), truth)
    with _capture(caplog, logging.INFO):
        b1 = suggest_next_batch(pool, tmp_path, **kw, **prune)
    # 30 unlabeled: 0.9 asks for 27, but only 10 may go so 20 remain.
    assert b1.height == 20
    assert _pruned(tmp_path, 1).height == 10
    assert any(
        r.levelno == logging.INFO and 'Pruning fraction lowered' in r.getMessage()
        for r in caplog.records
    )


def test_pruning_skipped_when_no_full_batch_would_remain(
    tmp_path, pool, truth, selection_path, caplog
):
    kw = {**KW, 'batch_size': 40}
    prune = dict(pruning_fraction=0.3, pruning_start=1)
    suggest_next_batch(pool, tmp_path, **kw, **prune)
    fill(batch_path(tmp_path, 0), truth)
    with _capture(caplog, logging.INFO):
        b1 = suggest_next_batch(pool, tmp_path, **kw, **prune)
    assert b1.height == 10
    assert not list(tmp_path.glob('pruned_cycle_*'))
    assert any('pruning skipped' in r.getMessage() for r in caplog.records)


def test_prune_file_unknown_id_raises(tmp_path, pool, truth):
    _rounds(pool, tmp_path, truth, 2)
    pl.DataFrame({'ID': ['nope'], 'SMILES': ['CCO']}).write_parquet(
        tmp_path / 'pruned_cycle_1.parquet'
    )
    with pytest.raises(ValidationError, match=r'Prune file IDs.*nope'):
        suggest_next_batch(pool, tmp_path, **KW)


def test_prune_file_changed_smiles_raises(tmp_path, pool, truth):
    b = _rounds(pool, tmp_path, truth, 2)
    victim = pl.read_csv(pool, infer_schema_length=0).filter(
        ~pl.col('ID').is_in(b[0]['ID'].to_list() + b[1]['ID'].to_list())
    )['ID'][0]
    pl.DataFrame({'ID': [victim], 'SMILES': ['C1CCCCC1']}).write_parquet(
        tmp_path / 'pruned_cycle_1.parquet'
    )
    with pytest.raises(ValidationError, match=r'SMILES in prune files'):
        suggest_next_batch(pool, tmp_path, **KW)

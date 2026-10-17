import polars as pl
import pytest

from learnm8.core.lab_batches import batch_path, next_cycle, read_batches, write_batch
from learnm8.exceptions import ConfigurationError, ValidationError

pytestmark = pytest.mark.unit


def test_empty_dir_returns_empty_ledger(tmp_path):
    led = read_batches(tmp_path / 'missing', 'IC50')
    assert led.height == 0 and led.schema == {
        'ID': pl.Utf8,
        'cycle': pl.Int64,
        'value': pl.Float64,
        'state': pl.Utf8,
        'raw': pl.Utf8,
        'smiles': pl.Utf8,
    }
    assert next_cycle(led) == 0


def test_value_rules(tmp_path):
    (tmp_path / 'batch_cycle_0.csv').write_text(
        'rank,ID,IC50,SMILES\n1,A,12.5,C\n2,B,,C\n3,C,failed,C\n4,D,nan,C\n5,E,inf,C\n6,F,<0.1,C\n7,G, 3 ,C\n'
    )
    led = read_batches(tmp_path, 'IC50').sort('ID')
    assert led['state'].to_list() == [
        'measured',
        'pending',
        'failed',
        'failed',
        'failed',
        'failed',
        'measured',
    ]
    assert led['value'].to_list() == [12.5, None, None, None, None, None, 3.0]
    assert led['cycle'].unique().to_list() == [0]
    assert next_cycle(led) == 1


def test_german_excel_resave(tmp_path):
    # BOM, ';' separator, CRLF, decimal comma, lone cp1252 byte (0xb5), extra column, reordered rows
    (tmp_path / 'batch_cycle_2.csv').write_bytes(
        b'\xef\xbb\xbfrank;ID;IC50;SMILES;Notiz\r\n2;B;12,3;C;\xb5M\r\n1;A;;C;\r\n'
    )
    led = read_batches(tmp_path, 'IC50').sort('ID')
    assert led['ID'].to_list() == ['A', 'B']
    assert led['state'].to_list() == ['pending', 'measured']
    assert led['value'].to_list() == [None, 12.3]


def test_stray_files_ignored_with_warning(tmp_path, caplog):
    for name in [
        'batch_cycle_0.csv',
        'batch_cycle_0 (1).csv',
        '~$batch_cycle_0.csv',
        'notes.csv',
    ]:
        (tmp_path / name).write_text('rank,ID,IC50,SMILES\n1,A,1.0,C\n')
    led = read_batches(tmp_path, 'IC50')
    assert led.height == 1
    assert 'batch_cycle_0 (1).csv' in caplog.text


def test_missing_target_column_raises(tmp_path):
    (tmp_path / 'batch_cycle_0.csv').write_text('rank,ID,Value,SMILES\n1,A,1.0,C\n')
    with pytest.raises(ValidationError, match=r'batch_cycle_0\.csv.*IC50'):
        read_batches(tmp_path, 'IC50')


def test_duplicate_id_across_files_raises(tmp_path):
    for c in (0, 1):
        (tmp_path / f'batch_cycle_{c}.csv').write_text(
            'rank,ID,IC50,SMILES\n1,A,1.0,C\n'
        )
    with pytest.raises(ValidationError, match='A'):
        read_batches(tmp_path, 'IC50')


def test_duplicate_id_within_file_raises(tmp_path):
    (tmp_path / 'batch_cycle_0.csv').write_text(
        'rank,ID,IC50,SMILES\n1,A,1.0,C\n2,A,,C\n'
    )
    with pytest.raises(ValidationError, match='A'):
        read_batches(tmp_path, 'IC50')


def test_write_batch_layout_and_no_overwrite(tmp_path):
    batch = pl.DataFrame(
        {
            'rank': [1, 2],
            'ID': ['A', 'B'],
            'SMILES': ['C', 'CC'],
            'prediction': [0.5, 0.4],
        }
    )
    p = write_batch(batch, tmp_path, 3, 'IC50')
    assert p == batch_path(tmp_path, 3)
    assert p.read_text().splitlines()[0] == 'rank,ID,IC50,SMILES,prediction'
    assert read_batches(tmp_path, 'IC50')['state'].to_list() == ['pending', 'pending']
    with pytest.raises(ConfigurationError, match=r'batch_cycle_3\.csv'):
        write_batch(batch, tmp_path, 3, 'IC50')


def test_semicolon_dot_value_is_ambiguous(tmp_path):
    (tmp_path / 'batch_cycle_0.csv').write_text('rank;ID;IC50;SMILES\n1;A;1.234;C\n')
    with pytest.raises(ValidationError, match=r"batch_cycle_0\.csv.*'A'.*1\.234"):
        read_batches(tmp_path, 'IC50')


def test_comma_file_dot_is_decimal_and_semicolon_text_with_dot_fails(tmp_path):
    (tmp_path / 'batch_cycle_0.csv').write_text('rank,ID,IC50,SMILES\n1,A,1.234,C\n')
    (tmp_path / 'batch_cycle_1.csv').write_text('rank;ID;IC50;SMILES\n1;B;12. Mrz;C\n')
    led = read_batches(tmp_path, 'IC50').sort('ID')
    assert led['value'].to_list() == [1.234, None]
    assert led['state'].to_list() == ['measured', 'failed']
    assert led['raw'].to_list() == ['1.234', '12. Mrz']


def test_smiles_cell_kept_empty_is_null(tmp_path):
    (tmp_path / 'batch_cycle_0.csv').write_text(
        'rank,ID,IC50,SMILES\n1,A,1.0,CCO\n2,B,2.0,\n'
    )
    led = read_batches(tmp_path, 'IC50').sort('ID')
    assert led['smiles'].to_list() == ['CCO', None]


def test_unparseable_and_empty_files_name_the_file(tmp_path):
    (tmp_path / 'batch_cycle_0.csv').write_bytes(b'')
    with pytest.raises(ValidationError, match=r'batch_cycle_0\.csv'):
        read_batches(tmp_path, 'IC50')
    (tmp_path / 'batch_cycle_0.csv').write_text('rank,ID,IC50\n1,A,1.0,extra,cells\n')
    with pytest.raises(ValidationError, match=r'batch_cycle_0\.csv'):
        read_batches(tmp_path, 'IC50')


def test_blank_rows_dropped_and_missing_id_raises(tmp_path):
    (tmp_path / 'batch_cycle_0.csv').write_text(
        'rank;ID;IC50;SMILES\n1;A;1,5;C\n;;;\n  ; ;  ;\n'
    )
    assert read_batches(tmp_path, 'IC50')['ID'].to_list() == ['A']
    (tmp_path / 'batch_cycle_0.csv').write_text(
        'rank,ID,IC50,SMILES\n1,A,1.0,C\n2,,3.0,C\n'
    )
    with pytest.raises(ValidationError, match=r'batch_cycle_0\.csv.*\[3\]'):
        read_batches(tmp_path, 'IC50')


def test_cross_file_duplicate_names_files(tmp_path):
    for c in (0, 1):
        (tmp_path / f'batch_cycle_{c}.csv').write_text(
            'rank,ID,IC50,SMILES\n1,A,1.0,C\n'
        )
    with pytest.raises(
        ValidationError, match=r'batch_cycle_0\.csv.*batch_cycle_1\.csv'
    ):
        read_batches(tmp_path, 'IC50')

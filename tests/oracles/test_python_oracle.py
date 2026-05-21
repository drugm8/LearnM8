import pytest
import polars as pl
from learnm8.exceptions import OracleError
from learnm8.oracles.python_oracle import PythonOracle


def _write_oracle_file(tmp_path, content, filename="oracle.py"):
    f = tmp_path / filename
    f.write_text(content)
    return f


@pytest.mark.unit
class TestPythonOracleInit:

    def test_init_with_module_path(self, tmp_path):
        f = _write_oracle_file(tmp_path, "def oracle(compound_ids):\n    import polars as pl\n    return pl.DataFrame({'ID': compound_ids, 'score': [1.0]*len(compound_ids)})\n")
        oracle = PythonOracle(module_path=str(f))
        assert oracle.oracle_path == f

    def test_init_with_oracle_path_legacy(self, tmp_path):
        f = _write_oracle_file(tmp_path, "def oracle(compound_ids):\n    import polars as pl\n    return pl.DataFrame({'ID': compound_ids, 'score': [1.0]*len(compound_ids)})\n")
        oracle = PythonOracle(oracle_path=str(f))
        assert oracle.oracle_path == f

    def test_init_no_path_raises(self):
        with pytest.raises(OracleError, match="PythonOracle requires a path"):
            PythonOracle()

    def test_init_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PythonOracle(module_path=str(tmp_path / "nonexistent.py"))

    def test_init_explicit_function_name(self, tmp_path):
        f = _write_oracle_file(tmp_path,
            "def score(compound_ids):\n    import polars as pl\n    return pl.DataFrame({'ID': compound_ids, 'val': [0.5]*len(compound_ids)})\n"
            "def helper(x):\n    return x\n"
        )
        oracle = PythonOracle(module_path=str(f), function_name='score')
        assert oracle.oracle_function.__name__ == 'score'

    def test_init_function_not_found(self, tmp_path):
        f = _write_oracle_file(tmp_path, "def oracle(compound_ids):\n    import polars as pl\n    return pl.DataFrame({'ID': compound_ids})\n")
        with pytest.raises(OracleError, match="Function 'missing_fn' not found"):
            PythonOracle(module_path=str(f), function_name='missing_fn')

    def test_init_wrong_signature(self, tmp_path):
        f = _write_oracle_file(tmp_path, "def oracle(ids, extra_param):\n    pass\n")
        with pytest.raises(OracleError, match="exactly 1 parameter"):
            PythonOracle(module_path=str(f))

    def test_init_multiple_functions_no_common_name(self, tmp_path):
        f = _write_oracle_file(tmp_path,
            "def func_a(x):\n    return x\ndef func_b(x):\n    return x\n"
        )
        with pytest.raises(OracleError, match="Multiple functions found"):
            PythonOracle(module_path=str(f))

    def test_init_auto_detect_single_function(self, tmp_path):
        f = _write_oracle_file(tmp_path,
            "def my_scorer(compound_ids):\n    import polars as pl\n    return pl.DataFrame({'ID': compound_ids, 'score': [1.0]*len(compound_ids)})\n"
        )
        oracle = PythonOracle(module_path=str(f))
        assert oracle.oracle_function.__name__ == 'my_scorer'

    def test_init_auto_detect_common_name(self, tmp_path):
        f = _write_oracle_file(tmp_path,
            "def helper(x):\n    return x\ndef oracle(compound_ids):\n    import polars as pl\n    return pl.DataFrame({'ID': compound_ids, 'score': [1.0]*len(compound_ids)})\n"
        )
        oracle = PythonOracle(module_path=str(f))
        assert oracle.oracle_function.__name__ == 'oracle'


@pytest.mark.unit
class TestPythonOracleMeasure:

    @pytest.fixture
    def simple_oracle(self, tmp_path):
        f = _write_oracle_file(tmp_path,
            "import polars as pl\n"
            "def oracle(compound_ids):\n"
            "    scores = [float(len(cid)) for cid in compound_ids]\n"
            "    return pl.DataFrame({'ID': compound_ids, 'score': scores})\n"
        )
        return PythonOracle(module_path=str(f))

    def test_measure_calls_function(self, simple_oracle):
        compounds = pl.DataFrame({'ID': ['COMP_001', 'COMP_002'], 'SMILES': ['CCO', 'CCC']})
        result = simple_oracle.measure(compounds, ['score'])
        assert result.height == 2
        assert 'ID' in result.columns
        assert 'score' in result.columns

    def test_measure_correct_ids_passed(self, simple_oracle):
        compounds = pl.DataFrame({'ID': ['A', 'BB', 'CCC'], 'SMILES': ['CCO', 'CCC', 'CCCC']})
        result = simple_oracle.measure(compounds, ['score'])
        id_to_score = dict(zip(result['ID'].to_list(), result['score'].to_list()))
        assert id_to_score['A'] == pytest.approx(1.0)
        assert id_to_score['BB'] == pytest.approx(2.0)
        assert id_to_score['CCC'] == pytest.approx(3.0)

    def test_measure_returns_polars_dataframe(self, simple_oracle):
        compounds = pl.DataFrame({'ID': ['COMP_001'], 'SMILES': ['CCO']})
        result = simple_oracle.measure(compounds, ['score'])
        assert isinstance(result, pl.DataFrame)

    def test_measure_accepts_pandas_return(self, tmp_path):
        f = _write_oracle_file(tmp_path,
            "import pandas as pd\n"
            "def oracle(compound_ids):\n"
            "    return pd.DataFrame({'ID': compound_ids, 'score': [1.0]*len(compound_ids)})\n"
        )
        oracle = PythonOracle(module_path=str(f))
        compounds = pl.DataFrame({'ID': ['COMP_001'], 'SMILES': ['CCO']})
        result = oracle.measure(compounds, ['score'])
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1

    def test_measure_missing_id_in_result(self, tmp_path):
        f = _write_oracle_file(tmp_path,
            "import polars as pl\n"
            "def oracle(compound_ids):\n"
            "    return pl.DataFrame({'score': [1.0]*len(compound_ids)})\n"
        )
        oracle = PythonOracle(module_path=str(f))
        compounds = pl.DataFrame({'ID': ['COMP_001'], 'SMILES': ['CCO']})
        with pytest.raises(OracleError, match="'ID' column"):
            oracle.measure(compounds, ['score'])

    def test_measure_function_raises(self, tmp_path):
        f = _write_oracle_file(tmp_path,
            "def oracle(compound_ids):\n"
            "    raise RuntimeError('scoring failed')\n"
        )
        oracle = PythonOracle(module_path=str(f))
        compounds = pl.DataFrame({'ID': ['COMP_001'], 'SMILES': ['CCO']})
        with pytest.raises(OracleError, match="Oracle function failed"):
            oracle.measure(compounds, ['score'])

    def test_measure_invalid_return_type(self, tmp_path):
        f = _write_oracle_file(tmp_path,
            "def oracle(compound_ids):\n"
            "    return [1.0, 2.0]\n"
        )
        oracle = PythonOracle(module_path=str(f))
        compounds = pl.DataFrame({'ID': ['COMP_001'], 'SMILES': ['CCO']})
        with pytest.raises(OracleError, match="must return"):
            oracle.measure(compounds, ['score'])

    def test_measure_missing_compound_in_result(self, tmp_path, caplog):
        f = _write_oracle_file(tmp_path,
            "import polars as pl\n"
            "def oracle(compound_ids):\n"
            "    return pl.DataFrame({'ID': compound_ids[:1], 'score': [1.0]})\n"
        )
        oracle = PythonOracle(module_path=str(f))
        compounds = pl.DataFrame({'ID': ['COMP_001', 'COMP_002'], 'SMILES': ['CCO', 'CCC']})
        import logging
        logging.getLogger('learnm8').propagate = True
        with caplog.at_level('WARNING', logger='learnm8.oracles.python_oracle'):
            result = oracle.measure(compounds, ['score'])
        assert any('did not return results' in rec.message for rec in caplog.records)
        assert result.height == 1

    def test_measure_empty_compounds(self, simple_oracle):
        compounds = pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8})
        result = simple_oracle.measure(compounds, ['score'])
        assert isinstance(result, pl.DataFrame)
        assert result.height == 0

    def test_measure_shuffled_result_no_mislabeling(self, tmp_path):
        f = _write_oracle_file(tmp_path,
            "import polars as pl\n"
            "def oracle(compound_ids):\n"
            "    rows = [(cid, float(len(cid))) for cid in compound_ids][::-1]\n"
            "    return pl.DataFrame({'ID': [r[0] for r in rows], 'score': [r[1] for r in rows]})\n"
        )
        oracle = PythonOracle(module_path=str(f))
        compounds = pl.DataFrame({'ID': ['A', 'BB', 'CCC', 'DDDD'], 'SMILES': ['C', 'CC', 'CCC', 'CCCC']})
        result = oracle.measure(compounds, ['score'])
        true_values = {'A': 1.0, 'BB': 2.0, 'CCC': 3.0, 'DDDD': 4.0}
        for rid, rscore in zip(
            result['ID'].to_list(), result['score'].to_list(), strict=True
        ):
            assert rscore == pytest.approx(true_values[rid])
        assert result['ID'].to_list() == ['A', 'BB', 'CCC', 'DDDD']

    def test_measure_shuffled_result_logs_warning(self, tmp_path, caplog):
        f = _write_oracle_file(tmp_path,
            "import polars as pl\n"
            "def oracle(compound_ids):\n"
            "    rows = [(cid, float(len(cid))) for cid in compound_ids][::-1]\n"
            "    return pl.DataFrame({'ID': [r[0] for r in rows], 'score': [r[1] for r in rows]})\n"
        )
        oracle = PythonOracle(module_path=str(f))
        compounds = pl.DataFrame({'ID': ['A', 'BB', 'CCC', 'DDDD'], 'SMILES': ['C', 'CC', 'CCC', 'CCCC']})
        import logging
        logging.getLogger('learnm8').propagate = True
        with caplog.at_level('WARNING', logger='learnm8.oracles.python_oracle'):
            oracle.measure(compounds, ['score'])
        assert any('row order' in rec.message for rec in caplog.records)

    def test_measure_in_order_result_no_warning(self, simple_oracle, caplog):
        compounds = pl.DataFrame({'ID': ['A', 'BB', 'CCC'], 'SMILES': ['C', 'CC', 'CCC']})
        import logging
        logging.getLogger('learnm8').propagate = True
        with caplog.at_level('WARNING', logger='learnm8.oracles.python_oracle'):
            result = simple_oracle.measure(compounds, ['score'])
        assert not any('row order' in rec.message for rec in caplog.records)
        assert result['ID'].to_list() == ['A', 'BB', 'CCC']
        true_values = {'A': 1.0, 'BB': 2.0, 'CCC': 3.0}
        for rid, rscore in zip(
            result['ID'].to_list(), result['score'].to_list(), strict=True
        ):
            assert rscore == pytest.approx(true_values[rid])

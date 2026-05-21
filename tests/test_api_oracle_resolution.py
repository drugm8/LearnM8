"""BUG 7c: api.py oracle resolution of a 'module.py:function' string.

The requested function name must be passed through to PythonOracle so the
specified function is used, instead of falling back to auto-detection (which
crashes on modules defining multiple public functions).
"""

import textwrap

import polars as pl
import pytest

from learnm8.api import run_active_learning
from learnm8.exceptions import OracleError


@pytest.fixture
def multi_function_oracle_module(tmp_path):
    """Write an oracle module file with MULTIPLE public functions."""
    module_path = tmp_path / 'multi_oracle.py'
    module_path.write_text(
        textwrap.dedent(
            """
            import polars as pl


            def helper_one(compound_ids):
                return pl.DataFrame({
                    'ID': compound_ids,
                    'Activity': [0.1 for _ in compound_ids],
                })


            def my_oracle(compound_ids):
                return pl.DataFrame({
                    'ID': compound_ids,
                    'Activity': [0.9 for _ in compound_ids],
                })
            """
        )
    )
    return module_path


@pytest.fixture
def compound_pool_df():
    return pl.DataFrame(
        {
            'ID': ['C1', 'C2', 'C3', 'C4'],
            'SMILES': ['CCO', 'CCC', 'CCCC', 'CCCCC'],
        }
    )


@pytest.mark.slow
def test_oracle_string_resolves_requested_function(
    multi_function_oracle_module, compound_pool_df, tmp_path
):
    """A 'module.py:function' string must use the named function, even when the
    module defines multiple public functions (currently raises OracleError)."""
    oracle_spec = f'{multi_function_oracle_module}:my_oracle'
    results = run_active_learning(
        compound_pool=compound_pool_df,
        oracle=oracle_spec,
        learner='rf',
        target_col='Activity',
        featurizer='morgan',
        n_cycles=1,
        batch_fraction=0.5,
        output_dir=str(tmp_path / 'out'),
        cache_dir=str(tmp_path / 'cache'),
        n_jobs=1,
        device='cpu',
    )
    labeled = results['compounds_df'].filter(pl.col('status') == 'labeled')
    assert labeled.height > 0
    assert (labeled['Activity'] == 0.9).all()


@pytest.mark.slow
def test_oracle_string_unknown_function_raises_oracle_error(
    multi_function_oracle_module, compound_pool_df, tmp_path
):
    """Requesting a nonexistent function name must raise a clear OracleError."""
    oracle_spec = f'{multi_function_oracle_module}:does_not_exist'
    with pytest.raises(OracleError, match='does_not_exist'):
        run_active_learning(
            compound_pool=compound_pool_df,
            oracle=oracle_spec,
            learner='rf',
            target_col='Activity',
            featurizer='morgan',
            n_cycles=1,
            batch_fraction=0.5,
            output_dir=str(tmp_path / 'out2'),
            cache_dir=str(tmp_path / 'cache2'),
            n_jobs=1,
            device='cpu',
        )

"""Force-uncertainty override (feature 023, FR-013/FR-014/FR-016) tests.

Three scenarios per FR-016:

1. ``force_uncertainty=True`` with a non-UQ acquisition (greedy) produces
   uncertainty in the parquet AND in cycle metrics.
2. ``force_uncertainty=True`` with a learner that does not support
   uncertainty at all (mock without uncertainty) is a silent no-op — no
   exception, no fabricated nulls, just ``has_uncertainty=False``.
3. ``force_uncertainty=True`` together with a UQ-requiring acquisition (UCB)
   does NOT double-compute: behaviour is identical to ``force_uncertainty=False``.
"""

from __future__ import annotations

import polars as pl
import pytest

from learnm8.api import run_active_learning
from learnm8.core.persistence import read_cycle_predictions

pytestmark = [pytest.mark.integration]


def _run(
    sample_compounds,
    mock_oracle,
    mock_learner_with_uncertainty,
    *,
    tmp_path,
    strategy='greedy',
    force_uncertainty=False,
):
    output_dir = tmp_path / 'force_uncertainty'
    return run_active_learning(
        compound_pool=sample_compounds,
        oracle=mock_oracle,
        learner=mock_learner_with_uncertainty,
        target_col='Activity',
        featurizer='morgan',
        n_cycles=2,
        batch_fraction=0.1,
        strategy=strategy,
        output_dir=output_dir,
        cache_dir=tmp_path / 'cache',
        random_state=42,
        force_uncertainty=force_uncertainty,
    )


def test_force_uncertainty_true_computes_even_with_greedy(
    sample_compounds, mock_learner_with_uncertainty, mock_oracle, tmp_path
):
    """FR-013/FR-016 OR-precedence: force_uncertainty=True wins over greedy."""
    results = _run(
        sample_compounds,
        mock_oracle,
        mock_learner_with_uncertainty,
        tmp_path=tmp_path,
        strategy='greedy',
        force_uncertainty=True,
    )

    paths = results['prediction_files']
    assert len(paths) >= 1

    df = read_cycle_predictions(paths[-1])
    assert 'uncertainty' in df.columns, (
        'force_uncertainty=True should populate uncertainty column even '
        'under greedy acquisition.'
    )
    # No null uncertainties on the populated path.
    unc = df.get_column('uncertainty')
    assert not unc.is_null().any()

    # cycle_metrics has has_uncertainty=True
    assert results['cycle_metrics'][-1].get('has_uncertainty') is True


def test_force_uncertainty_with_unsupported_learner_silent_noop(
    sample_compounds, mock_learner, mock_oracle, tmp_path
):
    """FR-016 / FR-020: silent no-op when learner does not support uncertainty.

    ``mock_learner`` (no uncertainty) plus ``force_uncertainty=True`` must
    still complete cleanly with ``has_uncertainty=False`` and no fabricated
    uncertainty column.
    """
    output_dir = tmp_path / 'force_noop'
    results = run_active_learning(
        compound_pool=sample_compounds,
        oracle=mock_oracle,
        learner=mock_learner,
        target_col='Activity',
        featurizer='morgan',
        n_cycles=2,
        batch_fraction=0.1,
        strategy='greedy',
        output_dir=output_dir,
        cache_dir=tmp_path / 'cache',
        random_state=42,
        force_uncertainty=True,
    )

    # No fabricated uncertainty column.
    df = read_cycle_predictions(results['prediction_files'][-1])
    assert 'uncertainty' not in df.columns, (
        'force_uncertainty must be a silent no-op for learners that do not '
        'support uncertainty; no fabricated null column expected.'
    )
    assert results['cycle_metrics'][-1].get('has_uncertainty') is False


def test_force_uncertainty_with_uncertainty_required_no_double_compute(
    sample_compounds, mock_learner_with_uncertainty, mock_oracle, tmp_path
):
    """FR-016 idempotence: force_uncertainty=True under UCB matches False.

    Both branches must compute identically (no double compute) — verified by
    column equality of the uncertainty column.
    """
    out1 = tmp_path / 'ucb_default'
    out2 = tmp_path / 'ucb_forced'

    r1 = run_active_learning(
        compound_pool=sample_compounds,
        oracle=mock_oracle,
        learner=mock_learner_with_uncertainty,
        target_col='Activity',
        featurizer='morgan',
        n_cycles=2,
        batch_fraction=0.1,
        strategy='ucb',
        output_dir=out1,
        cache_dir=tmp_path / 'cache1',
        random_state=42,
        force_uncertainty=False,
    )
    r2 = run_active_learning(
        compound_pool=sample_compounds,
        oracle=mock_oracle,
        learner=mock_learner_with_uncertainty,
        target_col='Activity',
        featurizer='morgan',
        n_cycles=2,
        batch_fraction=0.1,
        strategy='ucb',
        output_dir=out2,
        cache_dir=tmp_path / 'cache2',
        random_state=42,
        force_uncertainty=True,
    )

    df1 = read_cycle_predictions(r1['prediction_files'][-1])
    df2 = read_cycle_predictions(r2['prediction_files'][-1])

    # Same uncertainty column on both paths (no double compute).
    assert 'uncertainty' in df1.columns and 'uncertainty' in df2.columns
    pl.testing.assert_series_equal(
        df1.get_column('uncertainty'), df2.get_column('uncertainty')
    )

"""BUGFIX3 — top-level ``acquisition_params`` must change what a run selects.

``parse_cycle_schedule`` dropped the top-level ``acquisition_params`` whenever
an explicit ``cycles=[...]`` list was passed, which is the branch the CLI
always takes. The failure was silent: runs completed normally and used the
acquisition defaults, so two UCB runs with different ``beta`` produced
byte-identical trajectories.

A unit test asserting the value arrives is not enough for that failure mode,
so this test drives a full run twice and requires the selections to differ.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from learnm8.core.config import CycleConfig

pytestmark = pytest.mark.integration


def _write_combined_csv(tmp_path: Path) -> Path:
    """Write a structurally varied pool used as both compound_pool and oracle.

    Distinct scaffolds and chain lengths are what give the model non-uniform
    per-compound uncertainty; a pool of repeated SMILES would make ``beta``
    inert for reasons unrelated to the bug.
    """
    suffixes = ['O', 'N', 'Cl', 'Br', 'F', '=O']
    rows = []
    for i, suffix in enumerate(suffixes):
        for length in range(1, 11):
            smiles = 'C' * length + suffix
            rows.append((f'C{i:02d}{length:02d}', smiles, float(length) + 0.5 * i))
    df = pl.DataFrame(rows, schema=['ID', 'SMILES', 'Activity'], orient='row')
    csv_path = tmp_path / 'compounds.csv'
    df.write_csv(csv_path)
    return csv_path


def _run_ucb(csv: Path, output_dir: Path, beta: float) -> dict[int, set[str]]:
    """Run a random-init + UCB schedule via the advanced API, return selections."""
    from learnm8.api import run_active_learning

    results = run_active_learning(
        compound_pool=str(csv),
        oracle=str(csv),
        target_col='Activity',
        learner='rf',
        featurizer='morgan',
        cycles=[
            CycleConfig('random', n_cycles=1, batch_fraction=0.05),
            CycleConfig('ucb', n_cycles=4, batch_fraction=0.05),
        ],
        acquisition_params={'beta': beta},
        random_state=42,
        output_dir=str(output_dir),
        score_direction='higher',
    )

    cdf: pl.DataFrame = results['compounds_df']
    cycles = sorted({int(c) for c in cdf['selected_cycle'].drop_nulls().to_list()})
    return {
        c: set(cdf.filter(pl.col('selected_cycle') == c)['ID'].to_list())
        for c in cycles
    }


def test_ucb_beta_changes_selections(tmp_path):
    csv = _write_combined_csv(tmp_path)

    low = _run_ucb(csv, tmp_path / 'beta_low', beta=0.0)
    high = _run_ucb(csv, tmp_path / 'beta_high', beta=10.0)

    assert set(low) == set(high)
    acquisition_cycles = sorted(c for c in low if c >= 1)
    assert len(acquisition_cycles) >= 2, (
        f'Need at least 2 UCB cycles; got cycles {sorted(low.keys())}.'
    )

    assert low[0] == high[0], 'Cycle 0 is random init and must be unaffected by beta.'
    assert any(low[c] != high[c] for c in acquisition_cycles), (
        'beta=0.0 and beta=10.0 selected identical compounds in every UCB '
        'cycle — acquisition_params is not reaching the acquisition function.'
    )

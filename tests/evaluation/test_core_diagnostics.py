"""Integration tests for diagnostic-metric wiring in evaluate_cycle (014).

Verifies that:
- evaluate_cycle emits all three new keys (selected_percentile,
  prediction_entropy, pool_size_after_prune) regardless of mode
- benchmark mode populates selected_percentile with a sensible value
- run mode emits selected_percentile=None
- the new columns round-trip through export_metrics_csv → pl.read_csv
- existing metric keys are not perturbed
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from learnm8.evaluation.core import evaluate_cycle, export_metrics_csv


def _make_pool_df(n: int = 100, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        'ID': [f'm{i}' for i in range(n)],
        'prediction': rng.uniform(0.0, 1.0, size=n),
    })


def _make_ground_truth(n: int = 100) -> pl.DataFrame:
    return pl.DataFrame({
        'ID': [f'm{i}' for i in range(n)],
        'Activity': np.linspace(0.0, 1.0, n),
    })


def _make_selection(ground_truth: pl.DataFrame, top_k: int = 5) -> pl.DataFrame:
    # Pick the top-k by Activity (simulates greedy on perfect predictions).
    return ground_truth.sort('Activity', descending=True).head(top_k)


@pytest.mark.unit
class TestEvaluateCycleDiagnosticKeys:
    """The three new keys must always be present in the returned dict."""

    def test_benchmark_mode_emits_all_three_keys(self):
        gt = _make_ground_truth(100)
        pool = _make_pool_df(100, seed=1)
        sel = _make_selection(gt, top_k=5)
        # Make the labeled set just the selected compounds so cumulative_selected_ids is well-defined.
        labeled = sel.clone()

        result = evaluate_cycle(
            cycle=1,
            predictions=np.array(labeled['Activity'].to_list()),
            ground_truth=np.array(labeled['Activity'].to_list()),
            labeled_data=labeled,
            selected_compounds=sel,
            target_col='Activity',
            ground_truth_data=gt,
            pool_df=pool,
            cumulative_selected_ids=set(sel['ID'].to_list()),
            score_direction='higher',
        )

        for key in ('selected_percentile', 'prediction_entropy', 'pool_size_after_prune'):
            assert key in result, f"missing key: {key}"

        # Selected the true top 5 of 100 → mean percentile must be high (>=95).
        assert result['selected_percentile'] is not None
        assert result['selected_percentile'] >= 95.0

        # Uniform predictions → entropy should be substantial (closer to log2(50) ≈ 5.6).
        assert result['prediction_entropy'] is not None
        assert result['prediction_entropy'] > 3.0

        assert result['pool_size_after_prune'] == 100

    def test_run_mode_emits_keys_with_percentile_none(self):
        # Run mode = no ground_truth_data.
        labeled = pl.DataFrame({
            'ID': [f'm{i}' for i in range(5)],
            'SMILES': ['C'] * 5,
            'Activity': [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        sel = labeled.head(2)
        pool = _make_pool_df(50, seed=2)

        result = evaluate_cycle(
            cycle=1,
            predictions=np.array(labeled['Activity'].to_list()),
            ground_truth=np.array(labeled['Activity'].to_list()),
            labeled_data=labeled,
            selected_compounds=sel,
            target_col='Activity',
            ground_truth_data=None,            # run mode
            pool_df=pool,
            cumulative_selected_ids=set(sel['ID'].to_list()),
        )

        for key in ('selected_percentile', 'prediction_entropy', 'pool_size_after_prune'):
            assert key in result, f"missing key: {key}"
        assert result['selected_percentile'] is None
        assert result['prediction_entropy'] is not None  # pool present → still computed
        assert result['pool_size_after_prune'] == 50

    def test_pool_df_none_yields_entropy_and_pool_size_none(self):
        labeled = pl.DataFrame({
            'ID': ['a', 'b'],
            'Activity': [0.1, 0.2],
        })
        sel = labeled.head(1)

        result = evaluate_cycle(
            cycle=0,
            predictions=np.array([0.1, 0.2]),
            ground_truth=np.array([0.1, 0.2]),
            labeled_data=labeled,
            selected_compounds=sel,
            target_col='Activity',
            pool_df=None,
        )
        assert result['prediction_entropy'] is None
        assert result['pool_size_after_prune'] is None

    def test_csv_round_trip_preserves_columns(self, tmp_path):
        gt = _make_ground_truth(50)
        pool = _make_pool_df(50, seed=3)
        sel = _make_selection(gt, top_k=3)
        labeled = sel.clone()

        cycle_metrics = evaluate_cycle(
            cycle=2,
            predictions=np.array(labeled['Activity'].to_list()),
            ground_truth=np.array(labeled['Activity'].to_list()),
            labeled_data=labeled,
            selected_compounds=sel,
            target_col='Activity',
            ground_truth_data=gt,
            pool_df=pool,
            cumulative_selected_ids=set(sel['ID'].to_list()),
        )

        csv_path = tmp_path / 'cycle_metrics.csv'
        export_metrics_csv([cycle_metrics], str(csv_path), oracle_type='benchmark',
                            target_col='Activity')

        # comment_prefix='#' to skip metadata header lines emitted by export_metrics_csv
        df = pl.read_csv(csv_path, comment_prefix='#')
        for col in ('selected_percentile', 'prediction_entropy', 'pool_size_after_prune'):
            assert col in df.columns, f"CSV missing column: {col}"

        # Spot-check round-trip values match what evaluate_cycle returned (allow for
        # the 4-decimal rounding evaluate_cycle applies to floats).
        assert df['pool_size_after_prune'][0] == 50
        if cycle_metrics['selected_percentile'] is not None:
            assert df['selected_percentile'][0] == pytest.approx(
                cycle_metrics['selected_percentile'], abs=1e-3)
        if cycle_metrics['prediction_entropy'] is not None:
            assert df['prediction_entropy'][0] == pytest.approx(
                cycle_metrics['prediction_entropy'], abs=1e-3)


@pytest.mark.unit
class TestEvaluateCycleNoRegression:
    """Adding diagnostic keys must not perturb existing metric values."""

    def test_existing_keys_unchanged(self):
        # Compute metrics; then manually verify a sample of well-known existing
        # keys still match the documented evaluate_cycle contract.
        gt = _make_ground_truth(50)
        pool = _make_pool_df(50, seed=5)
        sel = _make_selection(gt, top_k=3)
        labeled = sel.clone()

        result = evaluate_cycle(
            cycle=3,
            predictions=np.array(labeled['Activity'].to_list()),
            ground_truth=np.array(labeled['Activity'].to_list()),
            labeled_data=labeled,
            selected_compounds=sel,
            target_col='Activity',
            ground_truth_data=gt,
            pool_df=pool,
            cumulative_selected_ids=set(sel['ID'].to_list()),
        )

        # Cycle, batch_size, cumulative_labeled — invariants of evaluate_cycle.
        assert result['cycle'] == 3
        assert result['batch_size'] == 3
        assert result['cumulative_labeled'] == 3
        # avg_score_selected is the mean of the selected Activity column (top 3 of 0..1).
        # Selected = top 3 by Activity → values 1.0, 0.9796, 0.9592 (linspace 0..1, 50 pts).
        assert result['avg_score_selected'] is not None
        assert 0.9 < result['avg_score_selected'] <= 1.0


@pytest.mark.unit
class TestUncertaintyAwareMetrics:
    """Feature 023 FR-011: cycle metrics dict keeps uncertainty_* keys present
    with None values when uncertainty was skipped."""

    def test_diagnostic_uq_metrics_return_none_when_uncertainty_is_none(self):
        labeled = pl.DataFrame({
            'ID': ['a', 'b', 'c'],
            'SMILES': ['C', 'CC', 'CCC'],
            'Activity': [0.1, 0.2, 0.3],
        })
        sel = labeled.head(2)
        pool = _make_pool_df(20, seed=7)

        result = evaluate_cycle(
            cycle=2,
            predictions=np.array(labeled['Activity'].to_list()),
            ground_truth=np.array(labeled['Activity'].to_list()),
            labeled_data=labeled,
            selected_compounds=sel,
            target_col='Activity',
            ground_truth_data=None,
            pool_df=pool,
            cumulative_selected_ids=set(sel['ID'].to_list()),
            uncertainties=None,  # << feature 023 contract: skip-path
        )

        # Schema-honest: uncertainty_mean / uncertainty_std keys must be
        # present (no fabricated nulls) but None-valued (FR-011).
        assert 'uncertainty_mean' in result
        assert 'uncertainty_std' in result
        assert result['uncertainty_mean'] is None
        assert result['uncertainty_std'] is None

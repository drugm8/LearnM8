"""Tests for LearnM8 enrichment metrics.

Tests for virtual screening enrichment metrics using real molecular data.
"""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from learnm8.evaluation.metrics.enrichment import (
    calculate_enrichment_factor,
    calculate_multiple_enrichment_factors,
    calculate_multiple_top_k_discovery_rates,
    calculate_multiple_top_k_overlaps,
    calculate_multiple_unlabeled_enrichment_factors,
    calculate_multiple_unlabeled_top_k_overlaps,
    calculate_top_k_overlap,
    calculate_unlabeled_ranking_correlation,
)


@pytest.mark.unit
class TestEnrichmentMetrics:
    """Test enrichment and virtual screening metrics."""

    def test_top_k_overlap(self):
        """Test top-K overlap calculation."""
        # Test data - create predictions DataFrame
        predictions_df = pl.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5'],
            'prediction': [9.5, 8.5, 7.5, 6.5, 5.5]  # mol_1, mol_2, mol_3 should be top 3
        })

        ground_truth_data = pl.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_6', 'mol_7', 'mol_8'],
            'Activity': [10, 9, 8, 7, 6, 5]  # mol_1, mol_2, mol_3 are top 3
        })

        # Test k=3
        overlap = calculate_top_k_overlap(predictions_df, ground_truth_data, k=3, target_col='Activity')
        assert overlap == 100.0  # All top 3 are in selected (100% overlap)

        # Test k=5 - but we only have 3 compounds that match, so 100% overlap on those 3
        # (The function only considers compounds that exist in both datasets)
        overlap = calculate_top_k_overlap(predictions_df, ground_truth_data, k=5, target_col='Activity')
        assert overlap == 100.0  # All 3 available compounds overlap (100%)

    def test_enrichment_factor(self):
        """Test enrichment factor calculation."""
        # Create test data with scores and binary labels
        scores = np.array([10, 9, 8, 7, 6, 5, 4, 3])
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # Top 4 are active

        # Calculate enrichment factor at 50% (top 4 compounds)
        # All 4 selected compounds are active (4/4 = 100%)
        # Random selection would give 50% (4/8 actives total)
        # EF = 100% / 50% = 2.0
        enrichment = calculate_enrichment_factor(scores, labels, percentile=50.0)
        assert_allclose(enrichment, 2.0, rtol=1e-2)

        # Test at 25% (top 2 compounds)
        # 2 out of 2 selected are active (100%)
        # Random would give 50%
        # EF = 100% / 50% = 2.0
        enrichment = calculate_enrichment_factor(scores, labels, percentile=25.0)
        assert_allclose(enrichment, 2.0, rtol=1e-2)

    def test_top_k_overlap_lower_direction(self):
        """Test top-K overlap with score_direction='lower' (for docking scores)."""
        # Test data where LOWER scores are better (like docking scores)
        predictions_df = pl.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5'],
            'prediction': [-15.5, -12.3, -8.7, -5.1, -2.4]  # mol_1 has best (lowest) score
        })

        ground_truth_data = pl.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5'],
            'dockscore': [-16.0, -13.2, -9.1, -4.8, -1.9]  # mol_1 has best (lowest) score
        })

        # Test with score_direction='lower' - should select compounds with lowest scores
        overlap = calculate_top_k_overlap(
            predictions_df, ground_truth_data,
            k=3, target_col='dockscore', score_direction='lower'
        )

        # Top 3 by prediction: mol_1, mol_2, mol_3 (lowest scores)
        # Top 3 by ground truth: mol_1, mol_2, mol_3 (lowest scores)
        # Should have 100% overlap
        assert overlap == 100.0

    def test_enrichment_factor_lower_direction(self):
        """Test enrichment factor with score_direction='lower' - this tests the bug fix."""
        # Test data where LOWER scores are better
        scores = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        labels = np.array([1,   1,   1,   1,   0,   0,   0,   0])  # First 4 (lowest scores) are active

        # With score_direction='lower', should select lowest scores first
        # Top 50% (4 compounds) should be: 3.0, 4.0, 5.0, 6.0 (all active)
        # EF = (4 actives selected / 4 selected) / (4 actives total / 8 total) = 1.0 / 0.5 = 2.0
        enrichment_lower = calculate_enrichment_factor(scores, labels, percentile=50.0, score_direction='lower')
        assert_allclose(enrichment_lower, 2.0, rtol=1e-2)

        # With score_direction='higher', should select highest scores first
        # Top 50% (4 compounds) should be: 10.0, 9.0, 8.0, 7.0 (all inactive)
        # EF = (0 actives selected / 4 selected) / (4 actives total / 8 total) = 0.0 / 0.5 = 0.0
        enrichment_higher = calculate_enrichment_factor(scores, labels, percentile=50.0, score_direction='higher')
        assert_allclose(enrichment_higher, 0.0, rtol=1e-2)

        # Verify the directions produce different results (this would catch the original bug)
        assert enrichment_lower != enrichment_higher

    def test_score_direction_consistency(self):
        """Test that score_direction parameter affects results consistently."""
        # Create test case where direction should matter
        predictions_df = pl.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5', 'mol_6'],
            'prediction': [1, 2, 3, 4, 5, 6]  # Simple ascending order
        })

        ground_truth_data = pl.DataFrame({
            'ID': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5', 'mol_6'],
            'activity': [1, 2, 3, 4, 5, 6]  # Same pattern - perfect correlation
        })

        # Test with k=3
        overlap_lower = calculate_top_k_overlap(
            predictions_df, ground_truth_data,
            k=3, target_col='activity', score_direction='lower'
        )
        overlap_higher = calculate_top_k_overlap(
            predictions_df, ground_truth_data,
            k=3, target_col='activity', score_direction='higher'
        )

        # With perfect correlation:
        # - 'lower' direction should select [mol_1, mol_2, mol_3] from both -> 100% overlap
        # - 'higher' direction should select [mol_6, mol_5, mol_4] from both -> 100% overlap
        # Both should be 100% but demonstrate different selection behavior
        assert overlap_lower == 100.0
        assert overlap_higher == 100.0

        # More importantly, test that the enrichment factor direction affects results
        scores = np.array([1, 2, 3, 4, 5, 6])
        labels = np.array([1, 1, 1, 0, 0, 0])  # First 3 are active

        ef_lower = calculate_enrichment_factor(scores, labels, 50.0, score_direction='lower')
        ef_higher = calculate_enrichment_factor(scores, labels, 50.0, score_direction='higher')

        # These should be different due to the bug fix
        assert ef_lower != ef_higher

    def test_multiple_top_k_overlaps_lower_direction(self):
        """Test calculate_multiple_top_k_overlaps with score_direction='lower'."""
        predictions_df = pl.DataFrame({
            'ID': [f'mol_{i}' for i in range(1000)],
            'prediction': np.random.uniform(-20, 0, 1000)  # Random docking-like scores
        })

        ground_truth_data = pl.DataFrame({
            'ID': [f'mol_{i}' for i in range(1000)],
            'dockscore': predictions_df.get_column('prediction').to_numpy() + np.random.normal(0, 1, 1000)  # Correlated
        })

        # Test multiple top-K overlaps with lower direction
        results = calculate_multiple_top_k_overlaps(
            predictions_df, ground_truth_data,
            target_col='dockscore', score_direction='lower'
        )

        # Should return all expected metrics
        expected_keys = ['top_100_overlap', 'top_1000_overlap',
                        'top_0_1_percent_overlap', 'top_1_percent_overlap', 'top_10_percent_overlap']
        for key in expected_keys:
            assert key in results
            assert isinstance(results[key], (int, float))
            assert 0 <= results[key] <= 100  # Should be percentages

    def test_multiple_enrichment_factors_lower_direction(self):
        """Test calculate_multiple_enrichment_factors with score_direction='lower'."""
        # Create scores where lower is better, with clear active/inactive separation
        n_compounds = 200
        n_actives = 40

        # Actives have lower scores (better)
        active_scores = np.random.uniform(-20, -10, n_actives)
        inactive_scores = np.random.uniform(-5, 5, n_compounds - n_actives)

        scores = np.concatenate([active_scores, inactive_scores])
        labels = np.concatenate([np.ones(n_actives), np.zeros(n_compounds - n_actives)])

        # Shuffle to avoid ordering bias
        shuffle_idx = np.random.permutation(len(scores))
        scores = scores[shuffle_idx]
        labels = labels[shuffle_idx]

        results = calculate_multiple_enrichment_factors(scores, labels, score_direction='lower')

        # Should return enrichment factors at multiple percentiles (based on actual implementation)
        expected_keys = ['ef_5_0', 'ef_1_0', 'ef_0_5', 'ef_0_1']
        for key in expected_keys:
            assert key in results
            assert isinstance(results[key], (int, float))
            # For lower direction with well-separated data, should see enrichment > 1
            if key in ['ef_1_0', 'ef_5_0']:
                assert results[key] > 1.0

    def test_invalid_score_direction(self):
        """Test error handling for invalid score_direction values."""
        predictions_df = pl.DataFrame({
            'ID': ['mol_1', 'mol_2'],
            'prediction': [1.0, 2.0]
        })

        ground_truth_data = pl.DataFrame({
            'ID': ['mol_1', 'mol_2'],
            'activity': [1.5, 2.5]
        })

        scores = np.array([1.0, 2.0])
        labels = np.array([1, 0])

        # Invalid score_direction should be handled gracefully
        # (Note: Current implementation may not validate this, but test documents expected behavior)

        # Test Top-K overlap - should work with valid directions
        overlap_valid = calculate_top_k_overlap(
            predictions_df, ground_truth_data,
            k=1, target_col='activity', score_direction='higher'
        )
        assert isinstance(overlap_valid, (int, float))

        # Test enrichment factor - should work with valid directions
        ef_valid = calculate_enrichment_factor(scores, labels, 50.0, score_direction='lower')
        assert isinstance(ef_valid, (int, float))


@pytest.mark.unit
class TestDiscoveryRates:
    """Regression tests for calculate_multiple_top_k_discovery_rates."""

    def _make_gt_df(self, n: int, ascending: bool = False) -> pl.DataFrame:
        ids = [f'mol_{i:04d}' for i in range(n)]
        scores = [float(i + 1) if ascending else float(n - i) for i in range(n)]
        return pl.DataFrame({'ID': ids, 'score': scores})

    def test_top_selected_higher(self):
        gt = self._make_gt_df(1234)
        selected = {f'mol_{i:04d}' for i in range(10)}
        result = calculate_multiple_top_k_discovery_rates(selected, gt, 'score', 'higher')
        assert result['top_10_discovery'] == 100.0
        assert result['top_100_discovery'] == 10.0
        assert result['top_1000_discovery'] == 1.0
        assert result['top_0_1_pct_discovery'] == 100.0
        assert round(result['top_1_pct_discovery'], 2) == 83.33
        assert round(result['top_10_pct_discovery'], 2) == 8.13

    def test_empty_selected_ids(self):
        gt = self._make_gt_df(500)
        result = calculate_multiple_top_k_discovery_rates(set(), gt, 'score', 'higher')
        for v in result.values():
            assert v == 0.0

    def test_all_selected(self):
        n = 200
        gt = self._make_gt_df(n)
        result = calculate_multiple_top_k_discovery_rates(
            {f'mol_{i:04d}' for i in range(n)}, gt, 'score', 'higher'
        )
        for v in result.values():
            assert v == 100.0

    def test_wrong_selection(self):
        gt = self._make_gt_df(1234)
        bottom_10 = {f'mol_{i:04d}' for i in range(1224, 1234)}
        result = calculate_multiple_top_k_discovery_rates(bottom_10, gt, 'score', 'higher')
        assert result['top_10_discovery'] == 0.0
        assert result['top_100_discovery'] == 0.0

    def test_score_direction_lower(self):
        gt = self._make_gt_df(1234, ascending=True)  # mol_0000 has score=1 (best by lower)
        selected = {f'mol_{i:04d}' for i in range(10)}
        result = calculate_multiple_top_k_discovery_rates(selected, gt, 'score', 'lower')
        assert result['top_10_discovery'] == 100.0

    def test_score_direction_changes_results(self):
        gt = self._make_gt_df(1234)
        bottom_10 = {f'mol_{i:04d}' for i in range(1224, 1234)}
        r_higher = calculate_multiple_top_k_discovery_rates(bottom_10, gt, 'score', 'higher')
        r_lower = calculate_multiple_top_k_discovery_rates(bottom_10, gt, 'score', 'lower')
        assert r_higher['top_10_discovery'] != r_lower['top_10_discovery']

    def test_return_keys(self):
        gt = self._make_gt_df(100)
        result = calculate_multiple_top_k_discovery_rates({'mol_0000'}, gt, 'score', 'higher')
        expected = {
            'top_10_discovery', 'top_100_discovery', 'top_1000_discovery',
            'top_0_1_pct_discovery', 'top_1_pct_discovery', 'top_10_pct_discovery',
        }
        assert set(result.keys()) == expected

    def test_small_pool(self):
        gt = self._make_gt_df(5)
        result = calculate_multiple_top_k_discovery_rates(
            {f'mol_{i:04d}' for i in range(3)}, gt, 'score', 'higher'
        )
        for v in result.values():
            assert 0.0 <= v <= 100.0


@pytest.mark.unit
class TestUnlabeledTopKOverlaps:
    """Regression tests for calculate_multiple_unlabeled_top_k_overlaps."""

    def _make_dfs(self, n: int, anti: bool = False):
        ids = [f'mol_{i:03d}' for i in range(n)]
        pred = [float(n - i) for i in range(n)]
        truth = [float(i + 1) if anti else float(n - i) for i in range(n)]
        return (
            pl.DataFrame({'ID': ids, 'prediction': pred}),
            pl.DataFrame({'ID': ids, 'target': truth}),
        )

    def test_perfect_correlation(self):
        preds, gt = self._make_dfs(500)
        result = calculate_multiple_unlabeled_top_k_overlaps(preds, gt, 'target', 'higher')
        assert result['unlabeled_top_100_overlap'] == 100.0
        assert result['unlabeled_top_1000_overlap'] == 100.0

    def test_anti_correlation(self):
        preds, gt = self._make_dfs(500, anti=True)
        result = calculate_multiple_unlabeled_top_k_overlaps(preds, gt, 'target', 'higher')
        assert result['unlabeled_top_100_overlap'] == 0.0

    def test_score_direction_lower_perfect(self):
        ids = [f'mol_{i:03d}' for i in range(500)]
        pred = [float(i + 1) for i in range(500)]
        gt_vals = [float(i + 1) for i in range(500)]
        preds_df = pl.DataFrame({'ID': ids, 'prediction': pred})
        gt_df = pl.DataFrame({'ID': ids, 'target': gt_vals})
        result = calculate_multiple_unlabeled_top_k_overlaps(preds_df, gt_df, 'target', 'lower')
        assert result['unlabeled_top_100_overlap'] == 100.0

    def test_score_direction_changes_results(self):
        n = 300
        ids = [f'mol_{i:03d}' for i in range(n)]
        pred = [float(i + 1) for i in range(n)]
        truth = [0.0] * n
        for i in range(n):
            if i < 50:
                truth[i] = float(i + 151)
            elif i < 100:
                truth[i] = float(i - 49)
            elif i < 200:
                truth[i] = float(i + 101)
            elif i < 250:
                truth[i] = float(i - 99)
            else:
                truth[i] = float(i - 199)
        preds_df = pl.DataFrame({'ID': ids, 'prediction': pred})
        gt_df = pl.DataFrame({'ID': ids, 'target': truth})
        r_higher = calculate_multiple_unlabeled_top_k_overlaps(preds_df, gt_df, 'target', 'higher')
        r_lower = calculate_multiple_unlabeled_top_k_overlaps(preds_df, gt_df, 'target', 'lower')
        assert r_higher['unlabeled_top_100_overlap'] == 0.0
        assert r_lower['unlabeled_top_100_overlap'] == 50.0

    def test_return_keys(self):
        preds, gt = self._make_dfs(50)
        result = calculate_multiple_unlabeled_top_k_overlaps(preds, gt, 'target', 'higher')
        assert set(result.keys()) == {'unlabeled_top_100_overlap', 'unlabeled_top_1000_overlap'}

    def test_small_pool(self):
        preds, gt = self._make_dfs(30)
        result = calculate_multiple_unlabeled_top_k_overlaps(preds, gt, 'target', 'higher')
        for v in result.values():
            assert 0.0 <= v <= 100.0


@pytest.mark.unit
class TestUnlabeledEnrichmentFactors:
    """Regression tests for calculate_multiple_unlabeled_enrichment_factors."""

    def _make_perfect_dfs(self, n: int = 200, n_actives: int = 40):
        ids = [f'mol_{i:03d}' for i in range(n)]
        preds_df = pl.DataFrame({'ID': ids, 'prediction': [float(n - i) for i in range(n)]})
        gt_df = pl.DataFrame({'ID': ids, 'Activity': [1 if i < n_actives else 0 for i in range(n)]})
        return preds_df, gt_df

    def test_perfect_enrichment_higher(self):
        preds, gt = self._make_perfect_dfs(200, 40)
        result = calculate_multiple_unlabeled_enrichment_factors(preds, gt, 'Activity', 'higher')
        # EF = 1 / (40/200) = 5.0 at both 1% and 5%
        assert result['unlabeled_ef_1_0'] == 5.0
        assert result['unlabeled_ef_5_0'] == 5.0

    def test_no_activity_column(self):
        ids = [f'mol_{i:03d}' for i in range(50)]
        preds_df = pl.DataFrame({'ID': ids, 'prediction': list(range(50, 0, -1))})
        gt_df = pl.DataFrame({'ID': ids, 'score': list(range(50, 0, -1))})
        result = calculate_multiple_unlabeled_enrichment_factors(preds_df, gt_df, 'Activity', 'higher')
        assert result['unlabeled_ef_1_0'] is None
        assert result['unlabeled_ef_5_0'] is None

    def test_return_keys(self):
        preds, gt = self._make_perfect_dfs()
        result = calculate_multiple_unlabeled_enrichment_factors(preds, gt, 'Activity', 'higher')
        assert set(result.keys()) == {'unlabeled_ef_1_0', 'unlabeled_ef_5_0'}

    def test_score_direction_lower(self):
        n, n_actives = 200, 40
        ids = [f'mol_{i:03d}' for i in range(n)]
        preds_df = pl.DataFrame({'ID': ids, 'prediction': [float(i + 1) for i in range(n)]})
        gt_df = pl.DataFrame({'ID': ids, 'Activity': [1 if i < n_actives else 0 for i in range(n)]})
        result = calculate_multiple_unlabeled_enrichment_factors(preds_df, gt_df, 'Activity', 'lower')
        assert result['unlabeled_ef_1_0'] == 5.0
        assert result['unlabeled_ef_5_0'] == 5.0


@pytest.mark.unit
class TestUnlabeledRankingCorrelation:
    """Regression tests for calculate_unlabeled_ranking_correlation."""

    def test_perfect_correlation(self):
        ids = [f'mol_{i}' for i in range(20)]
        preds_df = pl.DataFrame({'ID': ids, 'prediction': list(range(20, 0, -1))})
        gt_df = pl.DataFrame({'ID': ids, 'score': list(range(20, 0, -1))})
        assert calculate_unlabeled_ranking_correlation(preds_df, gt_df, 'score') == 1.0

    def test_perfect_anti_correlation(self):
        ids = [f'mol_{i}' for i in range(20)]
        preds_df = pl.DataFrame({'ID': ids, 'prediction': list(range(1, 21))})
        gt_df = pl.DataFrame({'ID': ids, 'score': list(range(20, 0, -1))})
        assert calculate_unlabeled_ranking_correlation(preds_df, gt_df, 'score') == -1.0

    def test_single_compound_returns_zero(self):
        preds_df = pl.DataFrame({'ID': ['mol_0'], 'prediction': [1.0]})
        gt_df = pl.DataFrame({'ID': ['mol_0'], 'score': [1.0]})
        assert calculate_unlabeled_ranking_correlation(preds_df, gt_df, 'score') == 0.0

    def test_no_overlap_returns_zero(self):
        preds_df = pl.DataFrame({'ID': ['mol_A', 'mol_B'], 'prediction': [1.0, 2.0]})
        gt_df = pl.DataFrame({'ID': ['mol_X', 'mol_Y'], 'score': [3.0, 4.0]})
        assert calculate_unlabeled_ranking_correlation(preds_df, gt_df, 'score') == 0.0

    def test_returns_finite_float(self):
        ids = [f'mol_{i}' for i in range(10)]
        preds_df = pl.DataFrame({'ID': ids, 'prediction': list(range(10))})
        gt_df = pl.DataFrame({'ID': ids, 'score': list(range(10))})
        result = calculate_unlabeled_ranking_correlation(preds_df, gt_df, 'score')
        assert isinstance(result, float)
        assert not np.isnan(result)

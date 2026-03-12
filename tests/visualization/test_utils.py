import pytest
import numpy as np
import polars as pl
from learnm8.visualization.utils import (
    downsample_for_viz,
    get_status_colors,
    detect_benchmark_mode,
    format_metric_value
)


@pytest.mark.unit
class TestDownsampleForViz:
    def test_no_downsampling_needed(self):
        data = np.arange(100)
        downsampled, indices = downsample_for_viz(data, max_points=200)

        assert len(downsampled) == 100
        assert len(indices) == 100
        np.testing.assert_array_equal(downsampled, data)
        np.testing.assert_array_equal(indices, np.arange(100))

    def test_downsampling_required(self):
        data = np.arange(10000)
        downsampled, indices = downsample_for_viz(data, max_points=5000)

        assert len(downsampled) == 5000
        assert len(indices) == 5000
        assert np.array_equal(indices, np.sort(indices))

    def test_reproducible_downsampling(self):
        data = np.arange(10000)
        down1, idx1 = downsample_for_viz(data, max_points=1000, random_state=42)
        down2, idx2 = downsample_for_viz(data, max_points=1000, random_state=42)

        np.testing.assert_array_equal(down1, down2)
        np.testing.assert_array_equal(idx1, idx2)


@pytest.mark.unit
class TestGetStatusColors:
    def test_all_unlabeled(self):
        colors = get_status_colors(n_compounds=100, labeled_ids=set(), selected_ids=set())

        assert len(colors) == 100
        assert np.all(colors == 0)

    def test_some_labeled(self):
        labeled_ids = {10, 20, 30}
        colors = get_status_colors(n_compounds=100, labeled_ids=labeled_ids, selected_ids=set())

        assert colors[10] == 1
        assert colors[20] == 1
        assert colors[30] == 1
        assert colors[0] == 0
        assert colors[50] == 0

    def test_some_selected(self):
        selected_ids = {5, 15, 25}
        colors = get_status_colors(n_compounds=100, labeled_ids=set(), selected_ids=selected_ids)

        assert colors[5] == 2
        assert colors[15] == 2
        assert colors[25] == 2
        assert colors[0] == 0

    def test_selected_overrides_labeled(self):
        labeled_ids = {10, 20}
        selected_ids = {10}
        colors = get_status_colors(n_compounds=100, labeled_ids=labeled_ids, selected_ids=selected_ids)

        assert colors[10] == 2
        assert colors[20] == 1


@pytest.mark.unit
class TestDetectBenchmarkMode:
    def test_benchmark_mode_detected(self):
        metrics_df = pl.DataFrame({
            'cycle': [0, 1, 2],
            'top_10_percent_overlap': [0.5, 0.6, 0.7],
            'ef_1_0': [1.5, 2.0, 2.5]
        })

        assert detect_benchmark_mode(metrics_df) is True

    def test_benchmark_mode_not_detected(self):
        metrics_df = pl.DataFrame({
            'cycle': [0, 1, 2],
            'spearman_r': [0.5, 0.6, 0.7],
            'cumulative_labeled': [10, 20, 30]
        })

        assert detect_benchmark_mode(metrics_df) is False

    def test_ground_truth_ef_column(self):
        metrics_df = pl.DataFrame({
            'cycle': [0, 1],
            'ground_truth_ef_1_0': [1.5, 2.0]
        })

        assert detect_benchmark_mode(metrics_df) is True


@pytest.mark.unit
class TestFormatMetricValue:
    def test_format_none(self):
        assert format_metric_value(None, 'any_metric') == 'N/A'

    def test_format_nan(self):
        assert format_metric_value(float('nan'), 'any_metric') == 'N/A'

    def test_format_percent(self):
        assert format_metric_value(50.5, 'top_percent_overlap') == '50.5%'
        assert format_metric_value(75.0, 'overlap_metric') == '75.0%'

    def test_format_enrichment_factor(self):
        assert format_metric_value(2.5, 'enrichment_factor') == '2.50x'
        assert format_metric_value(1.8, 'ef_1_0') == '1.80x'

    def test_format_r2(self):
        assert format_metric_value(0.856, 'r2_score') == '0.856'
        assert format_metric_value(0.95, 'spearman_correlation') == '0.950'

    def test_format_generic(self):
        assert format_metric_value(123.456, 'some_metric') == '123.456'
        assert format_metric_value(0.001, 'generic') == '0.001'

import pytest
import polars as pl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil

from learnm8.visualization.animation import (
    load_csv_data,
    _get_strategy_color_map,
    create_dashboard_animation_from_csv
)


@pytest.fixture
def temp_output_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_data_files(temp_output_dir):
    output_path = Path(temp_output_dir)

    predictions_df = pl.DataFrame({
        'ID': [f'cmp_{i}' for i in range(100)],
        'SMILES': [f'C{i}' for i in range(100)],
        'status': ['labeled' if i < 30 else 'unlabeled' for i in range(100)],
        'labeled_cycle': [i // 10 if i < 30 else None for i in range(100)],
        'selected_cycle': [i // 10 if i < 30 else None for i in range(100)],
        'Activity': np.random.randn(100),
        'prediction_cycle_0': np.random.randn(100),
        'prediction_cycle_1': np.random.randn(100),
        'prediction_cycle_2': np.random.randn(100),
        'uncertainty_cycle_0': np.abs(np.random.randn(100)),
        'uncertainty_cycle_1': np.abs(np.random.randn(100)),
        'uncertainty_cycle_2': np.abs(np.random.randn(100)),
    })
    predictions_df.write_csv(output_path / 'compounds_final.csv')

    metrics_df = pl.DataFrame({
        'cycle': [0, 1, 2],
        'cumulative_labeled': [10, 20, 30],
        'top_10_discovery': [0.0, 20.0, 40.0],
        'top_100_discovery': [0.0, 10.0, 20.0],
        'top_0_1_pct_discovery': [0.0, 5.0, 10.0],
        'top_1_pct_discovery': [0.0, 15.0, 25.0],
        'batch_avg_score_ratio': [1.0, 1.2, 1.5],
        'cumulative_avg_score_ratio': [1.0, 1.1, 1.3],
        'unlabeled_spearman_correlation': [0.5, 0.6, 0.7],
        'unlabeled_top_1000_overlap': [30.0, 40.0, 50.0],
        'unlabeled_top_100_overlap': [20.0, 30.0, 40.0],
    })
    metrics_df.write_csv(output_path / 'cycle_metrics.csv')

    selections_df = pl.DataFrame({
        'ID': [f'cmp_{i}' for i in range(30)],
        'cycle': [i // 10 for i in range(30)],
        'strategy': ['random'] * 10 + ['greedy'] * 10 + ['ucb'] * 10,
        'prediction_at_selection': np.random.randn(30),
        'uncertainty_at_selection': np.abs(np.random.randn(30)),
        'measured_value': np.random.randn(30),
    })
    selections_df.write_csv(output_path / 'selection_history.csv')

    return temp_output_dir


@pytest.mark.unit
class TestLoadCsvData:
    def test_load_all_files(self, sample_data_files):
        data = load_csv_data(sample_data_files)

        assert 'predictions' in data
        assert 'metrics' in data
        assert 'selections' in data

        assert isinstance(data['predictions'], pl.DataFrame)
        assert isinstance(data['metrics'], pl.DataFrame)
        assert isinstance(data['selections'], pl.DataFrame)

        assert data['predictions'].height == 100
        assert data['metrics'].height == 3
        assert data['selections'].height == 30

    def test_missing_predictions_file(self, temp_output_dir):
        with pytest.raises(FileNotFoundError, match='compounds_final.csv'):
            load_csv_data(temp_output_dir)

    def test_missing_metrics_file(self, temp_output_dir):
        predictions_df = pl.DataFrame({'ID': [1, 2, 3]})
        predictions_df.write_csv(Path(temp_output_dir) / 'compounds_final.csv')

        with pytest.raises(FileNotFoundError, match='cycle_metrics.csv'):
            load_csv_data(temp_output_dir)

    def test_missing_selections_file(self, temp_output_dir):
        predictions_df = pl.DataFrame({'ID': [1, 2, 3]})
        predictions_df.write_csv(Path(temp_output_dir) / 'compounds_final.csv')

        metrics_df = pl.DataFrame({'cycle': [0, 1]})
        metrics_df.write_csv(Path(temp_output_dir) / 'cycle_metrics.csv')

        data = load_csv_data(temp_output_dir)

        assert data['selections'].height == 0


@pytest.mark.unit
class TestGetStrategyColorMap:
    def test_standard_strategies(self):
        strategies = ['random', 'greedy', 'ucb']
        color_map = _get_strategy_color_map(strategies)

        assert color_map['random'] == 'yellow'
        assert color_map['greedy'] == 'red'
        assert color_map['ucb'] == 'orange'

    def test_unknown_strategies_are_assigned_colors_in_map(self):
        strategies = ['custom_strategy_1', 'custom_strategy_2']
        color_map = _get_strategy_color_map(strategies)

        assert 'custom_strategy_1' in color_map
        assert 'custom_strategy_2' in color_map

    def test_mixed_strategies(self):
        strategies = ['greedy', 'custom_strategy', 'diverse']
        color_map = _get_strategy_color_map(strategies)

        assert color_map['greedy'] == 'red'
        assert color_map['diverse'] == 'blue'
        assert 'custom_strategy' in color_map


@pytest.mark.unit
class TestCreateDashboardAnimation:
    @pytest.mark.slow
    def test_create_animation_gif(self, sample_data_files, temp_output_dir):
        output_file = str(Path(temp_output_dir) / 'test_animation.gif')

        anim = create_dashboard_animation_from_csv(
            output_dir=sample_data_files,
            output_file=output_file,
            format='gif',
            fps=1,
            dpi=50
        )

        assert anim is not None
        assert Path(output_file).exists()

    @pytest.mark.slow
    def test_create_animation_no_output(self, sample_data_files):
        anim = create_dashboard_animation_from_csv(
            output_dir=sample_data_files,
            output_file=None,
            fps=1
        )

        assert anim is not None
        plt.close('all')

    def test_missing_prediction_columns(self, temp_output_dir):
        predictions_df = pl.DataFrame({
            'ID': [1, 2, 3],
            'SMILES': ['C1', 'C2', 'C3'],
        })
        predictions_df.write_csv(Path(temp_output_dir) / 'compounds_final.csv')

        metrics_df = pl.DataFrame({'cycle': [0, 1]})
        metrics_df.write_csv(Path(temp_output_dir) / 'cycle_metrics.csv')

        with pytest.raises(ValueError, match='No prediction cycles found'):
            create_dashboard_animation_from_csv(temp_output_dir)

    def test_empty_selections(self, temp_output_dir):
        predictions_df = pl.DataFrame({
            'ID': [f'cmp_{i}' for i in range(10)],
            'SMILES': [f'C{i}' for i in range(10)],
            'status': ['unlabeled'] * 10,
            'Activity': np.random.randn(10),
            'prediction_cycle_0': np.random.randn(10),
            'uncertainty_cycle_0': np.abs(np.random.randn(10)),
        })
        predictions_df.write_csv(Path(temp_output_dir) / 'compounds_final.csv')

        metrics_df = pl.DataFrame({
            'cycle': [0],
            'cumulative_labeled': [0],
        })
        metrics_df.write_csv(Path(temp_output_dir) / 'cycle_metrics.csv')

        anim = create_dashboard_animation_from_csv(temp_output_dir)
        assert anim is not None
        plt.close('all')

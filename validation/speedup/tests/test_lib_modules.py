from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from learnm8.core.interfaces import Learner
from validation.speedup.lib.assertions import (
    assert_predictions_match,
    assert_uncertainty_ranking_preserved,
)
from validation.speedup.lib.config import (
    DATASET_ENV_VAR,
    RESULTS_DIR_ENV_VAR,
    TIMED_ENV_VAR,
    WARMUP_ENV_VAR,
    get_benchmark_dataset_path,
)
from validation.speedup.lib.data_setup import generate_synthetic_data
from validation.speedup.lib.model_factory import create_learner
from validation.speedup.lib.reporting import create_result, save_result
from validation.speedup.lib.timing import time_prediction, time_training


class TestGenerateSyntheticData:
    def test_output_shapes(self):
        X_train, y_train, X_test, y_test = generate_synthetic_data()
        assert X_train.shape == (800, 1024)
        assert y_train.shape == (800,)
        assert X_test.shape == (200, 1024)
        assert y_test.shape == (200,)

    def test_features_dtype(self):
        X_train, _, X_test, _ = generate_synthetic_data()
        assert X_train.dtype == np.float64
        assert X_test.dtype == np.float64

    def test_features_are_binary(self):
        X_train, _, X_test, _ = generate_synthetic_data()
        assert set(np.unique(X_train)).issubset({0.0, 1.0})
        assert set(np.unique(X_test)).issubset({0.0, 1.0})

    def test_different_seeds_produce_different_data(self):
        X1, _, _, _ = generate_synthetic_data(seed=42)
        X2, _, _, _ = generate_synthetic_data(seed=99)
        assert not np.array_equal(X1, X2)

    def test_density_approximately_correct(self):
        X_train, _, X_test, _ = generate_synthetic_data(density=0.05, seed=42)
        X_all = np.vstack([X_train, X_test])
        observed_density = X_all.mean()
        assert abs(observed_density - 0.05) < 0.02


class TestCreateLearner:
    def test_returns_learner_for_rf(self):
        learner = create_learner("rf")
        assert isinstance(learner, Learner)

    def test_returns_learner_for_dt(self):
        learner = create_learner("dt")
        assert isinstance(learner, Learner)

    def test_returns_learner_for_lr(self):
        learner = create_learner("lr")
        assert isinstance(learner, Learner)

    def test_raises_for_unknown_learner(self):
        with pytest.raises(ValueError, match="Unknown learner type"):
            create_learner("nonexistent_learner")


class TestTimePrediction:
    def test_return_keys(self):
        result = time_prediction(lambda: None, n_warmup=1, n_runs=3)
        assert set(result.keys()) == {"median_time", "iqr", "times", "n_warmup", "n_runs"}

    def test_times_length_matches_n_runs(self):
        n_runs = 5
        result = time_prediction(lambda: None, n_warmup=1, n_runs=n_runs)
        assert len(result["times"]) == n_runs

    def test_median_time_positive(self):
        result = time_prediction(lambda: time.sleep(0.001), n_warmup=1, n_runs=3)
        assert result["median_time"] > 0

    def test_reads_run_counts_from_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(WARMUP_ENV_VAR, '2')
        monkeypatch.setenv(TIMED_ENV_VAR, '4')

        result = time_prediction(lambda: None)

        assert result['n_warmup'] == 2
        assert result['n_runs'] == 4


class TestTimeTraining:
    def test_return_keys(self):
        result = time_training(lambda: None, n_warmup=1, n_runs=3)
        assert set(result.keys()) == {'median_time', 'iqr', 'times', 'n_warmup', 'n_runs'}

    def test_times_length_matches_n_runs(self):
        n_runs = 4
        result = time_training(lambda: None, n_warmup=1, n_runs=n_runs)
        assert len(result['times']) == n_runs


class TestAssertPredictionsMatch:
    def test_identical_arrays_pass_zero_impact(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert_predictions_match(arr, arr.copy(), "zero_impact")

    def test_very_different_arrays_fail_zero_impact(self):
        baseline = np.array([1.0, 2.0, 3.0])
        optimized = np.array([100.0, 200.0, 300.0])
        with pytest.raises(AssertionError):
            assert_predictions_match(baseline, optimized, "zero_impact")


class TestAssertUncertaintyRankingPreserved:
    def test_identical_arrays_pass(self):
        arr = np.array([0.1, 0.5, 0.3, 0.8])
        assert_uncertainty_ranking_preserved(arr, arr.copy())

    def test_reversed_arrays_fail(self):
        baseline = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        reversed_arr = baseline[::-1].copy()
        with pytest.raises(AssertionError):
            assert_uncertainty_ranking_preserved(baseline, reversed_arr)


class TestReporting:
    def test_create_result_normalizes_prediction_rho_alias(self):
        result = create_result(
            optimization_id=1,
            optimization_name='test_opt',
            learner_name='rf',
            tier=0,
            baseline_time=2.0,
            optimized_time=1.0,
            prediction_match=True,
            spearman_rho=0.99,
        )

        assert result['prediction_spearman_rho'] == pytest.approx(0.99)
        assert result['spearman_rho'] == pytest.approx(0.99)
        assert result['uncertainty_spearman_rho'] is None

    def test_save_result_normalizes_legacy_uncertainty_key(self, tmp_path: Path):
        result = create_result(
            optimization_id=1,
            optimization_name='test_opt',
            learner_name='rf',
            tier=0,
            baseline_time=2.0,
            optimized_time=1.0,
            prediction_match=True,
            spearman_rho=0.99,
        )
        result['uncertainty_spearman'] = 0.97

        save_result(result, tmp_path)

        assert result['uncertainty_spearman_rho'] == pytest.approx(0.97)

    def test_save_result_uses_env_results_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        result = create_result(
            optimization_id=1,
            optimization_name='test_opt',
            learner_name='rf',
            tier=0,
            baseline_time=2.0,
            optimized_time=1.0,
            prediction_match=True,
        )
        env_output_dir = tmp_path / 'env-results'
        monkeypatch.setenv(RESULTS_DIR_ENV_VAR, str(env_output_dir))

        saved_path = save_result(result, tmp_path / 'ignored')

        assert saved_path.parent == env_output_dir
        assert saved_path.exists()

    def test_create_result_preserves_per_run_times(self):
        result = create_result(
            optimization_id=1,
            optimization_name='test_opt',
            learner_name='rf',
            tier=0,
            baseline_time=2.0,
            optimized_time=1.0,
            prediction_match=True,
            baseline_times=[2.1, 2.0, 1.9],
            optimized_times=[1.1, 1.0, 0.9],
            training_baseline_times=[0.5, 0.4],
            training_optimized_times=[0.3, 0.2],
        )

        assert result['baseline_times'] == [2.1, 2.0, 1.9]
        assert result['optimized_times'] == [1.1, 1.0, 0.9]
        assert result['training_baseline_times'] == [0.5, 0.4]
        assert result['training_optimized_times'] == [0.3, 0.2]


class TestBenchmarkConfig:
    def test_dataset_path_reads_environment(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        dataset_path = tmp_path / 'bench.csv'
        monkeypatch.setenv(DATASET_ENV_VAR, str(dataset_path))

        assert get_benchmark_dataset_path() == dataset_path.resolve()



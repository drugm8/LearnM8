"""Tests for the parallel Welford uncertainty helper.

Covers:
  - Numerical equivalence to the baseline ``np.stack().mean()/.std()`` pattern
  - Catastrophic-cancellation stress where naive sum/sum_sq would fail
  - Edge cases (empty / single estimator)
  - n_jobs determinism (within float tolerance)
"""

from __future__ import annotations

import numpy as np
import pytest

from learnm8.learners.sklearn._tree_uncertainty import (
    _resolve_n_workers,
    fused_mean_std,
)


class _ConstantEstimator:
    """Fake estimator that returns a fixed prediction vector.

    Lets us test the helper without standing up a full RandomForestRegressor.
    """

    def __init__(self, value: np.ndarray):
        self._value = np.asarray(value, dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Return one prediction per row, ignoring X content but using its shape.
        if X.shape[0] != self._value.shape[0]:
            raise AssertionError('shape mismatch in test estimator')
        return self._value.copy()


class _RecordingEstimator(_ConstantEstimator):
    """Fake estimator that records the array object handed to ``predict``."""

    def __init__(self, value: np.ndarray, seen: list[tuple[np.dtype, int]]):
        super().__init__(value)
        self._seen = seen

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._seen.append((X.dtype, id(X)))
        return super().predict(X)


def _baseline_mean_std(estimators, X):
    """Reference: stack all predictions in float64, take mean and ddof=0 std."""
    preds = np.stack([np.asarray(e.predict(X), dtype=np.float64) for e in estimators])
    return preds.mean(axis=0), preds.std(axis=0, ddof=0)


@pytest.mark.unit
class TestFusedMeanStd:
    """Helper-level tests, no sklearn fit required."""

    def test_matches_baseline_on_random_predictions(self):
        rng = np.random.default_rng(0)
        n, n_est = 5000, 50
        X = rng.standard_normal((n, 4))
        estimators = [_ConstantEstimator(rng.standard_normal(n)) for _ in range(n_est)]

        mean_ref, std_ref = _baseline_mean_std(estimators, X)
        mean, std = fused_mean_std(estimators, X, n_jobs=-1)

        np.testing.assert_allclose(mean, mean_ref, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(std, std_ref, atol=1e-12, rtol=1e-12)

    def test_output_is_float64(self):
        rng = np.random.default_rng(1)
        n = 100
        X = rng.standard_normal((n, 3))
        estimators = [
            _ConstantEstimator(rng.standard_normal(n).astype(np.float32))
            for _ in range(5)
        ]
        mean, std = fused_mean_std(estimators, X, n_jobs=1)

        assert mean.dtype == np.float64
        assert std.dtype == np.float64
        assert mean.shape == (n,)
        assert std.shape == (n,)
        assert np.all(std >= 0.0)

    def test_handles_catastrophic_cancellation_case(self):
        """Tight clusters at large mean magnitude — Welford must stay accurate.

        Naive ``sum(p**2)/n - (sum(p)/n)**2`` breaks here (see benchmarks),
        producing relative errors >5% or NaN. Welford should remain within
        ~1e-7 even in adversarial regimes.
        """
        rng = np.random.default_rng(2)
        n_est, n = 100, 5000
        # mean=1e6, jitter=1e-3 — the adversarial case from the stress benchmark
        preds = 1e6 + 1e-3 * rng.standard_normal((n_est, n))
        estimators = [_ConstantEstimator(p) for p in preds]

        mean_ref = preds.mean(axis=0)
        std_ref = preds.std(axis=0, ddof=0)

        mean, std = fused_mean_std(estimators, np.zeros((n, 1)), n_jobs=-1)

        # Welford should be accurate to ~1e-7 relative even at this scale.
        np.testing.assert_allclose(mean, mean_ref, atol=1e-6, rtol=1e-10)
        np.testing.assert_allclose(std, std_ref, rtol=1e-6, atol=1e-9)

    def test_empty_estimators_raises(self):
        with pytest.raises(ValueError, match='empty'):
            fused_mean_std([], np.zeros((5, 1)), n_jobs=1)

    def test_single_estimator_gives_zero_std(self):
        n = 50
        pred = np.linspace(0.0, 1.0, n)
        estimators = [_ConstantEstimator(pred)]
        mean, std = fused_mean_std(estimators, np.zeros((n, 1)), n_jobs=-1)

        np.testing.assert_array_equal(mean, pred)
        np.testing.assert_array_equal(std, np.zeros(n))

    def test_serial_vs_parallel_within_tolerance(self):
        """n_jobs=1 and n_jobs=-1 should agree to ~1e-12.

        Welford merges are order-dependent at float precision, but both paths
        compute the same mathematical quantity within machine epsilon.
        """
        rng = np.random.default_rng(3)
        n, n_est = 2000, 20
        estimators = [_ConstantEstimator(rng.standard_normal(n)) for _ in range(n_est)]
        X = np.zeros((n, 1))

        mean_serial, std_serial = fused_mean_std(estimators, X, n_jobs=1)
        mean_par, std_par = fused_mean_std(estimators, X, n_jobs=-1)

        np.testing.assert_allclose(mean_par, mean_serial, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(std_par, std_serial, atol=1e-12, rtol=1e-12)

    def test_non_float32_input_is_converted_once_before_the_fan_out(self):
        """Regression: peak memory must not scale with the worker count.

        sklearn's per-tree ``check_array(X, dtype=float32)`` allocates a new
        array whenever the input dtype differs, and ``require='sharedmem'``
        shares only the input — so a uint8 matrix used to be upcast inside
        every worker thread, costing ``n_threads`` full float32 copies.
        Every estimator must instead see the same, already-converted array.
        """
        rng = np.random.default_rng(5)
        n, n_est = 200, 16
        X = rng.integers(0, 2, size=(n, 32)).astype(np.uint8)
        seen: list[tuple[np.dtype, int]] = []
        estimators = [
            _RecordingEstimator(rng.standard_normal(n), seen) for _ in range(n_est)
        ]

        fused_mean_std(estimators, X, n_jobs=4)

        assert len(seen) == n_est
        assert {dtype for dtype, _ in seen} == {np.dtype(np.float32)}
        assert len({obj_id for _, obj_id in seen}) == 1, (
            'estimators saw more than one array object — the float32 '
            'conversion is happening per worker instead of once up front'
        )

    def test_float32_input_is_passed_through_without_a_copy(self):
        """An already-float32 matrix must reach the estimators untouched."""
        rng = np.random.default_rng(6)
        n, n_est = 100, 8
        X = rng.standard_normal((n, 4)).astype(np.float32)
        seen: list[tuple[np.dtype, int]] = []
        estimators = [
            _RecordingEstimator(rng.standard_normal(n), seen) for _ in range(n_est)
        ]

        fused_mean_std(estimators, X, n_jobs=4)

        assert {obj_id for _, obj_id in seen} == {id(X)}

    def test_n_workers_resolution(self):
        # joblib convention: -1 = all cores, -2 = all but one
        import os

        cpu = os.cpu_count() or 1

        assert _resolve_n_workers(1) == 1
        assert _resolve_n_workers(4) == 4
        assert _resolve_n_workers(-1) == cpu
        assert _resolve_n_workers(-2) == max(1, cpu - 1)
        assert _resolve_n_workers(0) == 1
        assert _resolve_n_workers(None) == 1  # type: ignore[arg-type]


@pytest.mark.integration
class TestIntegrationWithSklearnRF:
    """Integration test against a real RandomForestRegressor."""

    def test_matches_naive_pertree_baseline(self):
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.default_rng(4)
        X_train = rng.standard_normal((200, 8))
        y_train = (X_train[:, 0] ** 2 + 0.3 * X_train[:, 1]).astype(np.float32)
        X_pred = rng.standard_normal((300, 8))

        model = RandomForestRegressor(
            n_estimators=25, max_depth=6, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Naive baseline path (what the old code did)
        sklearn_pred = model.predict(X_pred)
        per_tree = np.stack([t.predict(X_pred) for t in model.estimators_])
        baseline_std = per_tree.std(axis=0, ddof=0)
        baseline_mean = per_tree.mean(axis=0)

        # New fused path
        mean, std = fused_mean_std(model.estimators_, X_pred, n_jobs=-1)

        # model.predict() is the mean of estimators_ — confirm equivalence.
        np.testing.assert_allclose(mean, baseline_mean, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(mean, sklearn_pred, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(std, baseline_std, atol=1e-12, rtol=1e-12)

    def test_uint8_input_matches_float32_input_bit_for_bit(self):
        """Hoisting the upcast must not change a single bit of the output.

        Guards against a future attempt to make the conversion cheaper by
        making it lossy (e.g. float16 or an in-place reinterpretation).
        """
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.default_rng(7)
        X_train = rng.integers(0, 2, size=(200, 16)).astype(np.uint8)
        y_train = (X_train[:, 0] * 0.7 + X_train[:, 3] * 0.2).astype(np.float32)
        X_pred = rng.integers(0, 2, size=(300, 16)).astype(np.uint8)

        model = RandomForestRegressor(
            n_estimators=25, max_depth=6, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        mean_u8, std_u8 = fused_mean_std(model.estimators_, X_pred, n_jobs=4)
        mean_f32, std_f32 = fused_mean_std(
            model.estimators_, X_pred.astype(np.float32), n_jobs=4
        )

        np.testing.assert_array_equal(mean_u8, mean_f32)
        np.testing.assert_array_equal(std_u8, std_f32)

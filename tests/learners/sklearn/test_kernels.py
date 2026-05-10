"""Tests for TanimotoKernel implementation."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from learnm8.learners.sklearn.kernels import TanimotoKernel


@pytest.mark.unit
class TestTanimotoKernel:
    """Unit tests for the TanimotoKernel sklearn Kernel subclass."""

    @pytest.fixture
    def kernel(self):
        return TanimotoKernel()

    @pytest.fixture
    def X_binary(self):
        return np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
        ], dtype=float)

    @pytest.fixture
    def X_with_zero(self):
        return np.array([
            [1, 0, 1],
            [0, 0, 0],
            [0, 1, 0],
        ], dtype=float)

    def test_kernel_matrix_binary_vectors(self, kernel, X_binary):
        K = kernel(X_binary)
        x0, x1 = X_binary[0], X_binary[1]
        dot01 = np.dot(x0, x1)
        denom01 = np.sum(x0**2) + np.sum(x1**2) - dot01
        expected_01 = dot01 / denom01
        assert K.shape == (4, 4)
        np.testing.assert_allclose(K[0, 1], expected_01, rtol=1e-10)

    def test_kernel_matrix_symmetric(self, kernel, X_binary):
        K = kernel(X_binary)
        np.testing.assert_allclose(K, K.T, atol=1e-15)

    def test_kernel_matrix_psd(self, kernel, X_binary):
        K = kernel(X_binary)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-10)

    def test_diagonal_equals_one(self, kernel, X_binary):
        diag = kernel.diag(X_binary)
        np.testing.assert_allclose(diag, 1.0, atol=1e-15)

    def test_zero_vector_handling(self, kernel, X_with_zero):
        # Feature 019 FR-014: zero-zero diagonal returns signal_variance=1.0
        # (canonical k(x,x)=1 identity); cross-terms with non-zero rows still 0.
        K = kernel(X_with_zero)
        assert np.all(np.isfinite(K))
        assert K[1, 0] == 0.0
        assert K[0, 1] == 0.0
        assert K[1, 1] == 1.0  # zero-zero pair → signal_variance, was 0.0
        assert K[1, 2] == 0.0

    def test_zero_vector_diag(self, kernel, X_with_zero):
        # Feature 019 FR-014: diag returns signal_variance for every row,
        # including zero-vector rows (was 0.0 before; caused LinAlgError).
        diag = kernel.diag(X_with_zero)
        assert diag[0] == pytest.approx(1.0)
        assert diag[1] == pytest.approx(1.0)
        assert diag[2] == pytest.approx(1.0)

    def test_eval_gradient_shape(self, kernel, X_binary):
        K, grad = kernel(X_binary, eval_gradient=True)
        assert K.shape == (4, 4)
        assert grad.shape == (4, 4, 1)

    def test_eval_gradient_fixed(self, X_binary):
        kernel = TanimotoKernel(signal_variance_bounds="fixed")
        K, grad = kernel(X_binary, eval_gradient=True)
        assert K.shape == (4, 4)
        assert grad.shape == (4, 4, 0)

    def test_eval_gradient_y_not_none_raises(self, kernel, X_binary):
        Y = X_binary[:2]
        with pytest.raises(ValueError, match="eval_gradient"):
            kernel(X_binary, Y=Y, eval_gradient=True)

    def test_gradient_numerical_check(self, kernel, X_binary):
        K, grad_analytic = kernel(X_binary, eval_gradient=True)
        theta = kernel.theta.copy()
        eps = 1e-5
        grad_numerical = np.zeros_like(grad_analytic)
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            kernel_plus = kernel.clone_with_theta(theta_plus)
            K_plus = kernel_plus(X_binary)

            theta_minus = theta.copy()
            theta_minus[i] -= eps
            kernel_minus = kernel.clone_with_theta(theta_minus)
            K_minus = kernel_minus(X_binary)

            grad_numerical[:, :, i] = (K_plus - K_minus) / (2 * eps)

        np.testing.assert_allclose(grad_analytic, grad_numerical, rtol=1e-3)

    def test_is_stationary(self, kernel):
        assert kernel.is_stationary() is False

    def test_hyperparameter_properties(self, kernel):
        assert len(kernel.theta) == 1
        assert kernel.bounds.shape == (1, 2)
        new_theta = np.array([np.log(2.0)])
        new_kernel = kernel.clone_with_theta(new_theta)
        assert new_kernel.signal_variance == pytest.approx(2.0, rel=1e-10)

    def test_get_params_set_params(self, kernel):
        params = kernel.get_params()
        assert "signal_variance" in params
        assert params["signal_variance"] == 1.0

        cloned = clone(kernel)
        assert isinstance(cloned, TanimotoKernel)
        assert cloned.signal_variance == kernel.signal_variance

    def test_signal_variance_scaling(self, X_binary):
        k1 = TanimotoKernel(signal_variance=1.0)
        k2 = TanimotoKernel(signal_variance=3.0)
        K1 = k1(X_binary)
        K2 = k2(X_binary)
        np.testing.assert_allclose(K2, 3.0 * K1, rtol=1e-10)

    def test_gpr_integration(self, kernel, X_binary):
        y = np.array([0.5, 0.3, 0.8, 0.2])
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, random_state=42)
        gpr.fit(X_binary, y)
        preds, std = gpr.predict(X_binary, return_std=True)
        assert preds.shape == (4,)
        assert std.shape == (4,)
        assert np.all(np.isfinite(preds))
        assert np.all(std >= 0)

    def test_repr(self, kernel):
        assert repr(kernel) == "TanimotoKernel(signal_variance=1)"

    def test_is_kernel_subclass(self, kernel):
        assert isinstance(kernel, Kernel)

    def test_xy_not_none(self, kernel, X_binary):
        Y = X_binary[:2]
        K = kernel(X_binary, Y)
        assert K.shape == (4, 2)
        assert np.all(np.isfinite(K))

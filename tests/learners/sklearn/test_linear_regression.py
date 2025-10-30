"""Tests for LinearRegressionLearner implementation."""

import pytest
import numpy as np
import pandas as pd

from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner
from learnm8.features.extraction import extract_features


class TestLinearRegressionLearner:
    """Test LinearRegressionLearner functionality with real molecular data."""

    @pytest.fixture
    def learner(self):
        """Create LinearRegressionLearner instance for testing."""
        return LinearRegressionLearner(random_state=42)

    @pytest.fixture
    def ridge_learner(self):
        """Create Ridge regression learner instance for testing."""
        return LinearRegressionLearner(alpha=1.0, random_state=42)

    def test_initialization(self, learner):
        """Test learner initialization with default Linear Regression mode."""
        assert learner.alpha is None
        assert learner.fit_intercept is True
        assert learner.n_jobs == -1
        assert learner.random_state == 42
        assert not learner.is_trained
        assert learner.supports_uncertainty() is False
        assert learner.is_ridge is False

    def test_initialization_ridge_mode(self, ridge_learner):
        """Test learner initialization with Ridge mode."""
        assert ridge_learner.alpha == 1.0
        assert ridge_learner.fit_intercept is True
        assert ridge_learner.n_jobs is None
        assert ridge_learner.random_state == 42
        assert not ridge_learner.is_trained
        assert ridge_learner.supports_uncertainty() is False
        assert ridge_learner.is_ridge is True

    def test_train_predict_integration(self, learner, small_real_compounds, tmp_path):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        targets = compounds['Activity'].values

        learner.train(features, targets)
        assert learner.is_trained

        predictions, uncertainty = learner.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is None
        assert np.all(np.isfinite(predictions))

    def test_ridge_vs_linear_mode(self, small_real_compounds, tmp_path):
        """Test both Ridge and Linear modes produce valid predictions."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        targets = compounds['Activity'].values

        linear_learner = LinearRegressionLearner(alpha=None, random_state=42)
        linear_learner.train(features, targets)
        linear_predictions, _ = linear_learner.predict(features)

        ridge_learner = LinearRegressionLearner(alpha=1.0, random_state=42)
        ridge_learner.train(features, targets)
        ridge_predictions, _ = ridge_learner.predict(features)

        assert linear_predictions.shape == ridge_predictions.shape
        assert np.all(np.isfinite(linear_predictions))
        assert np.all(np.isfinite(ridge_predictions))
        assert not np.allclose(linear_predictions, ridge_predictions)

    def test_get_coefficients(self, learner, small_real_compounds, tmp_path):
        """Test coefficient retrieval."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        assert learner.get_coefficients() is None

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        coefficients = learner.get_coefficients()

        assert coefficients is not None
        assert len(coefficients) == features.shape[1]
        assert np.all(np.isfinite(coefficients))

    def test_get_intercept(self, learner, small_real_compounds, tmp_path):
        """Test intercept retrieval."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        assert learner.get_intercept() is None

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        intercept = learner.get_intercept()

        assert intercept is not None
        assert isinstance(intercept, (float, np.floating))
        assert np.isfinite(intercept)

    def test_get_intercept_no_intercept_mode(self, small_real_compounds, tmp_path):
        """Test intercept retrieval when fit_intercept=False."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner = LinearRegressionLearner(fit_intercept=False, random_state=42)
        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        intercept = learner.get_intercept()

        assert intercept is not None
        assert isinstance(intercept, (float, np.floating))
        assert np.isclose(intercept, 0.0)

    def test_train_with_empty_arrays(self, learner):
        """Test error handling with empty arrays."""
        empty_features = np.array([]).reshape(0, 10)
        empty_targets = np.array([])

        with pytest.raises(ValueError, match="Cannot train on empty dataset"):
            learner.train(empty_features, empty_targets)

    def test_train_with_mismatched_shapes(self, learner):
        """Test error handling with mismatched feature/target shapes."""
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)

        with pytest.raises(ValueError, match="Features and targets must have same length"):
            learner.train(features, targets)

    def test_predict_without_training(self, learner):
        """Test error when predicting without training."""
        features = np.random.randn(5, 10)
        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            learner.predict(features)

    def test_get_name(self, learner, ridge_learner):
        """Test name generation for both Ridge and Linear modes."""
        linear_name = learner.get_name()
        assert "LinearRegression" in linear_name
        assert "with_intercept" in linear_name

        ridge_name = ridge_learner.get_name()
        assert "Ridge" in ridge_name
        assert "α=1.000" in ridge_name
        assert "with_intercept" in ridge_name

    def test_get_name_no_intercept(self):
        """Test name generation when fit_intercept=False."""
        learner = LinearRegressionLearner(fit_intercept=False, random_state=42)
        name = learner.get_name()
        assert "LinearRegression" in name
        assert "no_intercept" in name

    def test_supports_uncertainty(self, learner, ridge_learner):
        """Test that uncertainty is not supported in either mode."""
        assert learner.supports_uncertainty() is False
        assert ridge_learner.supports_uncertainty() is False

    def test_different_alpha_values(self, tmp_path, small_real_compounds):
        """Test Ridge regression with different alpha values."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        targets = compounds['Activity'].values

        alphas = [0.1, 1.0, 10.0]
        predictions_dict = {}

        for alpha in alphas:
            learner = LinearRegressionLearner(alpha=alpha, random_state=42)
            learner.train(features, targets)
            predictions, _ = learner.predict(features)
            predictions_dict[alpha] = predictions

        for alpha in alphas:
            assert np.all(np.isfinite(predictions_dict[alpha]))

        assert not np.allclose(predictions_dict[0.1], predictions_dict[10.0])

    def test_edge_case_single_compound(self, learner, tmp_path):
        """Test with single compound."""
        single_compound = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        features = extract_features(single_compound['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, single_compound['Activity'].values)
        predictions, _ = learner.predict(features)

        assert len(predictions) == 1
        assert np.isfinite(predictions[0])

    def test_coefficients_ridge_vs_linear(self, small_real_compounds, tmp_path):
        """Test that Ridge produces smaller coefficients than Linear (regularization effect)."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        targets = compounds['Activity'].values

        linear_learner = LinearRegressionLearner(alpha=None, random_state=42)
        linear_learner.train(features, targets)
        linear_coefs = linear_learner.get_coefficients()

        ridge_learner = LinearRegressionLearner(alpha=10.0, random_state=42)
        ridge_learner.train(features, targets)
        ridge_coefs = ridge_learner.get_coefficients()

        linear_l2_norm = np.linalg.norm(linear_coefs)
        ridge_l2_norm = np.linalg.norm(ridge_coefs)

        assert ridge_l2_norm < linear_l2_norm

    def test_n_jobs_parameter(self, small_real_compounds, tmp_path):
        """Test that n_jobs parameter is respected for LinearRegression mode."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        targets = compounds['Activity'].values

        learner_parallel = LinearRegressionLearner(n_jobs=-1, random_state=42)
        assert learner_parallel.n_jobs == -1
        learner_parallel.train(features, targets)
        predictions, _ = learner_parallel.predict(features)
        assert np.all(np.isfinite(predictions))

        learner_single = LinearRegressionLearner(n_jobs=1, random_state=42)
        assert learner_single.n_jobs == 1
        learner_single.train(features, targets)
        predictions_single, _ = learner_single.predict(features)
        assert np.allclose(predictions, predictions_single)

    def test_ridge_mode_no_n_jobs(self, ridge_learner):
        """Test that Ridge mode does not use n_jobs parameter."""
        assert ridge_learner.n_jobs is None

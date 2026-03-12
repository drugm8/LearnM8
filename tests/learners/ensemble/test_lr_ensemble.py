"""Tests for LREnsemble implementation."""

import pytest
import numpy as np
import polars as pl

from learnm8.learners.ensemble.lr_ensemble import LREnsemble
from learnm8.features.extraction import extract_features


@pytest.mark.integration
class TestLREnsemble:
    """Test LREnsemble functionality with real molecular data."""

    @pytest.fixture
    def lr_ensemble(self):
        """Create LREnsemble instance for testing."""
        return LREnsemble()

    def test_initialization(self, lr_ensemble):
        """Test ensemble initialization with default parameters."""
        assert len(lr_ensemble.learners) == 3
        assert lr_ensemble.aggregation_method == 'mean'
        assert lr_ensemble.uncertainty_method == 'std'
        assert lr_ensemble.weights is None
        assert not lr_ensemble.is_trained
        assert lr_ensemble.supports_uncertainty() is True
        assert lr_ensemble.regularization_strengths == [0.1, 1.0, 10.0]
        assert lr_ensemble.random_states == [42, 123, 456]

    def test_initialization_custom_regularization(self):
        """Test ensemble initialization with custom regularization strengths."""
        custom_alphas = [0.01, 0.1, 1.0]
        custom_states = [10, 20, 30]

        ensemble = LREnsemble(
            regularization_strengths=custom_alphas,
            random_states=custom_states
        )

        assert len(ensemble.learners) == 3
        assert ensemble.regularization_strengths == custom_alphas
        assert ensemble.random_states == custom_states

    def test_initialization_with_weights(self):
        """Test ensemble initialization with custom weights."""
        weights = [0.5, 0.3, 0.2]
        ensemble = LREnsemble(weights=weights)

        assert ensemble.weights is not None
        assert np.allclose(ensemble.weights, weights)

    def test_train_predict_integration(self, lr_ensemble, small_real_compounds, tmp_path):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        lr_ensemble.train(features, compounds['Activity'].to_numpy())
        assert lr_ensemble.is_trained

        predictions, uncertainty = lr_ensemble.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_uncertainty_estimation(self, lr_ensemble, small_real_compounds, tmp_path):
        """Test that ensemble provides uncertainty estimates."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        lr_ensemble.train(features, compounds['Activity'].to_numpy())

        predictions, uncertainty = lr_ensemble.predict(features)

        assert uncertainty is not None
        assert len(uncertainty) == len(compounds)
        assert np.all(np.isfinite(uncertainty))
        assert np.all(uncertainty >= 0)
        assert np.std(uncertainty) > 0

    def test_diverse_regularization(self, lr_ensemble, small_real_compounds, tmp_path):
        """Test that ensemble learners have different regularization strengths."""
        assert len(lr_ensemble.learners) == 3

        alphas = [learner.alpha for learner in lr_ensemble.learners]
        assert len(set(alphas)) == 3
        assert alphas == [0.1, 1.0, 10.0]

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        lr_ensemble.train(features, compounds['Activity'].to_numpy())

        individual_preds = lr_ensemble.get_individual_predictions(features)
        assert len(individual_preds) == 3

        pred_arrays = [preds for preds in individual_preds.values() if preds is not None]
        assert len(pred_arrays) == 3

        for i in range(len(pred_arrays)):
            for j in range(i+1, len(pred_arrays)):
                assert not np.allclose(pred_arrays[i], pred_arrays[j], rtol=1e-3)

    def test_train_with_empty_arrays(self, lr_ensemble):
        """Test error handling when training with empty arrays."""
        empty_features = np.array([]).reshape(0, 10)
        empty_targets = np.array([])

        with pytest.raises(ValueError):
            lr_ensemble.train(empty_features, empty_targets)

    def test_predict_without_training(self, lr_ensemble, small_real_compounds, tmp_path):
        """Test error when predicting without training."""
        features = extract_features(small_real_compounds['SMILES'].to_list(), 'morgan', tmp_path)
        with pytest.raises(RuntimeError, match="Ensemble must be trained before prediction"):
            lr_ensemble.predict(features)

    def test_get_name(self, lr_ensemble):
        """Test name generation for LR ensemble."""
        name = lr_ensemble.get_name()
        assert "LREnsemble" in name
        assert "3xRidge" in name
        assert "0.1" in name
        assert "1.0" in name
        assert "10.0" in name

    def test_get_name_custom_alphas(self):
        """Test name generation with custom regularization strengths."""
        ensemble = LREnsemble(regularization_strengths=[0.01, 0.5, 5.0])
        name = ensemble.get_name()
        assert "LREnsemble" in name
        assert "0.0" in name
        assert "0.5" in name
        assert "5.0" in name

    def test_supports_uncertainty(self, lr_ensemble):
        """Test that LR ensemble supports uncertainty estimation."""
        assert lr_ensemble.supports_uncertainty() is True

    def test_aggregation_methods(self, small_real_compounds, tmp_path):
        """Test different aggregation methods with LR ensemble."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)

        for method in ['mean', 'median']:
            ensemble = LREnsemble(aggregation_method=method)
            ensemble.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))

    def test_uncertainty_methods(self, small_real_compounds, tmp_path):
        """Test different uncertainty estimation methods."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)

        for method in ['std', 'mad', 'quantile']:
            ensemble = LREnsemble(uncertainty_method=method)
            ensemble.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(uncertainty >= 0)

    def test_weighted_ensemble(self, small_real_compounds, tmp_path):
        """Test weighted ensemble aggregation."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        weights = [0.6, 0.3, 0.1]
        ensemble = LREnsemble(weights=weights)

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)

    def test_ensemble_statistics(self, lr_ensemble, small_real_compounds, tmp_path):
        """Test ensemble statistics retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        stats = lr_ensemble.get_ensemble_statistics()
        assert stats['n_learners'] == 3
        assert stats['is_trained'] is False

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        lr_ensemble.train(features, compounds['Activity'].to_numpy())
        stats = lr_ensemble.get_ensemble_statistics()
        assert stats['is_trained'] is True
        assert 'learner_names' in stats
        assert 'learners_with_uncertainty' in stats
        assert len(stats['learner_names']) == 3

    def test_individual_predictions(self, lr_ensemble, small_real_compounds, tmp_path):
        """Test individual learner predictions retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        lr_ensemble.train(features, compounds['Activity'].to_numpy())
        individual_preds = lr_ensemble.get_individual_predictions(features)

        assert len(individual_preds) == 3
        for learner_name, preds in individual_preds.items():
            assert preds is not None
            assert len(preds) == len(compounds)
            assert np.all(np.isfinite(preds))

    def test_edge_case_single_compound(self, lr_ensemble, tmp_path):
        """Test with single compound."""
        single_compound = pl.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        features = extract_features(single_compound['SMILES'].to_list(), 'morgan', tmp_path)
        lr_ensemble.train(features, single_compound['Activity'].to_numpy())
        predictions, uncertainty = lr_ensemble.predict(features)

        assert len(predictions) == 1
        assert len(uncertainty) == 1
        assert np.isfinite(predictions[0])
        assert uncertainty[0] >= 0

    def test_uncertainty_diversity(self, lr_ensemble, small_real_compounds, tmp_path):
        """Test that ensemble uncertainty captures model diversity."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        lr_ensemble.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = lr_ensemble.predict(features)

        assert np.std(uncertainty) > 0
        assert np.all(uncertainty >= 0)

    def test_regularization_effect_on_predictions(self, small_real_compounds, tmp_path):
        """Test that different regularization strengths produce different predictions."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)

        ensemble_weak = LREnsemble(regularization_strengths=[0.01, 0.01, 0.01])
        ensemble_strong = LREnsemble(regularization_strengths=[10.0, 10.0, 10.0])

        ensemble_weak.train(features, compounds['Activity'].to_numpy())
        ensemble_strong.train(features, compounds['Activity'].to_numpy())

        preds_weak, _ = ensemble_weak.predict(features)
        preds_strong, _ = ensemble_strong.predict(features)

        assert not np.allclose(preds_weak, preds_strong, rtol=1e-2)

    def test_mismatched_array_lengths(self, lr_ensemble):
        """Test error handling with mismatched feature and target lengths."""
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)

        with pytest.raises(ValueError):
            lr_ensemble.train(features, targets)

    def test_add_learner_to_ensemble(self, lr_ensemble):
        """Test adding learners to LR ensemble."""
        from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner

        initial_count = len(lr_ensemble.learners)

        new_learner = LinearRegressionLearner(alpha=100.0, random_state=789)
        lr_ensemble.add_learner(new_learner)

        assert len(lr_ensemble.learners) == initial_count + 1
        assert not lr_ensemble.is_trained

    def test_remove_learner_from_ensemble(self, lr_ensemble):
        """Test removing learners from ensemble."""
        initial_count = len(lr_ensemble.learners)

        lr_ensemble.remove_learner(0)

        assert len(lr_ensemble.learners) == initial_count - 1
        assert not lr_ensemble.is_trained

    def test_prediction_consistency(self, lr_ensemble, small_real_compounds, tmp_path):
        """Test that predictions are consistent across multiple calls."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        lr_ensemble.train(features, compounds['Activity'].to_numpy())

        predictions1, uncertainty1 = lr_ensemble.predict(features)
        predictions2, uncertainty2 = lr_ensemble.predict(features)

        assert np.allclose(predictions1, predictions2)
        assert np.allclose(uncertainty1, uncertainty2)

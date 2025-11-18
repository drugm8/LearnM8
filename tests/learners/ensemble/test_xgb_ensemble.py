"""Tests for XGBEnsemble implementation."""

import pytest
import numpy as np
import polars as pl

from learnm8.learners.ensemble.xgb_ensemble import XGBEnsemble
from learnm8.features.extraction import extract_features


class TestXGBEnsemble:
    """Test XGBEnsemble functionality with real molecular data."""

    @pytest.fixture
    def xgb_ensemble(self):
        """Create XGBEnsemble instance for testing."""
        return XGBEnsemble()

    @pytest.fixture
    def custom_xgb_ensemble(self):
        """Create XGBEnsemble with custom hyperparameters."""
        return XGBEnsemble(
            learning_rates=[0.01, 0.05, 0.1],
            random_states=[10, 20, 30]
        )

    def test_initialization(self, xgb_ensemble):
        """Test XGBEnsemble initialization with default parameters."""
        assert len(xgb_ensemble.learners) == 3
        assert xgb_ensemble.learning_rates == [0.05, 0.1, 0.2]
        assert xgb_ensemble.random_states == [42, 123, 456]
        assert xgb_ensemble.aggregation_method == 'mean'
        assert xgb_ensemble.uncertainty_method == 'std'
        assert not xgb_ensemble.is_trained
        assert xgb_ensemble.supports_uncertainty() is True

    def test_initialization_with_custom_parameters(self, custom_xgb_ensemble):
        """Test XGBEnsemble initialization with custom hyperparameters."""
        assert len(custom_xgb_ensemble.learners) == 3
        assert custom_xgb_ensemble.learning_rates == [0.01, 0.05, 0.1]
        assert custom_xgb_ensemble.random_states == [10, 20, 30]
        assert custom_xgb_ensemble.supports_uncertainty() is True

    def test_diverse_hyperparameters(self, xgb_ensemble):
        """Test that ensemble models have different learning rates."""
        learning_rates = []
        random_states = []

        for learner in xgb_ensemble.learners:
            learning_rates.append(learner.learning_rate)
            random_states.append(learner.model.random_state)

        assert len(set(learning_rates)) == 3
        assert learning_rates == [0.05, 0.1, 0.2]
        assert len(set(random_states)) == 3
        assert random_states == [42, 123, 456]

    def test_train_predict_integration(self, xgb_ensemble, small_real_compounds, tmp_path):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        xgb_ensemble.train(features, compounds['Activity'].to_numpy())
        assert xgb_ensemble.is_trained

        predictions, uncertainty = xgb_ensemble.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_uncertainty_estimation(self, xgb_ensemble, small_real_compounds, tmp_path):
        """Test that XGBEnsemble provides uncertainty estimates."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        xgb_ensemble.train(features, compounds['Activity'].to_numpy())

        predictions, uncertainty = xgb_ensemble.predict(features)

        assert uncertainty is not None
        assert len(uncertainty) == len(predictions)
        assert np.all(uncertainty >= 0)
        assert np.std(uncertainty) > 0

    def test_train_with_empty_arrays(self, xgb_ensemble):
        """Test error handling when training with empty arrays."""
        empty_features = np.array([]).reshape(0, 2048)
        empty_targets = np.array([])

        with pytest.raises(ValueError, match="Cannot train on empty dataset"):
            xgb_ensemble.train(empty_features, empty_targets)

    def test_predict_without_training(self, xgb_ensemble, small_real_compounds, tmp_path):
        """Test error when predicting without training."""
        features = extract_features(small_real_compounds['SMILES'].to_list(), 'morgan', tmp_path)
        with pytest.raises(RuntimeError, match="Ensemble must be trained before prediction"):
            xgb_ensemble.predict(features)

    def test_get_name(self, xgb_ensemble):
        """Test name generation for XGBEnsemble."""
        name = xgb_ensemble.get_name()
        assert name == "XGBEnsemble(3xXGB,lr=[0.05,0.10,0.20])"
        assert "XGBEnsemble" in name
        assert "3xXGB" in name
        assert "lr=[" in name

    def test_get_name_custom_params(self, custom_xgb_ensemble):
        """Test name generation with custom parameters."""
        name = custom_xgb_ensemble.get_name()
        assert name == "XGBEnsemble(3xXGB,lr=[0.01,0.05,0.10])"

    def test_supports_uncertainty(self, xgb_ensemble):
        """Test that XGBEnsemble reports uncertainty support."""
        assert xgb_ensemble.supports_uncertainty() is True

    def test_different_aggregation_methods(self, small_real_compounds, tmp_path):
        """Test XGBEnsemble with different aggregation methods."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)

        for method in ['mean', 'median']:
            ensemble = XGBEnsemble(aggregation_method=method)
            ensemble.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))

    def test_different_uncertainty_methods(self, small_real_compounds, tmp_path):
        """Test XGBEnsemble with different uncertainty estimation methods."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)

        for method in ['std', 'mad', 'quantile']:
            ensemble = XGBEnsemble(uncertainty_method=method)
            ensemble.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(uncertainty >= 0)

    def test_weighted_ensemble(self, small_real_compounds, tmp_path):
        """Test weighted XGBEnsemble aggregation."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        weights = [0.5, 0.3, 0.2]
        ensemble = XGBEnsemble(weights=weights)

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)
        assert np.allclose(ensemble.weights, weights)

    def test_individual_predictions(self, xgb_ensemble, small_real_compounds, tmp_path):
        """Test individual learner predictions retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        xgb_ensemble.train(features, compounds['Activity'].to_numpy())
        individual_preds = xgb_ensemble.get_individual_predictions(features)

        assert len(individual_preds) == 3
        for learner_name, preds in individual_preds.items():
            assert preds is not None
            assert len(preds) == len(compounds)
            assert np.all(np.isfinite(preds))

    def test_ensemble_statistics(self, xgb_ensemble, small_real_compounds, tmp_path):
        """Test ensemble statistics retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        stats = xgb_ensemble.get_ensemble_statistics()
        assert stats['n_learners'] == 3
        assert stats['is_trained'] is False

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        xgb_ensemble.train(features, compounds['Activity'].to_numpy())
        stats = xgb_ensemble.get_ensemble_statistics()
        assert stats['is_trained'] is True
        assert 'learner_names' in stats
        assert 'learners_with_uncertainty' in stats
        assert stats['learners_with_uncertainty'] == 0

    def test_edge_case_single_compound(self, xgb_ensemble, tmp_path):
        """Test with single compound."""
        single_compound = pl.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        features = extract_features(single_compound['SMILES'].to_list(), 'morgan', tmp_path)
        xgb_ensemble.train(features, single_compound['Activity'].to_numpy())
        predictions, uncertainty = xgb_ensemble.predict(features)

        assert len(predictions) == 1
        assert len(uncertainty) == 1
        assert np.isfinite(predictions[0])
        assert uncertainty[0] >= 0

    def test_uncertainty_diversity_across_models(self, xgb_ensemble, small_real_compounds, tmp_path):
        """Test that ensemble uncertainty captures model diversity."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        xgb_ensemble.train(features, compounds['Activity'].to_numpy())

        individual_preds = xgb_ensemble.get_individual_predictions(features)
        predictions, uncertainty = xgb_ensemble.predict(features)

        pred_arrays = [preds for preds in individual_preds.values()]
        assert len(pred_arrays) == 3

        manual_std = np.std(pred_arrays, axis=0)
        np.testing.assert_allclose(uncertainty, manual_std, rtol=1e-5)

    def test_consistency_across_predictions(self, xgb_ensemble, small_real_compounds, tmp_path):
        """Test that predictions are consistent across multiple calls."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        xgb_ensemble.train(features, compounds['Activity'].to_numpy())

        predictions1, uncertainty1 = xgb_ensemble.predict(features)
        predictions2, uncertainty2 = xgb_ensemble.predict(features)

        np.testing.assert_array_equal(predictions1, predictions2)
        np.testing.assert_array_equal(uncertainty1, uncertainty2)

    def test_add_learner(self, xgb_ensemble):
        """Test adding learners to ensemble."""
        from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner

        initial_count = len(xgb_ensemble.learners)

        new_learner = XGBoostLearner(learning_rate=0.15, random_state=789)
        xgb_ensemble.add_learner(new_learner)

        assert len(xgb_ensemble.learners) == initial_count + 1
        assert not xgb_ensemble.is_trained

    def test_remove_learner(self, xgb_ensemble):
        """Test removing learners from ensemble."""
        initial_count = len(xgb_ensemble.learners)

        xgb_ensemble.remove_learner(0)

        assert len(xgb_ensemble.learners) == initial_count - 1
        assert not xgb_ensemble.is_trained

    def test_mismatched_learning_rates_and_random_states(self):
        """Test behavior when learning rates and random states don't match."""
        ensemble = XGBEnsemble(learning_rates=[0.1, 0.2], random_states=[42])
        assert len(ensemble.learners) == 1

    def test_prediction_quality_improves_with_training_data(self, xgb_ensemble, small_real_compounds, tmp_path):
        """Test that predictions are reasonable with sufficient training data."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        xgb_ensemble.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = xgb_ensemble.predict(features)

        assert predictions.min() >= compounds['Activity'].min() - 0.5
        assert predictions.max() <= compounds['Activity'].max() + 0.5

        activity_std = compounds['Activity'].std()
        assert np.mean(uncertainty) < activity_std
        assert np.max(uncertainty) < 5 * activity_std

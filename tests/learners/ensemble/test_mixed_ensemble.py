"""Tests for MixedEnsemble implementation."""

import pytest
import numpy as np
import pandas as pd

from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble
from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner
from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner
from learnm8.learners.sklearn.decision_tree import DecisionTreeLearner
from learnm8.features.extraction import extract_features


class TestMixedEnsemble:
    """Test MixedEnsemble functionality with real molecular data."""

    @pytest.fixture
    def mixed_ensemble(self):
        """Create MixedEnsemble instance for testing."""
        return MixedEnsemble(random_state=42)

    def test_initialization(self, mixed_ensemble):
        """Test mixed ensemble initialization."""
        assert len(mixed_ensemble.learners) == 3
        assert mixed_ensemble.aggregation_method == 'mean'
        assert mixed_ensemble.uncertainty_method == 'std'
        assert mixed_ensemble.weights is None
        assert not mixed_ensemble.is_trained
        assert mixed_ensemble.supports_uncertainty() is True
        assert mixed_ensemble.random_state == 42

    def test_mixed_model_types(self, mixed_ensemble):
        """Test that mixed ensemble contains RF, LR, and XGB."""
        learner_types = [type(learner).__name__ for learner in mixed_ensemble.learners]

        assert 'RandomForestLearner' in learner_types
        assert 'LinearRegressionLearner' in learner_types
        assert 'XGBoostLearner' in learner_types

    def test_model_diversity(self, mixed_ensemble):
        """Test that models are different types."""
        learner_types = [type(learner).__name__ for learner in mixed_ensemble.learners]
        unique_types = set(learner_types)

        assert len(unique_types) == 3
        assert len(learner_types) == 3

    def test_train_predict_integration(self, mixed_ensemble, small_real_compounds, tmp_path):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        mixed_ensemble.train(features, compounds['Activity'].values)
        assert mixed_ensemble.is_trained

        predictions, uncertainty = mixed_ensemble.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_uncertainty_estimation(self, mixed_ensemble, small_real_compounds, tmp_path):
        """Test that ensemble provides meaningful uncertainty estimates."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        mixed_ensemble.train(features, compounds['Activity'].values)
        predictions, uncertainty = mixed_ensemble.predict(features)

        assert uncertainty is not None
        assert np.all(uncertainty >= 0)
        assert np.std(uncertainty) > 0
        assert np.mean(uncertainty) > 0

    def test_train_with_empty_arrays(self, mixed_ensemble):
        """Test error handling with empty training data."""
        with pytest.raises(Exception):
            mixed_ensemble.train(np.array([]), np.array([]))

    def test_predict_without_training(self, mixed_ensemble, small_real_compounds, tmp_path):
        """Test error when predicting without training."""
        features = extract_features(small_real_compounds['SMILES'].tolist(), 'morgan', tmp_path)
        with pytest.raises(RuntimeError, match="Ensemble must be trained before prediction"):
            mixed_ensemble.predict(features)

    def test_get_name(self, mixed_ensemble):
        """Test name generation shows Mixed ensemble."""
        name = mixed_ensemble.get_name()
        assert name == "MixedEnsemble(RF+LR+XGB)"
        assert "Mixed" in name

    def test_supports_uncertainty(self, mixed_ensemble):
        """Test that MixedEnsemble supports uncertainty estimation."""
        assert mixed_ensemble.supports_uncertainty() is True

    def test_initialization_with_custom_random_state(self):
        """Test initialization with custom random state."""
        ensemble = MixedEnsemble(random_state=123)
        assert ensemble.random_state == 123
        assert len(ensemble.learners) == 3

    def test_initialization_with_custom_aggregation(self):
        """Test initialization with custom aggregation method."""
        ensemble = MixedEnsemble(aggregation_method='median', random_state=42)
        assert ensemble.aggregation_method == 'median'
        assert ensemble.uncertainty_method == 'std'

    def test_initialization_with_custom_uncertainty_method(self):
        """Test initialization with custom uncertainty estimation method."""
        ensemble = MixedEnsemble(uncertainty_method='mad', random_state=42)
        assert ensemble.uncertainty_method == 'mad'

    def test_ensemble_statistics(self, mixed_ensemble, small_real_compounds, tmp_path):
        """Test ensemble statistics retrieval."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        stats = mixed_ensemble.get_ensemble_statistics()
        assert stats['n_learners'] == 3
        assert stats['is_trained'] is False

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        mixed_ensemble.train(features, compounds['Activity'].values)
        stats = mixed_ensemble.get_ensemble_statistics()
        assert stats['is_trained'] is True
        assert 'learner_names' in stats
        assert 'learners_with_uncertainty' in stats

    def test_individual_predictions(self, mixed_ensemble, small_real_compounds, tmp_path):
        """Test individual learner predictions retrieval."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        mixed_ensemble.train(features, compounds['Activity'].values)
        individual_preds = mixed_ensemble.get_individual_predictions(features)

        assert len(individual_preds) == 3
        for learner_name, preds in individual_preds.items():
            if preds is not None:
                assert len(preds) == len(compounds)

    def test_edge_case_single_compound(self, mixed_ensemble, tmp_path):
        """Test with single compound."""
        single_compound = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        features = extract_features(single_compound['SMILES'].tolist(), 'morgan', tmp_path)
        mixed_ensemble.train(features, single_compound['Activity'].values)
        predictions, uncertainty = mixed_ensemble.predict(features)

        assert len(predictions) == 1
        assert len(uncertainty) == 1
        assert np.isfinite(predictions[0])
        assert uncertainty[0] >= 0

    def test_model_diversity_in_predictions(self, mixed_ensemble, small_real_compounds, tmp_path):
        """Test that different models produce diverse predictions."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        mixed_ensemble.train(features, compounds['Activity'].values)
        individual_preds = mixed_ensemble.get_individual_predictions(features)

        pred_arrays = [preds for preds in individual_preds.values() if preds is not None]
        assert len(pred_arrays) >= 2

        for i in range(len(pred_arrays) - 1):
            for j in range(i + 1, len(pred_arrays)):
                correlation = np.corrcoef(pred_arrays[i], pred_arrays[j])[0, 1]
                assert correlation < 1.0

    def test_uncertainty_captures_model_disagreement(self, mixed_ensemble, small_real_compounds, tmp_path):
        """Test that uncertainty reflects disagreement between models."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        mixed_ensemble.train(features, compounds['Activity'].values)

        predictions, uncertainty = mixed_ensemble.predict(features)
        individual_preds = mixed_ensemble.get_individual_predictions(features)

        pred_arrays = np.array([preds for preds in individual_preds.values() if preds is not None])
        expected_std = np.std(pred_arrays, axis=0)

        assert uncertainty.shape == expected_std.shape
        np.testing.assert_allclose(uncertainty, expected_std, rtol=1e-5)

    def test_all_learners_train_successfully(self, mixed_ensemble, small_real_compounds, tmp_path):
        """Test that all learners train without errors."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)

        initial_learner_count = len(mixed_ensemble.learners)
        mixed_ensemble.train(features, compounds['Activity'].values)

        assert len(mixed_ensemble.learners) == initial_learner_count
        assert mixed_ensemble.is_trained

    def test_prediction_range_reasonable(self, mixed_ensemble, small_real_compounds, tmp_path):
        """Test that predictions are in reasonable range given training data."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        mixed_ensemble.train(features, compounds['Activity'].values)
        predictions, _ = mixed_ensemble.predict(features)

        train_min = compounds['Activity'].min()
        train_max = compounds['Activity'].max()
        train_range = train_max - train_min

        pred_min = predictions.min()
        pred_max = predictions.max()

        assert pred_min < train_max + 2 * train_range
        assert pred_max > train_min - 2 * train_range

    def test_reproducibility_with_fixed_random_state(self, small_real_compounds, tmp_path):
        """Test that fixed random state produces reproducible results."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)

        ensemble1 = MixedEnsemble(random_state=42)
        ensemble1.train(features, compounds['Activity'].values)
        pred1, unc1 = ensemble1.predict(features)

        ensemble2 = MixedEnsemble(random_state=42)
        ensemble2.train(features, compounds['Activity'].values)
        pred2, unc2 = ensemble2.predict(features)

        np.testing.assert_array_almost_equal(pred1, pred2)
        np.testing.assert_array_almost_equal(unc1, unc2)

    def test_different_random_states_produce_different_results(self, small_real_compounds, tmp_path):
        """Test that different random states produce different results."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)

        ensemble1 = MixedEnsemble(random_state=42)
        ensemble1.train(features, compounds['Activity'].values)
        pred1, _ = ensemble1.predict(features)

        ensemble2 = MixedEnsemble(random_state=123)
        ensemble2.train(features, compounds['Activity'].values)
        pred2, _ = ensemble2.predict(features)

        assert not np.allclose(pred1, pred2)

    def test_empty_features_handling(self, mixed_ensemble):
        """Test handling of empty feature arrays after training."""
        dummy_compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002'],
            'SMILES': ['CCO', 'CCC'],
            'Activity': [0.5, 0.7]
        })

        dummy_features = np.random.randn(2, 2048)
        mixed_ensemble.train(dummy_features, dummy_compounds['Activity'].values)

        empty_features = np.array([]).reshape(0, 2048)
        predictions, uncertainty = mixed_ensemble.predict(empty_features)

        assert len(predictions) == 0
        assert len(uncertainty) == 0

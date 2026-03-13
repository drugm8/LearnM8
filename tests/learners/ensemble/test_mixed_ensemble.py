"""Tests for MixedEnsemble implementation."""

import pytest
import numpy as np
import polars as pl

from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble


@pytest.mark.slow
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

    def test_get_name(self, mixed_ensemble):
        """Test name generation shows Mixed ensemble."""
        name = mixed_ensemble.get_name()
        assert name == "MixedEnsemble(RF+LR+XGB)"
        assert "Mixed" in name

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

    def test_model_diversity_in_predictions(self, mixed_ensemble, small_real_compounds, small_real_morgan_features):
        """Test that different models produce diverse predictions."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features
        mixed_ensemble.train(features, compounds['Activity'].to_numpy())
        individual_preds = mixed_ensemble.get_individual_predictions(features)

        pred_arrays = [preds for preds in individual_preds.values() if preds is not None]
        assert len(pred_arrays) >= 2

        for i in range(len(pred_arrays) - 1):
            for j in range(i + 1, len(pred_arrays)):
                correlation = np.corrcoef(pred_arrays[i], pred_arrays[j])[0, 1]
                assert correlation < 1.0

    def test_uncertainty_captures_model_disagreement(self, mixed_ensemble, small_real_compounds, small_real_morgan_features):
        """Test that uncertainty reflects disagreement between models."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features
        mixed_ensemble.train(features, compounds['Activity'].to_numpy())

        predictions, uncertainty = mixed_ensemble.predict(features)
        individual_preds = mixed_ensemble.get_individual_predictions(features)

        pred_arrays = np.array([preds for preds in individual_preds.values() if preds is not None])
        expected_std = np.std(pred_arrays, axis=0)

        assert uncertainty.shape == expected_std.shape
        np.testing.assert_allclose(uncertainty, expected_std, rtol=1e-5)

    def test_all_learners_train_successfully(self, mixed_ensemble, small_real_compounds, small_real_morgan_features):
        """Test that all learners train without errors."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features

        initial_learner_count = len(mixed_ensemble.learners)
        mixed_ensemble.train(features, compounds['Activity'].to_numpy())

        assert len(mixed_ensemble.learners) == initial_learner_count
        assert mixed_ensemble.is_trained

    def test_prediction_range_reasonable(self, mixed_ensemble, small_real_compounds, small_real_morgan_features):
        """Test that predictions are in reasonable range given training data."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features
        mixed_ensemble.train(features, compounds['Activity'].to_numpy())
        predictions, _ = mixed_ensemble.predict(features)

        train_min = compounds['Activity'].min()
        train_max = compounds['Activity'].max()
        train_range = train_max - train_min

        pred_min = predictions.min()
        pred_max = predictions.max()

        assert pred_min < train_max + 2 * train_range
        assert pred_max > train_min - 2 * train_range

    def test_reproducibility_with_fixed_random_state(self, small_real_compounds, small_real_morgan_features):
        """Test that fixed random state produces reproducible results."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features

        ensemble1 = MixedEnsemble(random_state=42)
        ensemble1.train(features, compounds['Activity'].to_numpy())
        pred1, unc1 = ensemble1.predict(features)

        ensemble2 = MixedEnsemble(random_state=42)
        ensemble2.train(features, compounds['Activity'].to_numpy())
        pred2, unc2 = ensemble2.predict(features)

        np.testing.assert_array_almost_equal(pred1, pred2)
        np.testing.assert_array_almost_equal(unc1, unc2)

    def test_different_random_states_produce_different_results(self, small_real_compounds, small_real_morgan_features):
        """Test that different random states produce different results."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features

        ensemble1 = MixedEnsemble(random_state=42)
        ensemble1.train(features, compounds['Activity'].to_numpy())
        pred1, _ = ensemble1.predict(features)

        ensemble2 = MixedEnsemble(random_state=123)
        ensemble2.train(features, compounds['Activity'].to_numpy())
        pred2, _ = ensemble2.predict(features)

        assert not np.allclose(pred1, pred2)

    def test_empty_features_handling(self, mixed_ensemble):
        """Test handling of empty feature arrays after training."""
        dummy_compounds = pl.DataFrame({
            'ID': ['COMP_001', 'COMP_002'],
            'SMILES': ['CCO', 'CCC'],
            'Activity': [0.5, 0.7]
        })

        dummy_features = np.random.randn(2, 2048)
        mixed_ensemble.train(dummy_features, dummy_compounds['Activity'].to_numpy())

        empty_features = np.array([]).reshape(0, 2048)
        predictions, uncertainty = mixed_ensemble.predict(empty_features)

        assert len(predictions) == 0
        assert len(uncertainty) == 0

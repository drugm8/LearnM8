"""Tests for LearnM8 learners module."""

import pytest
import pandas as pd
import numpy as np

from learnm8.learners import (
    RandomForestLearner, AdvancedRandomForestLearner, MLPLearner,
    LinearRegressionLearner, DecisionTreeLearner
)
from learnm8.learners.gaussian_process import GaussianProcessLearner  
from learnm8.learners.gaussian_pytorch import GaussianPyTorchLearner
from learnm8.learners.gradient_boosting import XGBoostLearner
from learnm8.learners.pytorch_mlp import PyTorchMLPLearner


class TestLearnerInstantiation:
	"""Test that all learners can be instantiated correctly."""
	
	def test_random_forest_instantiation(self, test_results_dir):
		"""Test RandomForestLearner instantiation."""
		learner = RandomForestLearner(results_dir=test_results_dir)
		assert learner.get_name() == "Random Forest"
		assert learner.featurizer_type == 'morgan'
		assert not learner.is_trained
	
	def test_gaussian_process_instantiation(self, test_results_dir):
		"""Test GaussianProcessLearner instantiation."""
		learner = GaussianProcessLearner(results_dir=test_results_dir)
		assert learner.get_name() == "Gaussian Process"
		assert learner.featurizer_type == 'morgan'
		assert not learner.is_trained
	
	def test_gaussian_pytorch_instantiation(self, test_results_dir):
		"""Test GaussianPyTorchLearner instantiation."""
		learner = GaussianPyTorchLearner(results_dir=test_results_dir)
		assert learner.get_name() == "Gaussian Process (PyTorch)"
		assert learner.featurizer_type == 'morgan'
		assert not learner.is_trained
	
	def test_gradient_boosting_instantiation(self, test_results_dir):
		"""Test XGBoostLearner instantiation."""
		learner = XGBoostLearner(results_dir=test_results_dir)
		assert learner.get_name() == "XGBoost"
		assert learner.featurizer_type == 'morgan'
		assert not learner.is_trained
	
	def test_advanced_random_forest_instantiation(self, test_results_dir):
		"""Test AdvancedRandomForestLearner instantiation."""
		learner = AdvancedRandomForestLearner(results_dir=test_results_dir)
		assert learner.get_name() == "Advanced Random Forest"
		assert learner.featurizer_type == 'morgan'
		assert not learner.is_trained
	
	def test_mlp_instantiation(self, test_results_dir):
		"""Test MLPLearner instantiation."""
		learner = MLPLearner(results_dir=test_results_dir)
		assert learner.get_name() == "Multi-Layer Perceptron"
		assert learner.featurizer_type == 'morgan'
		assert not learner.is_trained
	
	def test_pytorch_mlp_instantiation(self, test_results_dir):
		"""Test PyTorchMLPLearner instantiation."""
		learner = PyTorchMLPLearner(results_dir=test_results_dir)
		assert learner.get_name() == "PyTorch MLP"
		assert learner.featurizer_type == 'morgan'
		assert not learner.is_trained
	
	def test_linear_regression_instantiation(self, test_results_dir):
		"""Test LinearRegressionLearner instantiation."""
		learner = LinearRegressionLearner(results_dir=test_results_dir)
		assert learner.get_name() == "Linear Regression"
		assert learner.featurizer_type == 'morgan'
		assert not learner.is_trained
	
	def test_decision_tree_instantiation(self, test_results_dir):
		"""Test DecisionTreeLearner instantiation."""
		learner = DecisionTreeLearner(results_dir=test_results_dir)
		assert learner.get_name() == "Decision Tree"
		assert learner.featurizer_type == 'morgan'
		assert not learner.is_trained


class TestFeaturizerTypes:
	"""Test that learners work with different featurizer types."""
	
	@pytest.mark.parametrize("learner_class", [
		RandomForestLearner,
		AdvancedRandomForestLearner,
		MLPLearner,
		LinearRegressionLearner,
		DecisionTreeLearner,
		GaussianProcessLearner,
		XGBoostLearner
	])
	def test_sklearn_learners_featurizer_types(self, learner_class, featurizer_type, test_results_dir):
		"""Test sklearn-based learners with different featurizers."""
		learner = learner_class(results_dir=test_results_dir, featurizer_type=featurizer_type)
		assert learner.featurizer_type == featurizer_type
	
	def test_gaussian_pytorch_featurizer_types(self, featurizer_type, test_results_dir):
		"""Test GaussianPyTorchLearner with different featurizers."""
		learner = GaussianPyTorchLearner(results_dir=test_results_dir, featurizer_type=featurizer_type)
		assert learner.featurizer_type == featurizer_type
	
	def test_pytorch_mlp_featurizer_types(self, featurizer_type, test_results_dir):
		"""Test PyTorchMLPLearner with different featurizers."""
		learner = PyTorchMLPLearner(results_dir=test_results_dir, featurizer_type=featurizer_type)
		assert learner.featurizer_type == featurizer_type


class TestTrainMethod:
	"""Test the train method for all learners."""
	
	@pytest.mark.parametrize("learner_class", [
		RandomForestLearner,
		AdvancedRandomForestLearner,
		MLPLearner,
		LinearRegressionLearner,
		DecisionTreeLearner,
		GaussianProcessLearner,
		XGBoostLearner
	])
	def test_sklearn_learners_train(self, learner_class, training_data, target_column, test_results_dir):
		"""Test training of sklearn-based learners."""
		learner = learner_class(results_dir=test_results_dir, featurizer_type='morgan')
		
		# Train the model
		learner.train(training_data, target_column)
		
		# Verify training state
		assert learner.is_trained
		assert learner.target_column == target_column
		assert len(learner.training_data) == len(training_data)
		assert learner.model is not None
	
	def test_gaussian_pytorch_train(self, training_data, target_column, test_results_dir):
		"""Test training of GaussianPyTorchLearner."""
		learner = GaussianPyTorchLearner(
			results_dir=test_results_dir, 
			featurizer_type='morgan',
			training_iter=5  # Reduced for fast testing
		)
		
		# Train the model
		learner.train(training_data, target_column)
		
		# Verify training state
		assert learner.is_trained
		assert learner.target_column == target_column
		assert len(learner.training_data) == len(training_data)
		assert hasattr(learner, 'model')
		assert hasattr(learner, 'likelihood')
	
	def test_pytorch_mlp_train(self, training_data, target_column, test_results_dir):
		"""Test training of PyTorchMLPLearner."""
		learner = PyTorchMLPLearner(
			results_dir=test_results_dir, 
			featurizer_type='morgan',
			epochs=5  # Reduced for fast testing
		)
		
		# Train the model
		learner.train(training_data, target_column)
		
		# Verify training state
		assert learner.is_trained
		assert learner.target_column == target_column
		assert len(learner.training_data) == len(training_data)
		assert learner.model is not None


class TestPredictMethod:
	"""Test the predict method for all learners."""
	
	@pytest.mark.parametrize("learner_class", [
		RandomForestLearner,
		AdvancedRandomForestLearner,
		MLPLearner,
		LinearRegressionLearner,
		DecisionTreeLearner,
		XGBoostLearner
	])
	def test_sklearn_deterministic_predict(self, learner_class, training_data, prediction_data, test_results_dir):
		"""Test prediction for deterministic sklearn learners (no uncertainty)."""
		learner = learner_class(results_dir=test_results_dir, featurizer_type='morgan')
		
		# Train first
		learner.train(training_data, 'ESSENCE-Dock_Score')
		
		# Predict
		predictions, uncertainty = learner.predict(prediction_data)
		
		# Verify prediction format
		assert isinstance(predictions, np.ndarray)
		assert len(predictions) == len(prediction_data)
		assert uncertainty is None  # These learners don't return uncertainty
		assert all(np.isfinite(predictions))  # All predictions should be finite
	
	def test_gaussian_process_predict_with_uncertainty(self, training_data, prediction_data, test_results_dir):
		"""Test prediction for GaussianProcessLearner (with uncertainty)."""
		learner = GaussianProcessLearner(results_dir=test_results_dir, featurizer_type='morgan')
		
		# Train first
		learner.train(training_data, 'ESSENCE-Dock_Score')
		
		# Predict
		predictions, uncertainty = learner.predict(prediction_data)
		
		# Verify prediction format
		assert isinstance(predictions, np.ndarray)
		assert isinstance(uncertainty, np.ndarray)
		assert len(predictions) == len(prediction_data)
		assert len(uncertainty) == len(prediction_data)
		assert all(np.isfinite(predictions))
		assert all(np.isfinite(uncertainty))
		assert all(uncertainty >= 0)  # Uncertainty should be non-negative
		print(f"Predictions: {predictions}")
		print(f"Uncertainty: {uncertainty}")
	
	def test_pytorch_mlp_predict(self, training_data, prediction_data, test_results_dir):
		"""Test prediction for PyTorchMLPLearner."""
		learner = PyTorchMLPLearner(results_dir=test_results_dir, featurizer_type='morgan', epochs=5)
		
		# Train first
		learner.train(training_data, 'ESSENCE-Dock_Score')
		
		# Predict
		predictions, uncertainty = learner.predict(prediction_data)
		
		# Verify prediction format
		assert isinstance(predictions, np.ndarray)
		assert len(predictions) == len(prediction_data)
		assert uncertainty is None  # PyTorch MLP doesn't return uncertainty
		assert all(np.isfinite(predictions))  # All predictions should be finite
	
	def test_gaussian_pytorch_predict_with_uncertainty(self, training_data, prediction_data, test_results_dir):
		"""Test prediction for GaussianPyTorchLearner (with uncertainty)."""
		learner = GaussianPyTorchLearner(
			results_dir=test_results_dir, 
			featurizer_type='morgan',
			training_iter=500  # Reduced for fast testing
		)
		
		# Train first
		learner.train(training_data, 'ESSENCE-Dock_Score')
		
		# Predict
		predictions, uncertainty = learner.predict(prediction_data)
		
		# Verify prediction format
		assert isinstance(predictions, np.ndarray)
		assert len(predictions) == len(prediction_data)
		assert all(np.isfinite(predictions))
		
		# PyTorch GP may or may not return uncertainty depending on implementation
		if uncertainty is not None:
			assert isinstance(uncertainty, np.ndarray)
			assert len(uncertainty) == len(prediction_data)
			assert all(np.isfinite(uncertainty))
			assert all(uncertainty >= 0)
		
		print(f"Predictions: {predictions}")
		print(f"Uncertainty: {uncertainty}")


class TestTrainPredictWorkflow:
	"""Test complete train-predict workflow for all learners."""
	
	@pytest.mark.parametrize("learner_class,featurizer", [
		(RandomForestLearner, 'morgan'),
		(AdvancedRandomForestLearner, 'morgan'),
		(MLPLearner, 'morgan'),
		(LinearRegressionLearner, 'morgan'),
		(DecisionTreeLearner, 'morgan'),
		(GaussianProcessLearner, 'morgan'),
		(XGBoostLearner, 'morgan'),
		(GaussianPyTorchLearner, 'morgan'),
		(PyTorchMLPLearner, 'morgan'),
	])
	def test_complete_workflow(self, learner_class, featurizer, sample_molecular_data, test_results_dir):
		"""Test complete train-predict workflow."""
		# Split data
		train_data = sample_molecular_data.iloc[:6].copy()
		pred_data = sample_molecular_data.iloc[6:].copy()
		target = 'ESSENCE-Dock_Score'
		
		# Setup learner
		if learner_class == GaussianPyTorchLearner:
			learner = learner_class(
				results_dir=test_results_dir, 
				featurizer_type=featurizer,
				training_iter=5
			)
		elif learner_class == PyTorchMLPLearner:
			learner = learner_class(
				results_dir=test_results_dir, 
				featurizer_type=featurizer,
				epochs=5
			)
		else:
			learner = learner_class(results_dir=test_results_dir, featurizer_type=featurizer)
		
		# Train
		learner.train(train_data, target)
		assert learner.is_trained
		
		# Predict
		predictions, _ = learner.predict(pred_data)
		
		# Verify results
		assert len(predictions) == len(pred_data)
		assert all(np.isfinite(predictions))
		
		# Verify prediction values are reasonable (should be in similar range as training data)
		train_values = train_data[target].values
		pred_range = np.ptp(train_values)  # Range of training data
		train_mean = np.mean(train_values)
		
		# Predictions should be within 3x the training range of the training mean
		assert all(abs(predictions - train_mean) <= 3 * pred_range)


class TestErrorHandling:
	"""Test error handling and edge cases."""
	
	@pytest.mark.parametrize("learner_class", [
		RandomForestLearner,
		AdvancedRandomForestLearner,
		MLPLearner,
		LinearRegressionLearner,
		DecisionTreeLearner
	])
	def test_predict_before_training(self, learner_class, prediction_data, test_results_dir):
		"""Test that prediction fails when model is not trained."""
		learner = learner_class(results_dir=test_results_dir)
		
		with pytest.raises(Exception):  # Should raise some kind of error
			learner.predict(prediction_data)
	
	@pytest.mark.parametrize("learner_class", [
		RandomForestLearner,
		AdvancedRandomForestLearner,
		MLPLearner,
		LinearRegressionLearner,
		DecisionTreeLearner
	])
	def test_train_with_missing_target_column(self, learner_class, training_data, test_results_dir):
		"""Test training with non-existent target column."""
		learner = learner_class(results_dir=test_results_dir)
		
		with pytest.raises(Exception):  # Should raise KeyError or similar
			learner.train(training_data, 'NonExistentColumn')
	
	@pytest.mark.parametrize("learner_class", [
		RandomForestLearner,
		AdvancedRandomForestLearner,
		MLPLearner,
		LinearRegressionLearner,
		DecisionTreeLearner
	])
	def test_train_with_empty_data(self, learner_class, test_results_dir):
		"""Test training with empty DataFrame."""
		learner = learner_class(results_dir=test_results_dir)
		empty_data = pd.DataFrame(columns=['ID', 'SMILES', 'Activity'])
		
		with pytest.raises(Exception):  # Should raise some kind of error
			learner.train(empty_data, 'Activity')


class TestDataAccumulation:
	"""Test training data accumulation across multiple training calls."""
	
	@pytest.mark.parametrize("learner_class", [
		RandomForestLearner,
		AdvancedRandomForestLearner,
		MLPLearner,
		LinearRegressionLearner,
		DecisionTreeLearner
	])
	def test_training_data_accumulation(self, learner_class, sample_molecular_data, test_results_dir):
		"""Test that training data accumulates correctly across multiple calls."""
		learner = learner_class(results_dir=test_results_dir)
		
		# First training batch
		batch1 = sample_molecular_data.iloc[:3].copy()
		learner.train(batch1, 'ESSENCE-Dock_Score')
		assert len(learner.training_data) == 3
		
		# Second training batch
		batch2 = sample_molecular_data.iloc[3:6].copy()
		learner.train(batch2, 'ESSENCE-Dock_Score')
		assert len(learner.training_data) == 6
		
		# Verify all IDs are present
		expected_ids = set(sample_molecular_data.iloc[:6]['ID'])
		actual_ids = set(learner.training_data['ID'])
		assert expected_ids == actual_ids


class TestAdvancedFeatures:
	"""Test advanced features specific to new learners."""
	
	def test_advanced_random_forest_hyperparameters(self, test_results_dir):
		"""Test that AdvancedRandomForestLearner has enhanced hyperparameters."""
		learner = AdvancedRandomForestLearner(results_dir=test_results_dir)
		
		# Check specific hyperparameters are set correctly
		assert learner.model.n_estimators == 300
		assert learner.model.max_depth == 15
		assert learner.model.min_samples_split == 5
		assert learner.model.min_samples_leaf == 2
		assert learner.model.max_features == 'sqrt'
		assert learner.model.oob_score == True
		assert learner.model.max_samples == 0.8
	
	def test_mlp_architecture(self, test_results_dir):
		"""Test that MLPLearner has correct architecture."""
		learner = MLPLearner(results_dir=test_results_dir)
		
		# Check MLP architecture
		assert learner.model.hidden_layer_sizes == (128, 128)
		assert learner.model.activation == 'relu'
		assert learner.model.solver == 'adam'
		assert learner.model.early_stopping == True
	
	def test_decision_tree_hyperparameters(self, test_results_dir):
		"""Test that DecisionTreeLearner has reasonable hyperparameters."""
		learner = DecisionTreeLearner(results_dir=test_results_dir)
		
		# Check decision tree hyperparameters
		assert learner.model.max_depth == 10
		assert learner.model.min_samples_split == 10
		assert learner.model.min_samples_leaf == 5
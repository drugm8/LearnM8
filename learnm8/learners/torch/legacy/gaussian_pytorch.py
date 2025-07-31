
"""Gaussian Process learner implementation using PyTorch and GPyTorch."""

import torch
import gpytorch
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path
from learnm8.core.interfaces import Learner
from learnm8.utils.featurizers import get_fingerprints, get_descriptors


class SimpleGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(SimpleGPModel, self).__init__(train_x, train_y, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianPyTorchLearner(Learner):
	"""Gaussian Process learner using PyTorch and GPyTorch."""
	
	def __init__(self, random_state: int = 42, results_dir: Path = None, featurizer_type: str = 'morgan', training_iter: int = 50):
		"""
		Initialize Gaussian Process learner.
		
		Args:
			random_state: Random seed for reproducibility
			results_dir: Directory for storing representations
			featurizer_type: Type of molecular featurizer ('morgan', 'maccs', 'ecfp6', or 'descriptors')
			training_iter: Number of training iterations for GP optimization
		"""
		self.random_state = random_state
		self.results_dir = results_dir
		self.featurizer_type = featurizer_type
		self.training_iter = training_iter
		self.is_trained = False
		self.training_data = pd.DataFrame()
		self.target_column = None
		self.model = None
		self.likelihood = None
		
		# Set random seeds
		torch.manual_seed(random_state)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(random_state)
	
	def train(self, compounds: pd.DataFrame, target_column: str) -> None:
		"""
		Train the Gaussian Process model on labeled compounds.
		
		Args:
			compounds: DataFrame with 'ID', 'SMILES', and target columns
			target_column: Name of column containing target values
		"""
		if target_column not in compounds.columns:
			raise ValueError(f"Target column '{target_column}' not found")
		
		# Accumulate training data
		if self.training_data.empty:
			self.training_data = compounds.copy()
		else:
			self.training_data = pd.concat([self.training_data, compounds], ignore_index=True)
		
		self.target_column = target_column
		
		# Get molecular representations based on featurizer type
		if self.results_dir is None:
			raise RuntimeError("results_dir must be set to use molecular representations")
		
		if self.featurizer_type in ['morgan', 'maccs', 'ecfp6']:
			X = get_fingerprints(self.training_data['ID'].tolist(), self.results_dir, self.featurizer_type)
		elif self.featurizer_type == 'descriptors':
			X = get_descriptors(self.training_data['ID'].tolist(), self.results_dir)
		else:
			raise ValueError(f"Unknown featurizer type: {self.featurizer_type}")
		
		y = self.training_data[target_column].values
		
		# Convert to torch tensors
		train_x = torch.tensor(X, dtype=torch.float32)
		train_y = torch.tensor(y, dtype=torch.float32)
		
		# Move to CUDA if available
		if torch.cuda.is_available():
			train_x = train_x.cuda()
			train_y = train_y.cuda()
		
		# Initialize likelihood and model
		self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
		self.model = SimpleGPModel(train_x, train_y, self.likelihood)
		
		if torch.cuda.is_available():
			self.model = self.model.cuda()
			self.likelihood = self.likelihood.cuda()
		
		# Train the model
		self.model.train()
		self.likelihood.train()
		
		optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
		mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
		
		for _ in range(self.training_iter):
			optimizer.zero_grad()
			output = self.model(train_x)
			loss = -mll(output, train_y)
			loss.backward()
			optimizer.step()
		
		self.is_trained = True
		
		from learnm8.utils.logging import get_logger
		logger = get_logger()
		logger.debug(f"Trained [cyan]{self.get_name()}[/cyan] on [bold]{len(self.training_data)}[/bold] compounds")
	
	def predict(self, compounds: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		"""
		Predict scores for compounds with uncertainty estimates.
		
		Args:
			compounds: DataFrame with 'SMILES' column
			
		Returns:
			Tuple of (predictions, uncertainty) where uncertainty contains GP standard deviations
		"""
		if not self.is_trained:
			raise RuntimeError("Model must be trained before prediction")
		
		# Get molecular representations based on featurizer type
		if self.results_dir is None:
			raise RuntimeError("results_dir must be set to use molecular representations")
		
		if self.featurizer_type in ['morgan', 'maccs', 'ecfp6']:
			X = get_fingerprints(compounds['ID'].tolist(), self.results_dir, self.featurizer_type)
		elif self.featurizer_type == 'descriptors':
			X = get_descriptors(compounds['ID'].tolist(), self.results_dir)
		else:
			raise ValueError(f"Unknown featurizer type: {self.featurizer_type}")

		# Convert to torch tensor
		test_x = torch.tensor(X, dtype=torch.float32)
		
		if torch.cuda.is_available():
			test_x = test_x.cuda()
		
		# Make predictions
		self.model.eval()
		self.likelihood.eval()
		
		with torch.no_grad(), gpytorch.settings.fast_pred_var():
			output = self.likelihood(self.model(test_x))
			
		# Extract mean and standard deviation
		predictions = output.mean.cpu().numpy()
		uncertainties = output.stddev.cpu().numpy()
		
		from learnm8.utils.logging import get_logger
		logger = get_logger()
		logger.debug(f"Predicted [bold]{len(predictions)}[/bold] compounds with [cyan]{self.get_name()}[/cyan]")
		
		return predictions, uncertainties
	
	def get_name(self) -> str:
		"""Return the model name."""
		return "Gaussian Process (PyTorch)"
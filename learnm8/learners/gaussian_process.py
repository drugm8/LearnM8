"""Gaussian Process learner implementation with uncertainty quantification."""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from learnm8.learners.base import SklearnLearner


class GaussianProcessLearner(SklearnLearner):
	"""Gaussian Process based active learner with uncertainty quantification."""
	
	def __init__(self, kernel=None, alpha: float = 1e-10, n_restarts_optimizer: int = 0,
				 random_state: int = 42, results_dir: Path = None, featurizer_type: str = 'morgan'):
		"""
		Initialize Gaussian Process learner.
		
		Args:
			kernel: Gaussian Process kernel. If None, uses RBF kernel with automatic length scale
			alpha: Noise level in the targets (Tikhonov regularization)
			n_restarts_optimizer: Number of restarts of the optimizer for finding hyperparameters
			random_state: Random seed for reproducibility
			results_dir: Directory for storing representations
			featurizer_type: Type of molecular featurizer ('morgan', 'maccs', 'ecfp6', or 'descriptors')
		"""
		if kernel is None:
			# Default to RBF kernel with constant factor and wider bounds to avoid convergence warnings
			kernel = C(1.0, (1e-4, 1e7)) * RBF(1.0, (1e-4, 1e7))
		
		model = GaussianProcessRegressor(
			kernel=kernel,
			alpha=alpha,
			n_restarts_optimizer=2,  # Increased from 0
			normalize_y=True,  # Added for stability
			random_state=random_state
		)
		
		super().__init__(model, results_dir, featurizer_type)
	
	def predict(self, compounds: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Predict scores and uncertainty for compounds.
		
		Args:
			compounds: DataFrame with 'SMILES' column
			
		Returns:
			Tuple of (predictions, uncertainty) where uncertainty contains standard deviations
		"""
		if not self.is_trained:
			raise RuntimeError("Model must be trained before prediction")
		
		# Get molecular representations based on featurizer type
		if self.results_dir is None:
			raise RuntimeError("results_dir must be set to use molecular representations")
		
		if self.featurizer_type in ['morgan', 'maccs', 'ecfp6']:
			from learnm8.utils.featurizers import get_fingerprints
			X = get_fingerprints(compounds['ID'].tolist(), self.results_dir, self.featurizer_type)
		elif self.featurizer_type == 'descriptors':
			from learnm8.utils.featurizers import get_descriptors
			X = get_descriptors(compounds['ID'].tolist(), self.results_dir)
		else:
			raise ValueError(f"Unknown featurizer type: {self.featurizer_type}")
		
		# Make predictions with uncertainty
		try:
			predictions, std = self.model.predict(X, return_std=True)
		except Exception as e:
			# If GP prediction fails, fall back to mean prediction without uncertainty
			from learnm8.utils.logging import get_logger
			logger = get_logger()
			logger.warning(f"GP uncertainty prediction failed: {e}. Returning predictions without uncertainty.")
			predictions = self.model.predict(X)
			std = np.zeros_like(predictions)
		
		from learnm8.utils.logging import get_logger
		logger = get_logger()
		logger.debug(f"Predicted [bold]{len(predictions)}[/bold] compounds with [cyan]{self.get_name()}[/cyan] (with uncertainty)")
		
		return predictions, std
	
	def get_name(self) -> str:
		"""Return the model name."""
		return "Gaussian Process"
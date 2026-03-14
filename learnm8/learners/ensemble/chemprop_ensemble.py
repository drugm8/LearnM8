"""Chemprop ensemble learner for message-passing neural networks."""

from pathlib import Path
import numpy as np
from .ensemble import EnsembleLearner
from ..torch.chemprop_learner import ChempropLearner
from learnm8.exceptions import LearnerError


class ChempropEnsemble(EnsembleLearner):
	"""Ensemble of 3 Chemprop learners for uncertainty quantification.

	Creates an ensemble of 3 ChempropLearner instances with different random seeds
	to provide uncertainty estimates via ensemble disagreement. Optionally accepts
	pre-computed molecular descriptors (x_d) that are passed to all models.

	Uncertainty Methods:
		- 'std': Standard deviation across models (default)
		- 'mad': Median absolute deviation (robust to outliers)
		- 'quantile': Interquartile range

	Features:
		- Ensemble uncertainty from model diversity
		- Inherits from EnsembleLearner (consistent behavior)
		- Optional x_d descriptor support passed to all models
		- Optional incremental fine-tuning from checkpoints
		- SMILES required for all models

	Fine-Tuning:
		When enable_fine_tuning=True, each ensemble member saves separate checkpoints:
		- {checkpoint_dir}/member_0/cycle_N.ckpt
		- {checkpoint_dir}/member_1/cycle_N.ckpt
		- {checkpoint_dir}/member_2/cycle_N.ckpt
	"""

	def __init__(self,
				 message_hidden_dim: int = 300,
				 depth: int = 3,
				 aggregation: str = 'mean',
				 atom_messages: bool = False,
				 batch_norm: bool = False,
				 message_bias: bool = False,
				 ffn_hidden_dim: int = 300,
				 ffn_num_layers: int = 1,
				 dropout: float = 0.0,
				 max_epochs: int = 50,
				 batch_size: int = 32,
				 predict_batch_size: int | None = None,
				 precision: str = 'auto',
				 pin_memory: bool = True,
				 learning_rate: float = 1e-4,
				 random_states: list[int] | None = None,
				 accelerator: str = 'auto',
				 device: str = 'auto',
				 early_stopping: bool = True,
				 early_stopping_patience: int = 10,
				 early_stopping_min_delta: float = 0.0,
				 val_fraction: float = 0.1,
				 enable_fine_tuning: bool = False,
				 checkpoint_dir: Path | None = None,
				 enable_aggressive_gc: bool = True,
				 **kwargs):
		"""Initialize Chemprop ensemble.

		Args:
			message_hidden_dim: Hidden dimension of messages (default: 300)
			depth: Number of message passing steps (default: 3)
			aggregation: Aggregation mode - mean, sum, norm (default: 'mean')
			atom_messages: Pass messages on atoms vs bonds (default: False)
			batch_norm: Turn on batch normalization (default: False)
			message_bias: Add bias to message passing layers (default: False)
			ffn_hidden_dim: FFN hidden dimension (default: 300)
			ffn_num_layers: Number of FFN layers (default: 1)
			dropout: Dropout probability (default: 0.0)
			max_epochs: Maximum training epochs per learner (default: 50)
			batch_size: Training batch size per learner (default: 32)
			predict_batch_size: Prediction batch size per learner (default: None, uses 4x batch_size)
			precision: Precision mode for all members - 'auto', '16-mixed', '32-true', 'bf16-mixed' (default: 'auto')
			pin_memory: Enable pinned memory for all members (default: True)
			learning_rate: Learning rate per learner (default: 1e-4)
			random_states: List of random states for diversity (default: [42, 123, 456])
			accelerator: PyTorch Lightning accelerator (default: 'auto')
			early_stopping: Enable early stopping (default: True)
			early_stopping_patience: Early stopping patience (default: 10)
			early_stopping_min_delta: Minimum delta for improvement (default: 0.0)
			val_fraction: Fraction of data for validation (default: 0.1)
			enable_fine_tuning: Enable checkpoint-based fine-tuning (default: False)
			checkpoint_dir: Directory for checkpoint storage (required if fine-tuning enabled)
			enable_aggressive_gc: Enable automatic GPU memory cleanup for all
				ensemble members and at ensemble level (default: True)
			**kwargs: Additional arguments passed to EnsembleLearner
		"""
		if random_states is None:
			random_states = [42, 123, 456]

		# Validate fine-tuning parameters
		if enable_fine_tuning and checkpoint_dir is None:
			raise ValueError("checkpoint_dir is required when enable_fine_tuning=True")

		# Setup checkpoint subdirectories for ensemble members
		if enable_fine_tuning:
			checkpoint_dir = Path(checkpoint_dir)

		learners = []
		for i, rs in enumerate(random_states):
			# Each ensemble member gets its own checkpoint subdirectory
			member_checkpoint_dir = None
			if enable_fine_tuning:
				member_checkpoint_dir = checkpoint_dir / f'member_{i}'

			chemprop = ChempropLearner(
				message_hidden_dim=message_hidden_dim,
				depth=depth,
				aggregation=aggregation,
				atom_messages=atom_messages,
				batch_norm=batch_norm,
				message_bias=message_bias,
				ffn_hidden_dim=ffn_hidden_dim,
				ffn_num_layers=ffn_num_layers,
				dropout=dropout,
				max_epochs=max_epochs,
				batch_size=batch_size,
				predict_batch_size=predict_batch_size,
				precision=precision,
				pin_memory=pin_memory,
				learning_rate=learning_rate,
				random_state=rs,
				accelerator=accelerator,
				device=device,
				early_stopping=early_stopping,
				early_stopping_patience=early_stopping_patience,
				early_stopping_min_delta=early_stopping_min_delta,
				val_fraction=val_fraction,
				enable_fine_tuning=enable_fine_tuning,
				checkpoint_dir=member_checkpoint_dir,
				enable_aggressive_gc=enable_aggressive_gc
			)
			learners.append(chemprop)

		kwargs.setdefault('aggregation_method', 'mean')
		kwargs.setdefault('uncertainty_method', 'std')

		super().__init__(learners, **kwargs)

		self.message_hidden_dim = message_hidden_dim
		self.depth = depth
		self.aggregation = aggregation
		self.atom_messages = atom_messages
		self.batch_norm = batch_norm
		self.message_bias = message_bias
		self.ffn_hidden_dim = ffn_hidden_dim
		self.ffn_num_layers = ffn_num_layers
		self.dropout = dropout
		self.max_epochs = max_epochs
		self.batch_size = batch_size
		self.predict_batch_size = predict_batch_size
		self.precision = precision
		self.pin_memory = pin_memory
		self.learning_rate = learning_rate
		self.random_states = random_states
		self.accelerator = accelerator
		self.early_stopping = early_stopping
		self.early_stopping_patience = early_stopping_patience
		self.early_stopping_min_delta = early_stopping_min_delta
		self.val_fraction = val_fraction

		# Store fine-tuning configuration
		self.enable_fine_tuning = enable_fine_tuning
		self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
		self.enable_aggressive_gc = enable_aggressive_gc

	def train(self, features, targets, smiles=None):
		"""Train ensemble with SMILES strings.

		Args:
			features: Ignored (ChempropEnsemble works with SMILES directly)
			targets: Target values (n_samples,)
			smiles: SMILES strings (required)
		"""
		if smiles is None:
			raise ValueError("ChempropEnsemble requires SMILES strings")

		if len(smiles) != len(targets):
			raise ValueError(f"SMILES and targets must have same length: {len(smiles)} vs {len(targets)}")

		if len(smiles) == 0:
			raise ValueError("Cannot train on empty dataset")

		# Train each learner with SMILES
		for i, learner in enumerate(self.learners):
			learner.train(features=None, targets=targets, smiles=smiles)

		self.is_trained = True

		self._cleanup_gpu_memory("after ensemble training")

	def predict(self, features, smiles=None):
		"""Predict with SMILES strings.

		Args:
			features: Ignored (ChempropEnsemble works with SMILES directly)
			smiles: SMILES strings (required)

		Returns:
			Tuple of (predictions, uncertainties)
		"""
		if smiles is None:
			raise ValueError("ChempropEnsemble requires SMILES strings")

		if not self.is_trained:
			raise LearnerError("Ensemble must be trained before prediction")

		# Get predictions from each learner
		predictions_list = []
		for learner in self.learners:
			pred, _ = learner.predict(features=None, smiles=smiles)
			predictions_list.append(pred)

		predictions_array = np.array(predictions_list)

		# Aggregate predictions
		ensemble_predictions = self._aggregate_predictions(predictions_array)
		uncertainties = self._calculate_uncertainty(predictions_array)

		self._cleanup_gpu_memory("after ensemble prediction")

		return ensemble_predictions, uncertainties

	def get_individual_predictions(self, features, smiles=None):
		"""Get predictions from individual ensemble members.

		Args:
			features: Ignored (ChempropEnsemble works with SMILES directly)
			smiles: SMILES strings (required)

		Returns:
			Dictionary mapping learner names to their predictions
		"""
		if smiles is None:
			raise ValueError("ChempropEnsemble requires SMILES strings")

		if not self.is_trained:
			raise LearnerError("Ensemble must be trained before prediction")

		individual_predictions = {}
		for i, learner in enumerate(self.learners):
			pred, _ = learner.predict(features=None, smiles=smiles)
			individual_predictions[f"{learner.get_name()}_{i}"] = pred

		return individual_predictions

	def get_name(self) -> str:
		"""Return a descriptive name for this learner."""
		return f"ChempropEnsemble(3xChemprop,depth={self.depth},hidden={self.message_hidden_dim})"

	def requires_smiles(self) -> bool:
		"""ChempropEnsemble requires SMILES strings."""
		return True

	def _cleanup_gpu_memory(self, context: str = "") -> None:
		"""Force garbage collection and clear GPU cache if enabled.

		This method performs two cleanup operations:
		1. torch.cuda.empty_cache() - Releases cached GPU memory
		2. gc.collect() - Forces Python garbage collection

		This is particularly important in active learning scenarios where
		models are trained repeatedly over many cycles, which can lead to
		GPU memory accumulation from unreferenced tensors and PyTorch's
		caching allocator.

		The cleanup is a best-effort operation that won't raise exceptions
		if it fails. It only runs if enable_aggressive_gc=True.

		Args:
			context: Optional description of when cleanup is being called,
					used for debug logging (e.g., "after training")

		Note:
			This is safe to call after predictions have been moved to CPU
			memory via .cpu().numpy(), as it only affects unreferenced
			GPU tensors and Python objects.
		"""
		if not self.enable_aggressive_gc:
			return

		try:
			import gc
			import torch

			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			gc.collect()

			if context:
				import logging
				logger = logging.getLogger(__name__)
				logger.debug(f"GPU memory cleanup: {context}")

		except (RuntimeError, OSError, ImportError) as e:
			import logging
			logger = logging.getLogger(__name__)
			logger.warning(f"GPU memory cleanup failed ({context}): {e}")

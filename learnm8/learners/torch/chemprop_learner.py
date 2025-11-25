from typing import Tuple, Optional, List
from pathlib import Path
import numpy as np
import logging
import warnings

try:
	from chemprop import data, models, nn
	from lightning import pytorch as pl
	CHEMPROP_AVAILABLE = True
except ImportError:
	CHEMPROP_AVAILABLE = False

from learnm8.core.interfaces import Learner

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*", category=UserWarning)

class ChempropLearner(Learner):
	"""Single Chemprop message passing neural network learner.

	This learner works directly with SMILES strings using graph neural networks to learn from
	molecular structure. It can optionally accept pre-computed molecular descriptors (Morgan
	fingerprints, MACCS keys, etc.) as extra features (x_d) concatenated after aggregation.

	Features:
		- Direct SMILES-to-prediction using graph neural networks
		- Optional integration with classical molecular descriptors (x_d)
		- No uncertainty quantification (use ChempropEnsemble for uncertainty)
		- Optional incremental fine-tuning from checkpoints for faster training

	Model Configuration:
		- message_hidden_dim: Hidden dimension of messages (default: 300)
		- depth: Number of message passing steps (default: 3)
		- aggregation: Aggregation mode - mean, sum, norm (default: 'mean')
		- atom_messages: Pass messages on atoms vs bonds (default: False)
		- batch_norm: Turn on batch normalization (default: False)
		- message_bias: Add bias to message passing layers (default: False)
		- ffn_hidden_dim: FFN hidden dimension (default: 300)
		- ffn_num_layers: Number of FFN layers (default: 1)
		- dropout: Dropout probability (default: 0.0)

	Training Configuration:
		- max_epochs: Maximum training epochs (default: 50)
		- batch_size: Training batch size (default: 32)
		- predict_batch_size: Prediction batch size (default: None, uses 4x batch_size)
		- precision: Precision mode - 'auto', '16-mixed', '32-true', 'bf16-mixed' (default: 'auto')
		- pin_memory: Enable pinned memory for GPU transfers (default: True)
		- learning_rate: Learning rate (default: 1e-4)
		- random_state: Random seed (default: 42)
		- accelerator: PyTorch Lightning accelerator (default: 'auto')

	Fine-Tuning Configuration:
		- enable_fine_tuning: Enable checkpoint-based fine-tuning (default: False)
		- checkpoint_dir: Directory for checkpoint storage (required if fine-tuning enabled)

	Memory Management Configuration:
		- enable_aggressive_gc: Enable automatic GPU memory cleanup after training
			and prediction. Recommended for active learning with many cycles.
			Default: True. Disable for maximum performance if GPU memory is abundant.

	Example:
		# Pure graph-based learning
		learner = ChempropLearner(depth=5, message_hidden_dim=500)
		learner.train(features=None, targets=y_train, smiles=smiles_train)
		predictions, _ = learner.predict(features=None, smiles=smiles_test)

		# Hybrid: graph + molecular descriptors
		from learnm8.features.extraction import extract_features
		features = extract_features(smiles_train, 'morgan', cache_dir)
		learner.train(features=features, targets=y_train, smiles=smiles_train)
		test_features = extract_features(smiles_test, 'morgan', cache_dir)
		predictions, _ = learner.predict(features=test_features, smiles=smiles_test)

		# With fine-tuning for active learning
		learner = ChempropLearner(
			enable_fine_tuning=True,
			checkpoint_dir=Path('./checkpoints')
		)
		# First training: trains from scratch, saves checkpoint
		learner.train(features=None, targets=y_train1, smiles=smiles_train1)
		# Second training: loads checkpoint, fine-tunes on expanded dataset
		learner.train(features=None, targets=y_train2, smiles=smiles_train2)
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
				 predict_batch_size: Optional[int] = None,
				 precision: str = 'auto',
				 pin_memory: bool = True,
				 learning_rate: float = 1e-4,
				 random_state: int = 42,
				 accelerator: str = 'auto',
				 early_stopping: bool = True,
				 early_stopping_patience: int = 10,
				 early_stopping_min_delta: float = 0.0,
				 val_fraction: float = 0.1,
				 enable_fine_tuning: bool = False,
				 checkpoint_dir: Optional[Path] = None,
				 enable_aggressive_gc: bool = True):
		if not CHEMPROP_AVAILABLE:
			raise ImportError(
				"Chemprop is required for ChempropLearner. "
				"Install with: pip install chemprop"
			)

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
		self.predict_batch_size = predict_batch_size if predict_batch_size is not None else (batch_size * 4)
		self.precision = self._validate_and_resolve_precision(precision)
		self.pin_memory = pin_memory
		self.learning_rate = learning_rate
		self.random_state = random_state
		self.accelerator = accelerator
		self.early_stopping = early_stopping
		self.early_stopping_patience = early_stopping_patience
		self.early_stopping_min_delta = early_stopping_min_delta
		self.val_fraction = val_fraction

		self.model = None
		self.trainer = None
		self.is_trained = False
		self.training_feature_dim = None
		self.x_d_dim = None

		# Fine-tuning configuration
		self.enable_fine_tuning = enable_fine_tuning
		self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
		self.current_checkpoint_path: Optional[Path] = None
		self.cycle_counter = 0

		# Memory management configuration
		self.enable_aggressive_gc = enable_aggressive_gc

		if self.enable_fine_tuning:
			if self.checkpoint_dir is None:
				raise ValueError(
					"checkpoint_dir is required when enable_fine_tuning=True"
				)
			self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
			logger.info(
				f"Fine-tuning enabled: checkpoints will be saved to {self.checkpoint_dir}"
			)

	def _validate_and_resolve_precision(self, precision: str) -> str:
		"""Validate precision parameter and resolve 'auto' setting."""
		import torch

		valid_precisions = ['auto', '16-mixed', '32-true', 'bf16-mixed', '32', '16']
		if precision not in valid_precisions:
			raise ValueError(
				f"Invalid precision '{precision}'. "
				f"Must be one of: {valid_precisions}"
			)

		if precision == 'auto':
			if torch.cuda.is_available():
				precision = '16-mixed'
				logger.info("Auto-detected GPU: using mixed precision '16-mixed'")
			else:
				precision = '32-true'
				logger.info("No GPU detected: using full precision '32-true'")

		if precision == '32':
			precision = '32-true'
		elif precision == '16':
			precision = '16-mixed'

		return precision

	def requires_smiles(self) -> bool:
		return True

	def train(self,
			  features: Optional[np.ndarray],
			  targets: np.ndarray,
			  smiles: Optional[List[str]] = None) -> None:
		"""Train single Chemprop model with optional early stopping.

		Args:
			features: Optional molecular descriptors to use as x_d (ignored if None)
			targets: Target values
			smiles: Required SMILES strings
		"""

		if smiles is None:
			raise ValueError("ChempropLearner requires SMILES strings")

		warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
		warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*", category=UserWarning)

		n_samples = len(targets)
		use_descriptors = features is not None

		if use_descriptors:
			self.training_feature_dim = features.shape[1]
			self.x_d_dim = features.shape[1]
			logger.debug(
				f"Training Chemprop with {features.shape[1]}-D extra descriptors (x_d) "
				f"on {n_samples} compounds"
			)
		else:
			self.training_feature_dim = None
			self.x_d_dim = 0
			logger.debug(f"Training Chemprop on {n_samples} compounds (graph-only)")

		# Create validation split if early stopping is enabled and we have enough samples
		# Need at least 2 samples to create a train/val split
		min_samples_for_split = max(2, int(2 / self.val_fraction))
		can_create_val_split = n_samples >= min_samples_for_split

		if self.early_stopping and self.val_fraction > 0 and can_create_val_split:
			from sklearn.model_selection import train_test_split

			# Split data for validation
			indices = np.arange(len(smiles))
			train_idx, val_idx = train_test_split(
				indices,
				test_size=self.val_fraction,
				random_state=self.random_state
			)

			# Create train and validation datasets
			train_smiles = [smiles[i] for i in train_idx]
			train_targets = targets[train_idx]
			val_smiles = [smiles[i] for i in val_idx]
			val_targets = targets[val_idx]

			if use_descriptors:
				train_features = features[train_idx]
				val_features = features[val_idx]
				train_data = self._create_dataset_with_descriptors(train_smiles, train_targets, train_features)
				val_data = self._create_dataset_with_descriptors(val_smiles, val_targets, val_features)
			else:
				train_data = self._create_dataset(train_smiles, train_targets)
				val_data = self._create_dataset(val_smiles, val_targets)

			train_loader = data.build_dataloader(
				train_data,
				batch_size=self.batch_size,
				shuffle=True,
				num_workers=0,
				pin_memory=self.pin_memory
			)
			val_loader = data.build_dataloader(
				val_data,
				batch_size=self.batch_size,
				shuffle=False,
				num_workers=0,
				pin_memory=self.pin_memory
			)

			logger.info(f"Using validation split: {len(train_smiles)} train, {len(val_smiles)} val")
		else:
			# No validation split - either disabled or too few samples
			if self.early_stopping and not can_create_val_split:
				logger.warning(
					f"Early stopping requested but insufficient samples ({n_samples}) "
					f"for validation split (need >= {min_samples_for_split}). "
					"Training without early stopping."
				)
			if use_descriptors:
				train_data = self._create_dataset_with_descriptors(smiles, targets, features)
			else:
				train_data = self._create_dataset(smiles, targets)
			train_loader = data.build_dataloader(
				train_data,
				batch_size=self.batch_size,
				shuffle=True,
				num_workers=0,
				pin_memory=self.pin_memory
			)
			val_loader = None

		# Load checkpoint if fine-tuning enabled and checkpoint exists
		if self.enable_fine_tuning and self.current_checkpoint_path is not None:
			if self.current_checkpoint_path.exists():
				try:
					logger.info(f"Loading checkpoint for fine-tuning: {self.current_checkpoint_path}")
					self.model = models.MPNN.load_from_checkpoint(str(self.current_checkpoint_path))
					logger.info("Checkpoint loaded successfully, fine-tuning from previous state")
				except FileNotFoundError:
					logger.warning(f"Checkpoint file not found: {self.current_checkpoint_path}. Training fresh model.")
					self.model = None
				except RuntimeError as e:
					# Catch architecture mismatch errors
					if "state_dict" in str(e).lower() or "unexpected key" in str(e).lower():
						logger.warning(
							f"Checkpoint incompatible with current architecture: {e}. "
							f"Training fresh model."
						)
						self.model = None
					else:
						logger.warning(f"Error loading checkpoint: {e}. Training fresh model.")
						self.model = None
				except Exception as e:
					logger.warning(f"Unexpected error loading checkpoint: {e}. Training fresh model.")
					self.model = None

		# Create fresh model if no checkpoint was loaded
		if self.model is None:
			self.model = self._create_model()

		# Configure callbacks
		callbacks = []
		if self.early_stopping and val_loader is not None:
			from lightning.pytorch.callbacks import EarlyStopping

			early_stop_callback = EarlyStopping(
				monitor='val_loss',
				min_delta=self.early_stopping_min_delta,
				patience=self.early_stopping_patience,
				verbose=True,
				mode='min'
			)
			callbacks.append(early_stop_callback)
			logger.info(f"Early stopping enabled (patience={self.early_stopping_patience})")

		# Enable progress bar and model summary if DEBUG logging is active
		enable_verbose = logging.getLogger().level <= logging.DEBUG

		# Suppress PyTorch Lightning's internal DEBUG logging
		logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

		self.trainer = pl.Trainer(
			max_epochs=self.max_epochs,
			accelerator=self.accelerator,
			precision=self.precision,
			enable_progress_bar=enable_verbose,
			enable_model_summary=False,
			logger=False,
			callbacks=callbacks
		)

		logger.info(
			f"Starting Chemprop training: {n_samples} samples, "
			f"{self.max_epochs} max epochs, precision={self.precision}"
		)
		self.trainer.fit(self.model, train_loader, val_loader)
		logger.info("Chemprop training completed successfully")
		self.is_trained = True

		# Save checkpoint if fine-tuning enabled
		if self.enable_fine_tuning:
			self.cycle_counter += 1
			next_checkpoint_path = self.checkpoint_dir / f"cycle_{self.cycle_counter}.ckpt"

			try:
				self.trainer.save_checkpoint(str(next_checkpoint_path))
				logger.info(f"Checkpoint saved: {next_checkpoint_path}")
				self.current_checkpoint_path = next_checkpoint_path
			except Exception as e:
				logger.error(f"Failed to save checkpoint to {next_checkpoint_path}: {e}")
				# Don't raise - training succeeded, just checkpoint save failed

		logger.debug("Training complete")

		self._cleanup_gpu_memory("after training")

	def predict(self,
				features: Optional[np.ndarray],
				smiles: Optional[List[str]] = None
				) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		"""Predict on SMILES strings (no uncertainty for single model).

		Args:
			features: Optional molecular descriptors (x_d)
			smiles: Required SMILES strings

		Returns:
			predictions: Predicted values
			uncertainties: None (single model has no uncertainty)
		"""

		if smiles is None:
			raise ValueError("ChempropLearner requires SMILES strings")

		if not self.is_trained:
			raise RuntimeError("Model must be trained before prediction")

		if features is not None:
			if self.training_feature_dim is None:
				raise ValueError(
					"Model was trained without descriptors but descriptors provided for prediction"
				)
			if features.shape[1] != self.training_feature_dim:
				raise ValueError(
					f"Feature dimension mismatch: trained with {self.training_feature_dim}-D "
					f"features but got {features.shape[1]}-D for prediction"
				)
		elif self.training_feature_dim is not None:
			raise ValueError(
				"Model was trained with descriptors but none provided for prediction"
			)

		predictions = self._predict_model(smiles, features)
		return predictions, None

	def supports_uncertainty(self) -> bool:
		return False

	def get_name(self) -> str:
		"""Return descriptive learner name."""
		return f"Chemprop(depth={self.depth},hidden={self.message_hidden_dim})"

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
				logger.debug(f"GPU memory cleanup: {context}")

		except Exception as e:
			logger.warning(f"GPU memory cleanup failed ({context}): {e}")

	def _create_dataset(self,
					   smiles: List[str],
					   targets: np.ndarray
					   ) -> data.MoleculeDataset:
		"""Create Chemprop dataset from SMILES and targets."""
		datapoints = []
		for smi, y in zip(smiles, targets):
			dp = data.MoleculeDatapoint.from_smi(smi, y=[y])
			datapoints.append(dp)

		return data.MoleculeDataset(datapoints)

	def _create_dataset_with_descriptors(self,
										 smiles: List[str],
										 targets: np.ndarray,
										 descriptors: np.ndarray
										 ) -> data.MoleculeDataset:
		"""Create Chemprop dataset with extra descriptors (x_d).

		Args:
			smiles: SMILES strings
			targets: Target values
			descriptors: Extra molecular descriptors (n_molecules × n_features)

		Returns:
			MoleculeDataset with x_d descriptors
		"""
		if len(smiles) != len(descriptors):
			raise ValueError(
				f"SMILES count ({len(smiles)}) != descriptor count ({len(descriptors)})"
			)

		datapoints = []
		for smi, y, x_d in zip(smiles, targets, descriptors):
			dp = data.MoleculeDatapoint.from_smi(smi, y=[y], x_d=x_d)
			datapoints.append(dp)

		return data.MoleculeDataset(datapoints)

	def _create_model(self) -> models.MPNN:
		"""Create a single MPNN model.

		Must be called after x_d_dim is set (during training).
		"""
		import torch

		torch.manual_seed(self.random_state)
		np.random.seed(self.random_state)

		if self.atom_messages:
			mp = nn.AtomMessagePassing(
				d_h=self.message_hidden_dim,
				depth=self.depth,
				bias=self.message_bias,
				dropout=self.dropout
			)
		else:
			mp = nn.BondMessagePassing(
				d_h=self.message_hidden_dim,
				depth=self.depth,
				bias=self.message_bias,
				dropout=self.dropout
			)

		if self.aggregation == 'mean':
			agg = nn.MeanAggregation()
		elif self.aggregation == 'sum':
			agg = nn.SumAggregation()
		elif self.aggregation == 'norm':
			agg = nn.NormAggregation()
		else:
			logger.warning(f"Unknown aggregation '{self.aggregation}', using mean")
			agg = nn.MeanAggregation()

		ffn_input_dim = self.message_hidden_dim + self.x_d_dim

		ffn = nn.RegressionFFN(
			n_tasks=1,
			input_dim=ffn_input_dim,
			hidden_dim=self.ffn_hidden_dim,
			n_layers=self.ffn_num_layers,
			dropout=self.dropout
		)

		mpnn = models.MPNN(
			mp,
			agg,
			ffn,
			batch_norm=self.batch_norm,
			metrics=None,
			warmup_epochs=2,
			init_lr=1e-4,
			max_lr=self.learning_rate,
			final_lr=1e-4
		)

		if self.x_d_dim > 0:
			logger.debug(
				f"Created Chemprop model (depth={self.depth}, hidden={self.message_hidden_dim}, "
				f"FFN input={ffn_input_dim} = {self.message_hidden_dim} graph + {self.x_d_dim} x_d)"
			)
		else:
			logger.debug(f"Created Chemprop model (depth={self.depth}, hidden={self.message_hidden_dim})")

		return mpnn

	def _predict_model(self,
					   smiles: List[str],
					   descriptors: Optional[np.ndarray] = None
					   ) -> np.ndarray:
		"""Get predictions from single model.

		Args:
			smiles: SMILES strings
			descriptors: Optional extra descriptors (x_d)

		Returns:
			Predictions array
		"""
		import torch

		if descriptors is not None:
			test_data = self._create_dataset_with_descriptors(
				smiles, np.zeros(len(smiles)), descriptors
			)
		else:
			test_data = self._create_dataset(smiles, np.zeros(len(smiles)))

		test_data.cache = True
		logger.debug("Enabled dataset graph caching for prediction")

		test_loader = data.build_dataloader(
			test_data,
			batch_size=self.predict_batch_size,
			shuffle=False,
			num_workers=0,
			pin_memory=self.pin_memory
		)

		predict_trainer = pl.Trainer(
			accelerator=self.accelerator,
			precision=self.precision,
			enable_progress_bar=False,
			enable_model_summary=False,
			logger=False
		)

		logger.info(
			f"Starting Chemprop prediction on {len(smiles)} samples "
			f"(batch_size={self.predict_batch_size}, precision={self.precision})"
		)
		predictions = predict_trainer.predict(self.model, test_loader)
		logger.info("Chemprop prediction completed, processing results")
		predictions = torch.cat(predictions, dim=0)
		predictions = predictions.cpu().numpy().flatten()

		self._cleanup_gpu_memory("after prediction")
		logger.info(f"Chemprop prediction finalized: {len(predictions)} predictions")

		return predictions

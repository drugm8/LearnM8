"""Ensemble learner implementations for the LearnM8 framework.

This module provides composition-based ensemble methods that combine multiple
learners to improve prediction accuracy and provide uncertainty estimates
through model diversity.
"""

import logging
import time

import numpy as np

from learnm8.exceptions import LearnerError

from ...core.interfaces import Learner

logger = logging.getLogger(__name__)


# Single source of truth for ensemble member seed offsets. Member i's seed is
# ``base_seed + ENSEMBLE_SEED_OFFSETS[i]``. Index ≥ 3 extends deterministically.
ENSEMBLE_SEED_OFFSETS: tuple[int, ...] = (0, 81, 314)


def _derive_random_states(
    base_seed: int,
    n_members: int,
    override: list[int] | None = None,
) -> list[int]:
    """Resolve per-member seeds for an ensemble.

    If ``override`` is provided it is returned unchanged (explicit user
    seeds win). Otherwise returns ``[base + ENSEMBLE_SEED_OFFSETS[i] for i in
    range(n_members)]``; for ``n_members > len(ENSEMBLE_SEED_OFFSETS)``,
    additional offsets are extended by chaining the last offset so two runs
    with the same base seed still produce identical member-seed lists.
    """
    if override is not None:
        return override
    if n_members <= 0:
        return []
    offsets: list[int] = list(ENSEMBLE_SEED_OFFSETS[:n_members])
    while len(offsets) < n_members:
        offsets.append(offsets[-1] + ENSEMBLE_SEED_OFFSETS[-1])
    return [base_seed + off for off in offsets]


class EnsembleLearner(Learner):
    """Meta-learner that combines multiple models through composition.

    This learner implements ensemble methods by combining predictions from
    multiple base learners. It provides uncertainty estimation through
    model diversity and supports various aggregation strategies.
    """

    def __init__(
        self,
        learners: list[Learner],
        aggregation_method: str = 'mean',
        uncertainty_method: str = 'std',
        weights: list[float] | None = None,
        enable_parallel_training: bool = False,
    ):
        """Initialize ensemble learner with composition pattern.

        Args:
            learners: List of base learners to ensemble
            aggregation_method: Method for combining predictions ('mean', 'median', 'weighted')
            uncertainty_method: Method for uncertainty estimation ('std', 'mad', 'quantile')
            weights: Optional weights for weighted aggregation (must sum to 1)
            enable_parallel_training: Whether to train learners in parallel (not implemented)
        """
        if not learners:
            raise LearnerError(
                'EnsembleLearner requires at least one base learner. '
                'Provide a list of Learner instances, e.g., learners=[rf_learner, gp_learner].'
            )

        self.learners = learners
        self.aggregation_method = aggregation_method
        self.uncertainty_method = uncertainty_method
        self.weights = weights
        self.enable_parallel_training = enable_parallel_training
        self.is_trained = False

        # Validate weights if provided
        if weights is not None:
            if len(weights) != len(learners):
                raise LearnerError(
                    f'Number of weights ({len(weights)}) must match number of learners ({len(learners)}). '
                    f'Provide one weight per ensemble member.'
                )
            if not np.isclose(sum(weights), 1.0):
                raise LearnerError(
                    f'Weights must sum to 1.0, got {sum(weights):.4f}. '
                    f'Normalize your weights so they sum to exactly 1.0.'
                )
            self.weights = np.array(weights)

        # Check learner consistency
        self._validate_learners()

        logger.debug(f'Initialized EnsembleLearner with {len(learners)} base learners')

    def _validate_learners(self) -> None:
        """Validate that all learners are compatible."""
        if not all(isinstance(learner, Learner) for learner in self.learners):
            invalid_types = [
                type(m).__name__ for m in self.learners if not isinstance(m, Learner)
            ]
            raise LearnerError(
                f'All ensemble members must be Learner instances. '
                f'Found invalid types: {invalid_types}.'
            )

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train all ensemble learners on feature matrix.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)

        Raises:
            ValueError: If input shapes invalid
            RuntimeError: If training fails for any learner
        """
        if features.shape[0] != targets.shape[0]:
            raise LearnerError(
                f'Feature and target arrays have mismatched lengths: '
                f'{features.shape[0]} features vs {targets.shape[0]} targets. '
                f'Ensure each sample has exactly one target value.'
            )

        if features.shape[0] == 0:
            raise LearnerError(
                f'Cannot train {self.get_name()} on an empty dataset (0 samples). '
                f'This usually means no labeled compounds are available. '
                f'Check that batch_fraction is large enough to select at least one compound.'
            )

        start_time = time.perf_counter()
        logger.debug(
            f'Training ensemble of {len(self.learners)} learners on {len(features)} samples'
        )

        for i, learner in enumerate(self.learners):
            try:
                logger.debug(
                    f'Training ensemble member {i + 1}/{len(self.learners)}: {learner.get_name()}'
                )
                learner.train(features, targets)
            except (
                ValueError,
                RuntimeError,
                TypeError,
                np.linalg.LinAlgError,
                LearnerError,
            ) as e:
                raise LearnerError(
                    f'Ensemble member {learner.get_name()} (index {i}) failed during training: {e}. '
                    f'All ensemble members must train successfully. '
                    f'Check that the training data and featurizer are compatible with all member learners.'
                ) from e

        self.is_trained = True
        train_time = time.perf_counter() - start_time
        logger.debug(
            f'Trained ensemble of {len(self.learners)} learners in {train_time:.2f}s'
        )

    def predict(
        self,
        features: np.ndarray,
        smiles: list[str] | None = None,
        *,
        compute_uncertainty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict on feature matrix with ensemble uncertainty.

        Feature 023 (FR-005, FR-006, D1, D10):
        * Each member receives ``compute_uncertainty`` directly so skip-eligible
          members can elide uncertainty compute.
        * The outer ``_calculate_uncertainty`` std-reduction is short-circuited
          when ``compute_uncertainty=False`` — saves ~4 GB at 100M x N-member
          scale and matches the uniform-contract for ensembles.

        Args:
            features: Feature matrix (n_samples, n_features)
            smiles: Ignored; present for interface compatibility.
            compute_uncertainty: Keyword-only. False → ``(mean, None)``.

        Returns:
            Tuple of (predictions, uncertainties) where uncertainties is
            ``None`` if ``compute_uncertainty=False``.

        Raises:
            RuntimeError: If ensemble is not trained or prediction fails
        """
        del smiles
        if not self.is_trained:
            raise LearnerError(
                f'{self.get_name()} must be trained before prediction. '
                f'Call train() with labeled data first.'
            )

        start_time = time.perf_counter()

        try:
            predictions_list = []

            for i, learner in enumerate(self.learners):
                try:
                    pred, _ = learner.predict(
                        features, compute_uncertainty=compute_uncertainty
                    )
                    predictions_list.append(pred)
                except (
                    ValueError,
                    RuntimeError,
                    TypeError,
                    np.linalg.LinAlgError,
                    LearnerError,
                ) as e:
                    raise LearnerError(
                        f'Ensemble member {learner.get_name()} (index {i}) failed during prediction: {e}. '
                        f'All ensemble members must predict successfully.'
                    ) from e

            predictions_array = np.array(predictions_list)
            ensemble_predictions = self._aggregate_predictions(predictions_array)

            if not compute_uncertainty:
                pred_time = time.perf_counter() - start_time
                logger.debug(
                    f'Ensemble prediction: aggregated '
                    f'{len(predictions_list)} predictions using '
                    f'{self.aggregation_method} in {pred_time:.2f}s'
                )
                return ensemble_predictions, None

            uncertainties = self._calculate_uncertainty(predictions_array)

            logger.debug(
                f'Ensemble prediction: aggregated {len(predictions_list)} predictions using {self.aggregation_method}'
            )

            pred_time = time.perf_counter() - start_time
            logger.debug(
                f'Predicted {len(ensemble_predictions)} samples with {self.get_name()} '
                f'using {len(predictions_list)} learners in {pred_time:.2f}s'
            )

            return ensemble_predictions, uncertainties

        except LearnerError:
            raise
        except (ValueError, RuntimeError, TypeError, np.linalg.LinAlgError) as e:
            logger.error(f'Failed to predict with {self.get_name()}: {e}')
            raise LearnerError(
                f'Ensemble prediction failed for {self.get_name()}: {e}. '
                f'Check that the input features have the same shape as training features.'
            ) from e

    def _aggregate_predictions(self, predictions_array: np.ndarray) -> np.ndarray:
        """Aggregate predictions from ensemble members.

        Args:
            predictions_array: Array of shape (n_learners, n_compounds)

        Returns:
            Aggregated predictions
        """
        if self.aggregation_method == 'mean':
            if self.weights is not None:
                # Weighted average
                valid_weights = self.weights[: len(predictions_array)]
                valid_weights = valid_weights / valid_weights.sum()  # Renormalize
                return np.average(predictions_array, axis=0, weights=valid_weights)
            else:
                # Simple average
                return np.mean(predictions_array, axis=0)

        elif self.aggregation_method == 'median':
            return np.median(predictions_array, axis=0)

        else:
            logger.warning(
                f"Unknown aggregation method '{self.aggregation_method}', using mean"
            )
            return np.mean(predictions_array, axis=0)

    def _calculate_uncertainty(self, predictions_array: np.ndarray) -> np.ndarray:
        """Calculate uncertainty from ensemble variance.

        Args:
            predictions_array: Array of shape (n_learners, n_compounds)

        Returns:
            Uncertainty estimates
        """
        if self.uncertainty_method == 'std':
            # ddof=0 by deliberate choice. The K ensemble members are treated as
            # the *complete population* of models for this run, not a sample
            # drawn from a model distribution — so the population standard
            # deviation (divide by K) is the right estimator and needs no
            # sample-size correction. This is also the NumPy/BoTorch default.
            # Under the alternative "K models as samples" interpretation one
            # would use ddof=1; for small K (e.g. 3) that scales sigma up by
            # sqrt(K/(K-1)) ~ 1.22, which only matters for sigma-nonlinear
            # acquisition (EI/PI), not for rank-based ones (greedy/UCB/topk).
            return np.std(predictions_array, axis=0)

        elif self.uncertainty_method == 'mad':
            # Median absolute deviation (more robust to outliers)
            median_pred = np.median(predictions_array, axis=0)
            return np.median(np.abs(predictions_array - median_pred), axis=0)

        elif self.uncertainty_method == 'quantile':
            # Interquartile range as uncertainty measure
            q75, q25 = np.percentile(predictions_array, [75, 25], axis=0)
            return (q75 - q25) / 2.0  # Half of IQR

        else:
            logger.warning(
                f"Unknown uncertainty method '{self.uncertainty_method}', using std"
            )
            return np.std(predictions_array, axis=0)

    def supports_uncertainty(self) -> bool:
        """Return True since ensemble provides uncertainty through model diversity."""
        return True

    def memory_profile(self, n_features: int) -> dict[str, int | float]:
        profiles = [m.memory_profile(n_features) for m in self.learners]
        return max(
            profiles, key=lambda p: p['bytes_per_sample'] * p['working_multiplier']
        )

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        learner_names = [learner.get_name() for learner in self.learners]

        # Truncate name if too long
        if len(learner_names) <= 3:
            names_str = '+'.join(learner_names)
        else:
            names_str = '+'.join(learner_names[:3]) + f'+{len(learner_names) - 3}more'

        return f'Ensemble({names_str},{self.aggregation_method})'

"""FIL-accelerated Random Forest learner.

Trains using sklearn RandomForestRegressor on CPU (float32) and converts to
RAPIDS FIL for GPU-accelerated inference via predict_per_tree().
"""

import logging
import shutil
import tempfile
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from learnm8.exceptions import LearnerError
from learnm8.learners.base import (
    SklearnLearner,
    _predict_with_oom_retry,
    _preprocess_features,
    _validate_predict_inputs,
    _validate_train_inputs,
)

logger = logging.getLogger(__name__)


class RfFilLearner(SklearnLearner):
    """FIL-accelerated Random Forest for GPU-optimised inference.

    Trains a sklearn RandomForestRegressor on CPU using float32 features and
    converts the fitted forest to RAPIDS Forest Inference Library (FIL) for
    fast GPU batch prediction via predict_per_tree().

    Note:
        float32 training introduces minor numerical divergence from the CPU
        RandomForestLearner (which defaults to float64).  Uncertainty is the
        population std (ddof=0) across per-tree predictions — ordinal (useful
        for acquisition ranking) and not a calibrated predictive interval.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees (None for unlimited).
        min_samples_split: Minimum samples required to split an internal node.
        min_samples_leaf: Minimum samples required at a leaf node.
        max_features: Features to consider at each split.
        random_state: Random seed for reproducibility.
        n_jobs: Parallel jobs for sklearn training (-1 for all cores).
        **kwargs: Additional arguments forwarded to SklearnLearner.

    Raises:
        LearnerError: If RAPIDS cuML or Treelite are not installed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ) -> None:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        super().__init__(model, random_state=random_state, **kwargs)
        self.device = 'cuda'
        self.n_estimators = n_estimators
        self._fil_model = None
        self._fil_tmp_dir: str | None = None

    def get_name(self) -> str:
        return 'rf_fil'

    def memory_profile(self, n_features: int) -> dict[str, int | float]:
        per_tree_output = self.n_estimators * 4
        input_cost = n_features * 4
        bytes_per_sample = input_cost + per_tree_output
        return {
            'bytes_per_sample': bytes_per_sample,
            'working_multiplier': 2.0,
            'fixed_overhead': 0,
        }

    def supports_uncertainty(self) -> bool:
        return True

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train the sklearn RF (float32) then convert to FIL.

        Args:
            features: Feature matrix (n_samples, n_features).
            targets: Target values (n_samples,).

        Raises:
            LearnerError: If inputs are invalid, training fails, or FIL
                conversion fails.
        """
        _validate_train_inputs(features, targets, self.get_name())
        features = features.astype(np.float32)
        super().train(features, targets)
        self._convert_to_fil()

    def _convert_to_fil(self) -> None:
        """Convert the fitted sklearn forest to a RAPIDS FIL model.

        Serialises the Treelite model to a temporary directory and loads it
        into cuML ForestInference.  The temporary directory is always removed
        in the finally block.

        Raises:
            LearnerError: If Treelite or cuML are missing, or if conversion
                fails for any other reason.
        """
        try:
            import treelite
            import treelite.sklearn
        except ImportError as exc:
            raise LearnerError(
                'Treelite >= 4.6 is required for rf_fil. '
                'Install with: pip install "treelite>=4.6,<5.0" or install via conda with RAPIDS.'
            ) from exc

        try:
            from cuml import ForestInference
        except ImportError as exc:
            raise LearnerError(
                'RAPIDS cuML >= 25.04 is required for rf_fil. '
                'Install via conda: conda install -c rapidsai cuml'
            ) from exc

        try:
            import cuml
            _ver = getattr(cuml, '__version__', '')
            _parts = tuple(int(x) for x in _ver.split('.')[:2])
            if _parts < (25, 4):
                raise LearnerError(
                    f'RAPIDS cuML >= 25.04 is required for rf_fil (found {_ver}). '
                    'Update via conda: conda install -c rapidsai cuml>=25.04'
                )
        except (ValueError, TypeError, AttributeError):
            pass

        tmp_dir = tempfile.mkdtemp(prefix='rf_fil_')
        try:
            tl_model = treelite.sklearn.import_model(self.model)
            model_path = tmp_dir + '/model.tl'
            tl_model.serialize(model_path)
            self._fil_model = ForestInference.load(
                model_path,
                is_classifier=False,
            )
            logger.debug('Converted sklearn RF to FIL model')
        except LearnerError:
            raise
        except Exception as exc:
            raise LearnerError(
                f'FIL conversion failed for {self.get_name()}: {exc}. '
                f'Ensure the forest was fitted successfully and that RAPIDS '
                f'cuML and Treelite versions are compatible.'
            ) from exc
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def predict(
        self,
        features: np.ndarray,
        smiles: list[str] | None = None,
        *,
        compute_uncertainty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict with FIL and return per-tree uncertainty.

        Args:
            features: Feature matrix (n_samples, n_features).
            smiles: Ignored; present for interface compatibility.
            compute_uncertainty: Keyword-only (feature 023). When False, the
                ``predict_per_tree`` GPU call is replaced with the fused
                ``model.predict`` and uncertainty is omitted (FR-004
                amendment / D3).

        Returns:
            Tuple of (predictions, uncertainty) where predictions is the mean
            across trees and uncertainty is the population std (ddof=0), or
            ``None`` when ``compute_uncertainty=False``.

        Raises:
            LearnerError: If model is not trained or prediction fails.
        """
        del smiles
        _validate_predict_inputs(self.is_trained, self.get_name())

        t0 = time.perf_counter_ns()
        features, _, _ = _preprocess_features(
            features,
            valid_feature_mask=self._valid_feature_mask,
            remove_zero_variance=self.remove_zero_variance,
            is_training=False,
            feature_type=self._feature_type,
            imputer=self._feature_imputer,
        )
        features = features.astype(np.float32)
        t_preprocess = time.perf_counter_ns() - t0

        if not compute_uncertainty:
            # Skip path uses the fused FIL .predict() — no per-tree return,
            # no (n_samples, n_trees) allocation.
            def predict_mean_fn(chunk: np.ndarray) -> np.ndarray:
                raw = self._fil_model.predict(chunk)
                return np.asarray(raw)

            predictions = _predict_with_oom_retry(features, predict_mean_fn)
            logger.debug(
                '%s predict: preprocess=%dns n_samples=%d',
                self.get_name(),
                t_preprocess,
                features.shape[0],
            )
            return predictions, None

        n_trees = self.model.n_estimators

        def predict_fn(chunk: np.ndarray) -> np.ndarray:
            t_inf_start = time.perf_counter_ns()
            raw = self._fil_model.predict_per_tree(chunk)
            t_inf = time.perf_counter_ns() - t_inf_start
            result = np.asarray(raw)
            t_copy = time.perf_counter_ns() - t_inf_start - t_inf
            logger.debug(
                'FIL predict_per_tree: inference=%dns copy_back=%dns chunk=%d',
                t_inf,
                t_copy,
                len(chunk),
            )
            return result

        t_transfer = time.perf_counter_ns()
        per_tree = _predict_with_oom_retry(features, predict_fn, n_output_cols=n_trees)
        t_total = time.perf_counter_ns() - t_transfer

        logger.debug(
            '%s predict: preprocess=%dns inference+transfer=%dns n_samples=%d',
            self.get_name(),
            t_preprocess,
            t_total,
            features.shape[0],
        )

        predictions = per_tree.mean(axis=1)
        uncertainty = np.std(per_tree, axis=1, ddof=0)
        return predictions, uncertainty

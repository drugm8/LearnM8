from __future__ import annotations

import time

import numpy as np

from learnm8.api import LEARNER_REGISTRY
from learnm8.core.interfaces import Learner
from learnm8.learners import DecisionTreeLearner, LinearRegressionLearner

SUPPLEMENTARY_LEARNERS: dict[str, type[Learner]] = {
    "dt": DecisionTreeLearner,
    "lr": LinearRegressionLearner,
}


def create_learner(learner_type: str, **kwargs) -> Learner:
    device = kwargs.get("device", "cpu")
    print(f"  Creating learner: {learner_type} (device={device})")
    if learner_type in LEARNER_REGISTRY:
        return LEARNER_REGISTRY[learner_type](**kwargs)
    if learner_type in SUPPLEMENTARY_LEARNERS:
        return SUPPLEMENTARY_LEARNERS[learner_type](**kwargs)
    available = sorted(set(LEARNER_REGISTRY) | set(SUPPLEMENTARY_LEARNERS))
    raise ValueError(
        f"Unknown learner type '{learner_type}'. Available: {', '.join(available)}"
    )


def train_learner(
    learner: Learner,
    features: np.ndarray,
    targets: np.ndarray,
    smiles: list[str] | None = None,
) -> Learner:
    n_train = len(targets)
    n_features = features.shape[1] if features.ndim > 1 else 1
    print(f"  Training on {n_train} samples, {n_features} features...")
    start = time.perf_counter()
    if learner.requires_smiles():
        learner.train(features, targets, smiles=smiles)
    else:
        learner.train(features, targets)
    elapsed = time.perf_counter() - start
    print(f"  Training complete in {elapsed:.2f}s")
    return learner


def predict_primary_output(
    learner: Learner,
    features: np.ndarray,
    *,
    preprocessed: np.ndarray | None = None,
) -> np.ndarray:
    from learnm8.learners.base import TorchLearner, _preprocess_features

    if preprocessed is None:
        preprocessed, _ = _preprocess_features(
            features,
            valid_feature_mask=getattr(learner, "_valid_feature_mask", None),
            remove_zero_variance=getattr(learner, "remove_zero_variance", True),
            is_training=False,
        )

    if isinstance(learner, TorchLearner):
        import torch

        scaled = learner.scaler.transform(preprocessed)
        x_tensor = torch.FloatTensor(scaled).to(learner.device)
        learner.model.eval()
        with torch.no_grad():
            predictions = learner.model(x_tensor).cpu().numpy().squeeze()
        return np.atleast_1d(predictions)

    predictions = learner.model.predict(preprocessed)
    return np.atleast_1d(np.asarray(predictions).squeeze())

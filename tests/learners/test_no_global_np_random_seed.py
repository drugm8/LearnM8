"""Spec 021 / US1 / FR-001 — no global numpy RNG mutations in TorchLearner.

Verifies that ``TorchLearner._split_validation`` does not call
``np.random.seed(...)`` (which mutates the process-global RNG state) and that
two learners constructed back-to-back are independent of each other.
"""

import numpy as np
import pytest

from learnm8.learners.torch.mlp import MLPLearner

pytestmark = pytest.mark.unit


def _make_dataset(n: int = 32, d: int = 8, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(dtype)
    y = rng.standard_normal(n).astype(dtype)
    return X, y


def test_split_validation_does_not_mutate_global_rng_state():
    learner = MLPLearner(random_state=42)
    X, y = _make_dataset()

    np.random.seed(123)
    expected_next = np.random.random(8)

    np.random.seed(123)
    learner._split_validation(X, y, val_fraction=0.2)
    actual_next = np.random.random(8)

    np.testing.assert_array_equal(expected_next, actual_next)


def test_two_learners_back_to_back_produce_order_independent_splits():
    X, y = _make_dataset()

    a1 = MLPLearner(random_state=42)
    b1 = MLPLearner(random_state=42)
    Xa1_train, _, _, _ = a1._split_validation(X, y, val_fraction=0.2)
    Xb1_train, _, _, _ = b1._split_validation(X, y, val_fraction=0.2)

    b2 = MLPLearner(random_state=42)
    a2 = MLPLearner(random_state=42)
    Xb2_train, _, _, _ = b2._split_validation(X, y, val_fraction=0.2)
    Xa2_train, _, _, _ = a2._split_validation(X, y, val_fraction=0.2)

    np.testing.assert_array_equal(Xa1_train, Xa2_train)
    np.testing.assert_array_equal(Xb1_train, Xb2_train)


def test_repeated_split_validation_calls_are_deterministic():
    learner = MLPLearner(random_state=42)
    X, y = _make_dataset()

    Xt1, Xv1, yt1, yv1 = learner._split_validation(X, y, val_fraction=0.2)
    Xt2, Xv2, yt2, yv2 = learner._split_validation(X, y, val_fraction=0.2)

    np.testing.assert_array_equal(Xt1, Xt2)
    np.testing.assert_array_equal(Xv1, Xv2)
    np.testing.assert_array_equal(yt1, yt2)
    np.testing.assert_array_equal(yv1, yv2)

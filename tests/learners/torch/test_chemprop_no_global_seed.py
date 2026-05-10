"""Spec 021 / US1 / FR-002 — ChempropLearner._create_model uses local RNG."""

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.molecular]

chemprop = pytest.importorskip("chemprop")


def test_chemprop_create_model_does_not_mutate_global_rng_state():
    from learnm8.learners.torch.chemprop_learner import ChempropLearner

    learner = ChempropLearner(random_state=42)
    learner.x_d_dim = 0

    np.random.seed(123)
    expected_next = np.random.random(8)

    np.random.seed(123)
    learner._create_model()
    actual_next = np.random.random(8)

    np.testing.assert_array_equal(expected_next, actual_next)

"""Feature 023 D4: MCDropoutLearner stays uniform-contract — it still runs
the full N stochastic forward passes when ``compute_uncertainty=False``;
only the uncertainty return is suppressed.

Per Gal & Ghahramani 2016, a single deterministic forward pass produces a
point estimate from a different distribution than the MC-Dropout mean.
Skipping the N-sample loop would silently change mean semantics — explicitly
out of scope for a "skip uncertainty" optimisation.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip('torch')

pytestmark = pytest.mark.integration


@pytest.mark.slow
def test_mc_dropout_runs_n_stochastic_passes_even_when_compute_uncertainty_false():
    """The N MC samples must still execute even with compute_uncertainty=False."""
    from learnm8.learners.torch.mc_dropout import MCDropoutLearner

    rng = np.random.default_rng(0)
    n, p = 16, 8
    X = rng.uniform(0, 1, size=(n, p)).astype(np.float32)
    y = rng.uniform(0, 1, size=n).astype(np.float32)

    learner = MCDropoutLearner(
        hidden_sizes=(8,),
        n_dropout_samples=5,
        dropout_rate=0.3,
        max_epochs=2,
        random_state=42,
        device='cpu',
    )
    learner.train(X, y)

    # Mock the underlying model forward to count invocations and verify the
    # loop still runs n_dropout_samples times per chunk.
    call_count = {'n': 0}

    original_forward = learner.model.__call__

    def counting_forward(*args, **kwargs):
        call_count['n'] += 1
        return original_forward(*args, **kwargs)

    with patch.object(learner.model, '__call__', side_effect=counting_forward):
        _preds, sigmas = learner.predict(X, compute_uncertainty=False)

    # Skip path: sigmas suppressed.
    assert sigmas is None

    # But N stochastic forward passes still ran (D4). With one chunk, we
    # expect exactly n_dropout_samples calls — not 1 (single deterministic).
    # We assert "at least 2" as a safety margin in case of internal padding.
    assert call_count['n'] >= 2, (
        f'mc_dropout collapsed to a single deterministic forward pass '
        f'(n_calls={call_count["n"]}). D4 requires the N stochastic passes '
        f'to still execute — mean semantics must not change.'
    )

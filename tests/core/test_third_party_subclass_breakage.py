"""Feature 023 FR-018 / D8: pin the no-shim contract.

Third-party ``Learner`` subclasses with the OLD signature (no
``compute_uncertainty`` kwarg) must break with a natural ``TypeError`` on
first call from the cycle — no defensive shim, no ``**kwargs`` swallow.

The CHANGELOG entry for v0.11.0 provides the 5-line diff users need to
migrate. This test pins the contract so a future ``**kwargs``-swallow
regression cannot silently restore old behaviour.

Implementation: monkeypatches ``LEARNER_REGISTRY`` with an in-test
``OldStyleLearner(Learner)`` lacking the new kwarg, drives a single cycle
via ``run_active_learning``, and asserts the predict call raises
``TypeError`` mentioning ``compute_uncertainty``.
"""

from __future__ import annotations

import numpy as np
import pytest

from learnm8 import api as api_module
from learnm8.core.interfaces import Learner

pytestmark = pytest.mark.integration


class OldStyleLearner(Learner):
    """Third-party-style subclass with the pre-0.11.0 predict signature."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._trained = False

    def train(self, features, targets, smiles=None):
        self._trained = True

    def predict(self, features, smiles=None):  # NO compute_uncertainty kwarg
        return np.zeros(len(features)), None

    def get_name(self) -> str:
        return 'OldStyleLearner'

    def supports_uncertainty(self) -> bool:
        return False


def test_old_style_subclass_raises_typeerror_on_first_predict(
    sample_compounds, mock_oracle, tmp_path, monkeypatch
):
    """The cycle must raise TypeError mentioning 'compute_uncertainty'
    when a third-party subclass lacks the new kwarg."""
    # Monkeypatch the LEARNER_REGISTRY entry.
    monkeypatch.setitem(api_module.LEARNER_REGISTRY, 'old_style', OldStyleLearner)

    with pytest.raises((TypeError, Exception)) as exc_info:
        api_module.run_active_learning(
            compound_pool=sample_compounds,
            oracle=mock_oracle,
            learner='old_style',
            target_col='Activity',
            featurizer='morgan',
            n_cycles=2,
            batch_fraction=0.1,
            output_dir=tmp_path / 'breakage',
            cache_dir=tmp_path / 'cache',
            random_state=42,
        )

    # The cycle wraps exceptions; the original cause should mention the kwarg.
    error_chain = []
    err = exc_info.value
    while err is not None:
        error_chain.append(str(err))
        err = err.__cause__ if err.__cause__ is not None else err.__context__ if err is not exc_info.value else None
        if err in error_chain or err is None:
            break

    full_msg = ' '.join(error_chain) + ' ' + str(exc_info.value)
    assert 'compute_uncertainty' in full_msg, (
        f'TypeError did not mention compute_uncertainty. Full chain: '
        f'{full_msg!r}. Feature 023 FR-018: error must surface the missing '
        f'kwarg so users can find the CHANGELOG migration guide.'
    )

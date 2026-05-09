"""End-to-end ``run_active_learning`` dtype routing test (T022, spec 017).

Captures the dtype that arrives at ``Learner.train`` via monkeypatch:
- ``rf`` + ``morgan`` (binary) → uint8 features arrive at the learner.
- ``gp`` + ``morgan`` (binary) → float32 (continuous-only learners stay on
  the existing path).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learnm8 import run_active_learning
from learnm8.learners.sklearn.gaussian_process import GaussianProcessLearner
from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.oracles.csv_oracle import CSVOracle


@pytest.mark.integration
@pytest.mark.parametrize(
    'learner_name, expected_dtype, learner_cls',
    [
        ('rf', np.uint8, RandomForestLearner),
        ('gp', np.float32, GaussianProcessLearner),
    ],
)
def test_learner_train_dtype_round_trip(
    diverse_real_compounds, tmp_path: Path, monkeypatch, learner_name, expected_dtype,
    learner_cls,
):
    seen: dict[str, np.dtype] = {}
    original_train = learner_cls.train

    def _spy_train(self, features, *args, **kwargs):
        seen['dtype'] = features.dtype
        return original_train(self, features, *args, **kwargs)

    monkeypatch.setattr(learner_cls, 'train', _spy_train)

    pool = diverse_real_compounds.head(60)

    oracle_path = tmp_path / 'oracle.csv'
    pool.select(['ID', 'SMILES', 'Activity']).write_csv(oracle_path)

    run_active_learning(
        compound_pool=pool.select(['ID', 'SMILES']),
        oracle=CSVOracle(str(oracle_path)),
        learner=learner_name,
        target_col='Activity',
        featurizer='morgan',
        n_cycles=2,
        batch_fraction=0.05,
        cache_dir=tmp_path / '.cache',
        output_dir=tmp_path / 'run',
    )

    assert 'dtype' in seen, f"{learner_cls.__name__}.train was never called"
    assert seen['dtype'] == np.dtype(expected_dtype), (
        f"{learner_name} expected feature dtype {expected_dtype}, got {seen['dtype']}"
    )

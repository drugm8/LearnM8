"""Tests for specific exception handling (ERR-004).

Verifies that:
1. Unexpected errors (MemoryError) propagate uncaught — not wrapped as domain errors
2. Expected errors (ValueError, RuntimeError, etc.) get properly wrapped
3. Graceful fallback blocks log at WARNING, not silently swallow
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from learnm8.exceptions import (
    ConfigurationError,
    FeatureExtractionError,
    LearnerError,
    OracleError,
)

pytestmark = [pytest.mark.unit]


class TestUnexpectedErrorsPropagateFromSklearnTrain:

    def test_memory_error_propagates(self):
        from learnm8.learners.base import SklearnLearner

        learner = SklearnLearner.__new__(SklearnLearner)
        learner.remove_zero_variance = False
        learner._valid_feature_mask = None
        learner.is_trained = False
        learner.model = MagicMock()
        learner.model.fit.side_effect = MemoryError('out of memory')

        with pytest.raises(MemoryError, match='out of memory'):
            learner.train(np.array([[1.0, 2.0]]), np.array([1.0]))

    def test_keyboard_interrupt_propagates(self):
        from learnm8.learners.base import SklearnLearner

        learner = SklearnLearner.__new__(SklearnLearner)
        learner.remove_zero_variance = False
        learner._valid_feature_mask = None
        learner.is_trained = False
        learner.model = MagicMock()
        learner.model.fit.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            learner.train(np.array([[1.0, 2.0]]), np.array([1.0]))

    def test_value_error_wrapped_as_learner_error(self):
        from learnm8.learners.base import SklearnLearner

        learner = SklearnLearner.__new__(SklearnLearner)
        learner.remove_zero_variance = False
        learner._valid_feature_mask = None
        learner.is_trained = False
        learner.model = MagicMock()
        learner.model.fit.side_effect = ValueError('bad input shape')

        with pytest.raises(LearnerError, match='bad input shape'):
            learner.train(np.array([[1.0, 2.0]]), np.array([1.0]))

    def test_linalg_error_wrapped_as_learner_error(self):
        from learnm8.learners.base import SklearnLearner

        learner = SklearnLearner.__new__(SklearnLearner)
        learner.remove_zero_variance = False
        learner._valid_feature_mask = None
        learner.is_trained = False
        learner.model = MagicMock()
        learner.model.fit.side_effect = np.linalg.LinAlgError('singular matrix')

        with pytest.raises(LearnerError, match='singular matrix'):
            learner.train(np.array([[1.0, 2.0]]), np.array([1.0]))


class TestUnexpectedErrorsPropagateFromSklearnPredict:

    def test_memory_error_propagates(self):
        from learnm8.learners.base import SklearnLearner

        learner = SklearnLearner.__new__(SklearnLearner)
        learner.remove_zero_variance = False
        learner._valid_feature_mask = None
        learner.is_trained = True
        learner.model = MagicMock()
        learner.model.predict.side_effect = MemoryError('out of memory')

        with pytest.raises(MemoryError, match='out of memory'):
            learner.predict(np.array([[1.0, 2.0]]))

    def test_runtime_error_wrapped_as_learner_error(self):
        from learnm8.learners.base import SklearnLearner

        learner = SklearnLearner.__new__(SklearnLearner)
        learner.remove_zero_variance = False
        learner._valid_feature_mask = None
        learner.is_trained = True
        learner.model = MagicMock()
        learner.model.predict.side_effect = RuntimeError('computation failed')

        with pytest.raises(LearnerError, match='computation failed'):
            learner.predict(np.array([[1.0, 2.0]]))


class TestEnsembleSpecificExceptions:

    def test_memory_error_propagates_from_train(self):
        from learnm8.learners.ensemble.ensemble import EnsembleLearner

        member = MagicMock()
        member.train.side_effect = MemoryError('out of memory')
        member.get_name.return_value = 'MockLearner'

        ensemble = EnsembleLearner.__new__(EnsembleLearner)
        ensemble.learners = [member]
        ensemble.is_trained = False
        ensemble.weights = None
        ensemble.aggregation_method = 'mean'

        with pytest.raises(MemoryError, match='out of memory'):
            ensemble.train(np.array([[1.0, 2.0]]), np.array([1.0]))

    def test_memory_error_propagates_from_predict_inner(self):
        from learnm8.learners.ensemble.ensemble import EnsembleLearner

        member = MagicMock()
        member.predict.side_effect = MemoryError('out of memory')
        member.get_name.return_value = 'MockLearner'

        ensemble = EnsembleLearner.__new__(EnsembleLearner)
        ensemble.learners = [member]
        ensemble.is_trained = True
        ensemble.weights = None
        ensemble.aggregation_method = 'mean'

        with pytest.raises(MemoryError, match='out of memory'):
            ensemble.predict(np.array([[1.0, 2.0]]))

    def test_value_error_handled_gracefully_in_train(self):
        from learnm8.learners.ensemble.ensemble import EnsembleLearner

        good_member = MagicMock()
        good_member.get_name.return_value = 'GoodLearner'
        bad_member = MagicMock()
        bad_member.train.side_effect = ValueError('bad data')
        bad_member.get_name.return_value = 'BadLearner'

        ensemble = EnsembleLearner.__new__(EnsembleLearner)
        ensemble.learners = [good_member, bad_member]
        ensemble.is_trained = False
        ensemble.weights = None
        ensemble.aggregation_method = 'mean'

        with pytest.raises(LearnerError, match='BadLearner'):
            ensemble.train(np.array([[1.0, 2.0]]), np.array([1.0]))

    def test_value_error_handled_gracefully_in_predict(self):
        from learnm8.learners.ensemble.ensemble import EnsembleLearner

        good_member = MagicMock()
        good_member.predict.return_value = (np.array([1.0]), None)
        good_member.get_name.return_value = 'GoodLearner'
        bad_member = MagicMock()
        bad_member.predict.side_effect = ValueError('bad predict')
        bad_member.get_name.return_value = 'BadLearner'

        ensemble = EnsembleLearner.__new__(EnsembleLearner)
        ensemble.learners = [good_member, bad_member]
        ensemble.is_trained = True
        ensemble.weights = None
        ensemble.aggregation_method = 'mean'
        ensemble.uncertainty_method = 'std'

        with pytest.raises(LearnerError, match='BadLearner'):
            ensemble.predict(np.array([[1.0, 2.0]]))


class TestGracefulFallbacksLogWarnings:

    def test_eval_metrics_fallback_logs_warning(self, caplog):
        from learnm8.core.initialization import calculate_initialization_metrics

        with patch(
            'learnm8.evaluation.evaluate_cycle',
            side_effect=ValueError('bad metric calc'),
        ):
            compounds = pl.DataFrame({
                'ID': ['c1'],
                'SMILES': ['C'],
                'status': ['labeled'],
                'Activity': [1.0],
            })

            logging.getLogger('learnm8').propagate = True
            with caplog.at_level(logging.WARNING, logger='learnm8.core.initialization'):
                calculate_initialization_metrics(
                    compounds_df=compounds,
                    selected_ids=['c1'],
                    target_col='Activity',
                    strategy='random',
                    score_direction='higher',
                    mode='benchmark',
                    original_pool=compounds,
                    cumulative_selected_ids={'c1'},
                )

            assert any('Failed to calculate' in r.message for r in caplog.records)


class TestPruningSpecificExceptions:

    def test_memory_error_propagates_from_strategy_creation(self):
        from learnm8.pruning.utils import create_pruning_strategy

        with patch(
            'learnm8.pruning.score_based.ScoreBasedPruner',
            side_effect=MemoryError('out of memory'),
        ), pytest.raises(MemoryError, match='out of memory'):
            create_pruning_strategy('score', {})

    def test_value_error_wrapped_properly(self):
        from learnm8.pruning.utils import create_pruning_strategy

        with patch(
            'learnm8.pruning.score_based.ScoreBasedPruner',
            side_effect=ValueError('bad param'),
        ), pytest.raises(ValueError, match='bad param'):
            create_pruning_strategy('score', {})


class TestFeatureExtractionSpecificExceptions:

    def test_memory_error_propagates(self):
        from learnm8.features.base import SkfpFeaturizer

        with patch.object(SkfpFeaturizer, '__abstractmethods__', set()):
            featurizer = SkfpFeaturizer.__new__(SkfpFeaturizer)
            featurizer.mol_from_smiles = MagicMock()
            featurizer.fingerprint = MagicMock()
            featurizer.fingerprint.transform.side_effect = MemoryError('out of memory')
            featurizer.fingerprint.requires_conformers = False
            featurizer.conformer_gen = None

            with pytest.raises(MemoryError, match='out of memory'):
                featurizer.transform(['C', 'CC'])

    def test_value_error_wrapped_as_feature_error(self):
        from learnm8.features.base import SkfpFeaturizer

        with patch.object(SkfpFeaturizer, '__abstractmethods__', set()):
            featurizer = SkfpFeaturizer.__new__(SkfpFeaturizer)
            featurizer.mol_from_smiles = MagicMock()
            featurizer.fingerprint = MagicMock()
            featurizer.fingerprint.transform.side_effect = ValueError('bad smiles')
            featurizer.fingerprint.requires_conformers = False
            featurizer.conformer_gen = None

            with pytest.raises(FeatureExtractionError, match='bad smiles'):
                featurizer.transform(['C', 'CC'])


class TestOracleSpecificExceptions:

    def test_memory_error_propagates_from_python_oracle(self):
        from learnm8.oracles.python_oracle import PythonOracle

        oracle = PythonOracle.__new__(PythonOracle)
        oracle.oracle_function = MagicMock(side_effect=MemoryError('out of memory'))

        compounds = pl.DataFrame({'ID': ['c1']})

        with pytest.raises(MemoryError, match='out of memory'):
            oracle.measure(compounds, ['Activity'])

    def test_runtime_error_wrapped_as_oracle_error(self):
        from learnm8.oracles.python_oracle import PythonOracle

        oracle = PythonOracle.__new__(PythonOracle)
        oracle.oracle_function = MagicMock(side_effect=RuntimeError('service down'))

        compounds = pl.DataFrame({'ID': ['c1']})

        with pytest.raises(OracleError, match='service down'):
            oracle.measure(compounds, ['Activity'])


class TestApiSpecificExceptions:

    def test_memory_error_propagates_from_learner_creation(self):
        from learnm8.api import _create_learner

        mock_class = MagicMock(side_effect=MemoryError('out of memory'))
        mock_class.__name__ = 'MockLearner'

        with patch('learnm8.api.LEARNER_REGISTRY', {'test': mock_class}), \
                pytest.raises(MemoryError, match='out of memory'):
            _create_learner('test', random_state=42)

    def test_type_error_wrapped_as_configuration_error(self):
        from learnm8.api import _create_learner

        mock_class = MagicMock(side_effect=TypeError('bad arg'))
        mock_class.__name__ = 'MockLearner'

        with patch('learnm8.api.LEARNER_REGISTRY', {'test': mock_class}), \
                pytest.raises(ConfigurationError, match='bad arg'):
            _create_learner('test', random_state=42)

"""Tests for proper exception chaining (ERR-007).

Verifies:
- C library errors (RDKit, skfp, PyTorch/Lightning) use `from None` to suppress internals
- Python-native errors (FileNotFoundError, ValueError) use `from e` to preserve chain
- No bare domain error raises in except blocks without explicit chaining
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from learnm8.exceptions import (
    FeatureExtractionError,
    LearnerError,
)

pytestmark = [pytest.mark.unit]


class TestFeatureExtractionChaining:
    """FR-002: features/extraction.py wraps featurizer errors with from None."""

    def test_featurizer_transform_error_suppresses_cause(self):
        """When featurizer.transform() raises, FeatureExtractionError has no __cause__."""
        from learnm8.features.extraction import _extract_features_with_featurizer

        mock_featurizer = MagicMock()
        mock_featurizer.get_name.return_value = 'MockFP'
        mock_featurizer.get_dimension.return_value = 2048
        mock_featurizer.get_config.return_value = {'test': True}
        mock_featurizer.get_config_hash.return_value = 'mockconfig'
        mock_featurizer.get_storage_dtype.return_value = 'packed_uint8'
        mock_featurizer.transform.side_effect = RuntimeError('skfp internal error')

        with pytest.raises(FeatureExtractionError) as exc_info:
            _extract_features_with_featurizer(
                ['CCO', 'CCC'], mock_featurizer, n_jobs=1
            )

        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True

    def test_featurizer_valueerror_suppresses_cause(self):
        """ValueError from C library also suppressed."""
        from learnm8.features.extraction import _extract_features_with_featurizer

        mock_featurizer = MagicMock()
        mock_featurizer.get_name.return_value = 'MockFP'
        mock_featurizer.get_dimension.return_value = 2048
        mock_featurizer.get_config.return_value = {'test': True}
        mock_featurizer.get_config_hash.return_value = 'mockconfig'
        mock_featurizer.get_storage_dtype.return_value = 'packed_uint8'
        mock_featurizer.transform.side_effect = ValueError('RDKit parse error')

        with pytest.raises(FeatureExtractionError) as exc_info:
            _extract_features_with_featurizer(
                ['CCO'], mock_featurizer, n_jobs=1
            )

        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True


class TestFastpropChainingTrain:
    """FR-004: fastprop_learner.py wraps Lightning train errors with from None."""

    def test_training_error_suppresses_lightning_cause(self):
        """Lightning errors during training should have __cause__ = None."""
        from learnm8.learners.torch.fastprop_learner import FastpropLearner

        learner = FastpropLearner(max_epochs=1)

        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        targets = np.array([0.5, 0.6])

        with patch.object(learner, '_validate_and_resolve_precision', return_value='32-true'):
            pass

        with patch('learnm8.learners.torch.fastprop_learner.Trainer') as mock_trainer_cls:
            mock_trainer = MagicMock()
            mock_trainer.fit.side_effect = RuntimeError('CUDA error: device-side assert')
            mock_trainer_cls.return_value = mock_trainer

            with pytest.raises(LearnerError) as exc_info:
                learner.train(features, targets)

            assert exc_info.value.__cause__ is None
            assert exc_info.value.__suppress_context__ is True


class TestFastpropChainingPredict:
    """FR-004: fastprop_learner.py wraps Lightning predict errors with from None."""

    def test_prediction_error_suppresses_lightning_cause(self):
        """Lightning errors during prediction should have __cause__ = None."""
        from learnm8.learners.torch.fastprop_learner import FastpropLearner

        learner = FastpropLearner(max_epochs=1)
        learner.is_trained = True
        learner.model = MagicMock()
        learner.feature_means = MagicMock()
        learner.feature_vars = MagicMock()

        features = np.array([[1.0, 2.0]])

        with patch('learnm8.learners.torch.fastprop_learner.Trainer') as mock_trainer_cls:
            mock_trainer = MagicMock()
            mock_trainer.predict.side_effect = RuntimeError('CUDA error')
            mock_trainer_cls.return_value = mock_trainer

            with patch('learnm8.learners.torch.fastprop_learner.standard_scale', return_value=MagicMock()):
                with pytest.raises(LearnerError) as exc_info:
                    learner.predict(features)

                assert exc_info.value.__cause__ is None
                assert exc_info.value.__suppress_context__ is True


class TestChempropChainingTrain:
    """FR-003: chemprop_learner.py wraps Lightning train errors with from None."""

    def test_training_error_suppresses_lightning_cause(self):
        """Lightning errors during Chemprop training should have __cause__ = None."""
        from learnm8.learners.torch.chemprop_learner import ChempropLearner

        learner = ChempropLearner(max_epochs=1)

        targets = np.array([0.5, 0.6])
        smiles = ['CCO', 'CCC']

        with (
            patch.object(learner, '_create_model', return_value=MagicMock()),
            patch('learnm8.learners.torch.chemprop_learner.data') as mock_data,
            patch('learnm8.learners.torch.chemprop_learner.pl.Trainer') as mock_trainer_cls,
        ):
            mock_data.MoleculeDatapoint.from_smi.return_value = MagicMock()
            mock_data.MoleculeDataset.return_value = MagicMock()
            mock_data.build_dataloader.return_value = MagicMock()

            mock_trainer = MagicMock()
            mock_trainer.fit.side_effect = RuntimeError('Lightning internal error')
            mock_trainer_cls.return_value = mock_trainer

            with pytest.raises(LearnerError) as exc_info:
                learner.train(features=None, targets=targets, smiles=smiles)

            assert exc_info.value.__cause__ is None
            assert exc_info.value.__suppress_context__ is True


class TestSelectCompoundsChaining:
    """cycle.py _select_compounds chaining."""

    def test_unknown_strategy_suppresses_keyerror(self):
        """KeyError from registry lookup should be suppressed (from None)."""
        import polars as pl

        from learnm8.core.cycle import _select_compounds

        pool = pl.DataFrame({
            'ID': ['C1', 'C2'],
            'prediction': [0.5, 0.6],
        })

        with pytest.raises(ValueError) as exc_info:
            _select_compounds(pool, 'nonexistent_strategy', 2, 'higher', {})

        assert exc_info.value.__suppress_context__ is True

    def test_acquisition_creation_preserves_cause(self):
        """ValueError during acquisition creation should preserve chain (from e)."""
        import polars as pl

        from learnm8.core.cycle import _select_compounds

        pool = pl.DataFrame({
            'ID': ['C1', 'C2'],
            'prediction': [0.5, 0.6],
        })

        with patch('learnm8.acquisition.get_acquisition_function') as mock_get:
            mock_class = MagicMock()
            mock_class.side_effect = TypeError('bad param')
            mock_get.return_value = mock_class

            with pytest.raises(ValueError) as exc_info:
                _select_compounds(pool, 'greedy', 2, 'higher', {})

            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, TypeError)


class TestPersistenceChainingPreserved:
    """FR-005: persistence.py preserves chain with from e."""

    def test_save_results_preserves_ioerror_chain(self):
        """IOError during save should preserve chain."""
        from pathlib import Path

        import polars as pl

        from learnm8.core.persistence import save_results
        from learnm8.core.validation import ValidationResult
        from learnm8.exceptions import PersistenceError

        compounds_df = pl.DataFrame({
            'ID': ['C1'],
            'SMILES': ['CCO'],
            'status': ['labeled'],
            'labeled_cycle': [0],
            'selected_cycle': [0],
            'pruned_cycle': [None],
            'Activity': [0.5],
        })

        validation_result = ValidationResult(
            valid_compounds=compounds_df,
            invalid_compounds=pl.DataFrame(schema={'ID': pl.Utf8, 'SMILES': pl.Utf8}),
            validation_errors={},
        )

        with patch('polars.DataFrame.write_csv', side_effect=OSError('disk full')):
            with pytest.raises(PersistenceError) as exc_info:
                save_results(
                    compounds_df, [], validation_result,
                    {'target_col': 'Activity'}, Path('/tmp/test_out')
                )

            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, IOError)

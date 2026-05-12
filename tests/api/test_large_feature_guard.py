"""Tests for check_large_feature_guard() and its integration into run_active_learning()."""

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from learnm8.exceptions import ConfigurationError, LearnM8Warning
from learnm8.features.guard import check_large_feature_guard


class MockContinuousFeaturizer:
    """Mock continuous featurizer with configurable dimension, dtype, and name."""

    feature_type = 'continuous'

    def __init__(self, n_features=1613, itemsize=4, name='MockContinuous'):
        self._n_features = n_features
        dtype_mock = MagicMock()
        dtype_mock.itemsize = itemsize
        self.fingerprint = MagicMock(dtype=dtype_mock)
        self.__class__.__name__ = name

    def get_dimension(self):
        return self._n_features


class MockBinaryFeaturizer:
    """Mock binary fingerprint featurizer."""

    feature_type = 'binary'

    def get_dimension(self):
        return 2048


class MockNoDtypeFeaturizer:
    """Continuous featurizer without fingerprint or dtype attributes."""

    feature_type = 'continuous'

    def __init__(self, n_features=500, name='CustomFeaturizer'):
        self._n_features = n_features
        self.__class__.__name__ = name

    def get_dimension(self):
        return self._n_features


@pytest.mark.unit
class TestCheckLargeFeatureGuard:
    """Unit tests for check_large_feature_guard() function."""

    def test_guard_mordred_15M_raises_configuration_error(self):
        """Mordred-like continuous featurizer on >10M pool raises ConfigurationError."""
        featurizer = MockContinuousFeaturizer(
            n_features=1613, itemsize=4, name='MordredFeaturizer'
        )

        with pytest.raises(ConfigurationError) as exc_info:
            check_large_feature_guard(
                featurizer, pool_size=15_000_000, large_features_ack=False
            )

        message = str(exc_info.value)
        assert 'MordredFeaturizer' in message
        assert 'N=15,000,000' in message
        assert 'GB' in message
        assert 'large_features_ack' in message

    def test_guard_morgan_15M_passes_silently(self):
        """Morgan (binary fingerprint) on any pool size raises no error."""
        featurizer = MockBinaryFeaturizer()
        check_large_feature_guard(
            featurizer, pool_size=15_000_000, large_features_ack=False
        )

    def test_guard_acknowledged_large_features_logs_warning_and_proceeds(self):
        """Acknowledged large feature run issues LearnM8Warning but does not raise."""
        featurizer = MockContinuousFeaturizer(
            n_features=1613, itemsize=4, name='MordredFeaturizer'
        )

        with pytest.warns(LearnM8Warning, match='MordredFeaturizer.*N=15,000,000.*GB'):
            check_large_feature_guard(
                featurizer, pool_size=15_000_000, large_features_ack=True
            )

    def test_guard_small_pool_skips_regardless_of_featurizer(self):
        """Pool size <= 10M skips guard even for continuous featurizers."""
        featurizer = MockContinuousFeaturizer(
            n_features=1613, itemsize=4, name='MordredFeaturizer'
        )

        check_large_feature_guard(
            featurizer, pool_size=10_000_000, large_features_ack=False
        )
        check_large_feature_guard(
            featurizer, pool_size=5_000_000, large_features_ack=False
        )

    def test_guard_custom_featurizer_missing_feature_type_skips(self):
        """Featurizer without feature_type defaults to 'binary' and skips guard."""
        featurizer = MagicMock(spec=[])

        check_large_feature_guard(
            featurizer, pool_size=15_000_000, large_features_ack=False
        )

    def test_guard_custom_featurizer_missing_dtype_falls_back_to_float32(self):
        """Continuous featurizer without fingerprint or dtype defaults itemsize=4."""
        featurizer = MockNoDtypeFeaturizer(n_features=500, name='CustomFeaturizer')

        with pytest.raises(ConfigurationError) as exc_info:
            check_large_feature_guard(
                featurizer, pool_size=15_000_000, large_features_ack=False
            )

        assert 'CustomFeaturizer' in str(exc_info.value)

    def test_guard_sklearn_pipeline_introspects_inner_featurizer(self):
        """sklearn Pipeline wrapping a continuous featurizer triggers the guard."""
        featurizer = MockContinuousFeaturizer(
            n_features=2000, itemsize=4, name='Pipeline'
        )

        with pytest.raises(ConfigurationError) as exc_info:
            check_large_feature_guard(
                featurizer, pool_size=15_000_000, large_features_ack=False
            )

        assert 'Pipeline' in str(exc_info.value)

    def test_guard_custom_featurizer_no_standard_attrs_guard_skips_safely(self):
        """Featurizer with feature_type='continuous' but no dimension probeable attrs skips guard."""

        class BadFeaturizer:
            feature_type = 'continuous'

        check_large_feature_guard(
            BadFeaturizer(), pool_size=15_000_000, large_features_ack=False
        )


@pytest.mark.unit
class TestLargeFeatureGuardIntegration:
    """Integration tests: guard wired into run_active_learning()."""

    def test_guard_called_during_run_active_learning(self, sample_compounds, tmp_path):
        """run_active_learning() calls check_large_feature_guard with correct arguments."""
        from learnm8 import run_active_learning

        oracle_path = tmp_path / 'oracle.csv'
        oracle_data = sample_compounds.clone()
        oracle_data = oracle_data.with_columns(
            pl.Series('Activity', np.random.uniform(0, 1, len(sample_compounds)))
        )
        oracle_data.write_csv(oracle_path)

        with patch('learnm8.features.guard.check_large_feature_guard') as mock_guard:
            mock_guard.side_effect = ConfigurationError('Blocked by guard')
            with pytest.raises(ConfigurationError, match='Blocked by guard'):
                run_active_learning(
                    compound_pool=sample_compounds,
                    oracle=str(oracle_path),
                    learner='rf',
                    target_col='Activity',
                    featurizer='mordred',
                    n_cycles=1,
                    batch_fraction=0.1,
                    output_dir=tmp_path / 'output',
                )
            mock_guard.assert_called_once()

    def test_guard_passes_large_features_ack_flag(self, sample_compounds, tmp_path):
        """run_active_learning() forwards large_features_ack=True to the guard."""
        from learnm8 import run_active_learning

        oracle_path = tmp_path / 'oracle.csv'
        oracle_data = sample_compounds.clone()
        oracle_data = oracle_data.with_columns(
            pl.Series('Activity', np.random.uniform(0, 1, len(sample_compounds)))
        )
        oracle_data.write_csv(oracle_path)

        with patch('learnm8.features.guard.check_large_feature_guard') as mock_guard:
            mock_guard.side_effect = ConfigurationError('Blocked by guard')
            with pytest.raises(ConfigurationError, match='Blocked by guard'):
                run_active_learning(
                    compound_pool=sample_compounds,
                    oracle=str(oracle_path),
                    learner='rf',
                    target_col='Activity',
                    featurizer='mordred',
                    n_cycles=1,
                    batch_fraction=0.1,
                    output_dir=tmp_path / 'output',
                    large_features_ack=True,
                )
            _, _, ack = mock_guard.call_args[0]
            assert ack is True

    def test_guard_blocks_run_when_not_acknowledged(self, sample_compounds, tmp_path):
        """When guard raises, the exception propagates to run_active_learning() caller."""
        from learnm8 import run_active_learning

        oracle_path = tmp_path / 'oracle.csv'
        oracle_data = sample_compounds.clone()
        oracle_data = oracle_data.with_columns(
            pl.Series('Activity', np.random.uniform(0, 1, len(sample_compounds)))
        )
        oracle_data.write_csv(oracle_path)

        with patch('learnm8.features.guard.check_large_feature_guard') as mock_guard:
            mock_guard.side_effect = ConfigurationError('Blocked by guard')
            with pytest.raises(ConfigurationError, match='Blocked by guard'):
                run_active_learning(
                    compound_pool=sample_compounds,
                    oracle=str(oracle_path),
                    learner='rf',
                    target_col='Activity',
                    featurizer='mordred',
                    n_cycles=1,
                    batch_fraction=0.1,
                    output_dir=tmp_path / 'output',
                )

    def test_guard_raises_before_any_rdkit_calls(self, sample_compounds, tmp_path):
        """Guard raising ConfigurationError prevents validate_compound_pool from ever being called."""
        from learnm8 import run_active_learning

        oracle_path = tmp_path / 'oracle.csv'
        oracle_data = sample_compounds.clone()
        oracle_data = oracle_data.with_columns(
            pl.Series('Activity', np.random.uniform(0, 1, len(sample_compounds)))
        )
        oracle_data.write_csv(oracle_path)

        with patch('learnm8.features.guard.check_large_feature_guard') as mock_guard:
            mock_guard.side_effect = ConfigurationError('Blocked by guard')
            with patch('learnm8.api.validate_compound_pool') as mock_validate:
                with pytest.raises(ConfigurationError, match='Blocked by guard'):
                    run_active_learning(
                        compound_pool=sample_compounds,
                        oracle=str(oracle_path),
                        learner='rf',
                        target_col='Activity',
                        featurizer='mordred',
                        n_cycles=1,
                        batch_fraction=0.1,
                        output_dir=tmp_path / 'output',
                    )
                mock_validate.assert_not_called()

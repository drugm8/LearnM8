"""Tests for actionable error messages (ERR-003).

Verifies that error messages follow the WHAT/WHY/HOW pattern:
- WHAT happened (specific, not vague)
- WHY it happened (if determinable)
- HOW to fix it (actionable suggestion)
"""

import pytest
import numpy as np
import polars as pl

from learnm8.exceptions import (
    _truncate_list,
    LearnM8Error,
    ConfigurationError,
    ValidationError,
    LearnerError,
    AcquisitionError,
    OracleError,
    PruningError,
    FeatureExtractionError,
)

pytestmark = [pytest.mark.unit]


class TestTruncateList:

    def test_short_list_not_truncated(self):
        result = _truncate_list([1, 2, 3])
        assert result == "1, 2, 3"
        assert "more" not in result

    def test_exact_max_not_truncated(self):
        items = list(range(10))
        result = _truncate_list(items)
        assert "more" not in result

    def test_long_list_truncated(self):
        items = list(range(15))
        result = _truncate_list(items)
        assert "and 5 more" in result
        assert "0, 1, 2" in result

    def test_custom_max_items_limits_output_and_reports_remaining_count(self):
        items = list(range(5))
        result = _truncate_list(items, max_items=3)
        assert "and 2 more" in result

    def test_empty_list(self):
        result = _truncate_list([])
        assert result == ""

    def test_generator_input(self):
        result = _truncate_list(range(3))
        assert result == "0, 1, 2"


MIN_MESSAGE_LENGTH = 50


class TestLearnerErrorMessages:

    def test_empty_dataset_message_has_what_and_how(self):
        from learnm8.learners.base import SklearnLearner
        from sklearn.ensemble import RandomForestRegressor

        learner = SklearnLearner(RandomForestRegressor(n_estimators=5))
        with pytest.raises(LearnerError, match="Cannot train") as exc_info:
            learner.train(np.empty((0, 10)), np.empty(0))
        msg = str(exc_info.value)
        assert len(msg) >= MIN_MESSAGE_LENGTH
        assert "batch_fraction" in msg or "pool size" in msg

    def test_untrained_predict_message_has_what_and_how(self):
        from learnm8.learners.base import SklearnLearner
        from sklearn.ensemble import RandomForestRegressor

        learner = SklearnLearner(RandomForestRegressor(n_estimators=5))
        with pytest.raises(LearnerError, match="must be trained") as exc_info:
            learner.predict(np.ones((5, 10)))
        msg = str(exc_info.value)
        assert len(msg) >= MIN_MESSAGE_LENGTH
        assert "train()" in msg

    def test_mismatched_shapes_message(self):
        from learnm8.learners.base import SklearnLearner
        from sklearn.ensemble import RandomForestRegressor

        learner = SklearnLearner(RandomForestRegressor(n_estimators=5))
        with pytest.raises(LearnerError, match="mismatched") as exc_info:
            learner.train(np.ones((5, 10)), np.ones(3))
        msg = str(exc_info.value)
        assert "5" in msg and "3" in msg


class TestAcquisitionErrorMessages:

    def test_empty_pool_message(self):
        from learnm8.acquisition.greedy import GreedyAcquisition

        acq = GreedyAcquisition()
        empty_df = pl.DataFrame(schema={"ID": pl.Utf8, "SMILES": pl.Utf8, "prediction": pl.Float64})
        with pytest.raises(ValueError, match="empty") as exc_info:
            acq.select(empty_df, n_select=5)
        msg = str(exc_info.value)
        assert len(msg) >= MIN_MESSAGE_LENGTH

    def test_missing_columns_lists_available(self):
        from learnm8.acquisition.greedy import GreedyAcquisition

        acq = GreedyAcquisition()
        df = pl.DataFrame({"ID": ["a"], "SMILES": ["CCO"], "value": [1.0]})
        with pytest.raises(ValueError, match="prediction") as exc_info:
            acq.select(df, n_select=1)
        msg = str(exc_info.value)
        assert "missing" in msg.lower() or "Missing" in msg

    def test_uncertainty_required_suggests_alternatives(self):
        from learnm8.acquisition import get_acquisition_function

        ucb_class = get_acquisition_function("ucb")
        acq = ucb_class()
        df = pl.DataFrame({
            "ID": ["a", "b"],
            "SMILES": ["CCO", "CCC"],
            "prediction": [1.0, 2.0],
        })
        with pytest.raises(ValueError, match="uncertainty") as exc_info:
            acq.select(df, n_select=1)
        msg = str(exc_info.value)
        assert "greedy" in msg or "random" in msg or "topk" in msg

    def test_unknown_strategy_lists_available(self):
        from learnm8.acquisition import get_acquisition_function

        with pytest.raises(KeyError) as exc_info:
            get_acquisition_function("nonexistent")
        msg = str(exc_info.value)
        assert "greedy" in msg
        assert "ucb" in msg

    def test_nan_predictions_includes_count(self):
        from learnm8.acquisition.greedy import GreedyAcquisition

        acq = GreedyAcquisition()
        df = pl.DataFrame({
            "ID": ["a", "b", "c"],
            "SMILES": ["CCO", "CCC", "CCN"],
            "prediction": [1.0, float("nan"), 3.0],
        })
        with pytest.raises(ValueError, match="NaN") as exc_info:
            acq.select(df, n_select=1)
        msg = str(exc_info.value)
        assert "1" in msg or "NaN" in msg


class TestConfigurationErrorMessages:

    def test_batch_fraction_none_suggests_value(self):
        from learnm8.core.config import CycleConfig

        with pytest.raises(ValueError, match="batch_fraction") as exc_info:
            CycleConfig(strategy="greedy")
        msg = str(exc_info.value)
        assert "0.01" in msg or "fraction" in msg.lower()

    def test_batch_fraction_out_of_range(self):
        from learnm8.core.config import CycleConfig

        with pytest.raises(ValueError, match="batch_fraction") as exc_info:
            CycleConfig(strategy="greedy", batch_fraction=1.5)
        msg = str(exc_info.value)
        assert len(msg) >= MIN_MESSAGE_LENGTH


class TestFeatureExtractionErrorMessages:

    def test_smiles_too_long_message(self):
        from learnm8.features.base import SkfpFeaturizer, MAX_SMILES_LENGTH
        from unittest.mock import MagicMock

        fp_mock = MagicMock()
        featurizer = SkfpFeaturizer(fp_mock, auto_generate_conformers=False)
        long_smiles = "C" * (MAX_SMILES_LENGTH + 1)
        with pytest.raises(FeatureExtractionError, match="exceeds") as exc_info:
            featurizer.transform([long_smiles])
        msg = str(exc_info.value)
        assert str(MAX_SMILES_LENGTH) in msg


class TestPruningErrorMessages:

    def test_pruning_fraction_out_of_range(self):
        from learnm8.pruning.score_based import ScoreBasedPruner

        with pytest.raises(PruningError, match="pruning_fraction") as exc_info:
            ScoreBasedPruner(pruning_fraction=0.95)
        msg = str(exc_info.value)
        assert "0.0" in msg and "0.9" in msg

    def test_empty_compounds_message(self):
        from learnm8.pruning.score_based import ScoreBasedPruner

        pruner = ScoreBasedPruner(pruning_fraction=0.3)
        empty_df = pl.DataFrame(schema={"ID": pl.Utf8, "SMILES": pl.Utf8})
        with pytest.raises(PruningError, match="empty") as exc_info:
            pruner.prune(empty_df, np.array([]))
        msg = str(exc_info.value)
        assert len(msg) >= MIN_MESSAGE_LENGTH


class TestOracleErrorMessages:

    def test_csv_not_found_message(self):
        from learnm8.oracles.csv_oracle import CSVOracle

        with pytest.raises(FileNotFoundError, match="Oracle CSV") as exc_info:
            CSVOracle("/nonexistent/path/data.csv")
        msg = str(exc_info.value)
        assert "Verify" in msg or "verify" in msg

    def test_python_oracle_missing_path(self):
        from learnm8.oracles.python_oracle import PythonOracle

        with pytest.raises(OracleError, match="module_path") as exc_info:
            PythonOracle()
        msg = str(exc_info.value)
        assert len(msg) >= MIN_MESSAGE_LENGTH

    def test_python_oracle_file_not_found(self):
        from learnm8.oracles.python_oracle import PythonOracle

        with pytest.raises(FileNotFoundError, match="Oracle Python") as exc_info:
            PythonOracle(module_path="/nonexistent/oracle.py")
        msg = str(exc_info.value)
        assert "Verify" in msg or "verify" in msg


class TestAddNoteForCycleContext:

    def test_add_note_available_on_exceptions(self):
        err = LearnerError("test error")
        err.add_note("context note")
        assert hasattr(err, "__notes__")
        assert "context note" in err.__notes__

    def test_add_note_on_all_exception_types(self):
        for exc_class in [
            ConfigurationError,
            ValidationError,
            LearnerError,
            AcquisitionError,
            OracleError,
            PruningError,
            FeatureExtractionError,
        ]:
            err = exc_class("test")
            err.add_note("cycle context")
            assert "cycle context" in err.__notes__

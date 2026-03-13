"""Tests for the LearnM8 exception hierarchy (ERR-001)."""

import warnings

import pytest

from learnm8.exceptions import (
    LearnM8Error,
    ConfigurationError,
    ValidationError,
    FeatureExtractionError,
    LearnerError,
    AcquisitionError,
    OracleError,
    PersistenceError,
    PruningError,
    LearnM8Warning,
    ConvergenceWarning,
    DataConversionWarning,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------


class TestLearnM8Error:

    def test_inherits_from_exception(self):
        assert issubclass(LearnM8Error, Exception)

    def test_instantiation_with_message(self):
        exc = LearnM8Error("something went wrong")
        assert str(exc) == "something went wrong"

    def test_catchable_as_exception(self):
        with pytest.raises(Exception):
            raise LearnM8Error("boom")


# ---------------------------------------------------------------------------
# Dual-inheritance error classes
# ---------------------------------------------------------------------------


class TestConfigurationError:

    def test_inherits_learnm8error(self):
        assert issubclass(ConfigurationError, LearnM8Error)

    def test_inherits_valueerror(self):
        assert issubclass(ConfigurationError, ValueError)

    def test_catchable_as_valueerror(self):
        with pytest.raises(ValueError):
            raise ConfigurationError("bad param")

    def test_catchable_as_learnm8error(self):
        with pytest.raises(LearnM8Error):
            raise ConfigurationError("bad param")

    def test_isinstance_checks(self):
        exc = ConfigurationError("x")
        assert isinstance(exc, LearnM8Error)
        assert isinstance(exc, ValueError)
        assert isinstance(exc, Exception)


class TestValidationError:

    def test_inherits_learnm8error(self):
        assert issubclass(ValidationError, LearnM8Error)

    def test_inherits_valueerror(self):
        assert issubclass(ValidationError, ValueError)

    def test_default_attributes_are_none(self):
        exc = ValidationError("bad data")
        assert exc.invalid_indices is None
        assert exc.invalid_smiles is None

    def test_invalid_indices_attribute(self):
        exc = ValidationError("bad rows", invalid_indices=[1, 3, 5])
        assert exc.invalid_indices == [1, 3, 5]
        assert exc.invalid_smiles is None

    def test_invalid_smiles_attribute(self):
        exc = ValidationError(
            "bad smiles",
            invalid_indices=[0, 2],
            invalid_smiles=["XXX", "YYY"],
        )
        assert exc.invalid_indices == [0, 2]
        assert exc.invalid_smiles == ["XXX", "YYY"]

    def test_message_preserved(self):
        exc = ValidationError("test message", invalid_indices=[1])
        assert str(exc) == "test message"

    def test_catchable_as_valueerror(self):
        with pytest.raises(ValueError):
            raise ValidationError("bad", invalid_indices=[0])

    def test_catchable_as_learnm8error(self):
        with pytest.raises(LearnM8Error):
            raise ValidationError("bad")


class TestFeatureExtractionError:

    def test_inherits_learnm8error(self):
        assert issubclass(FeatureExtractionError, LearnM8Error)

    def test_inherits_runtimeerror(self):
        assert issubclass(FeatureExtractionError, RuntimeError)

    def test_catchable_as_runtimeerror(self):
        with pytest.raises(RuntimeError):
            raise FeatureExtractionError("featurizer failed")

    def test_catchable_as_learnm8error(self):
        with pytest.raises(LearnM8Error):
            raise FeatureExtractionError("featurizer failed")


class TestLearnerError:

    def test_inherits_learnm8error(self):
        assert issubclass(LearnerError, LearnM8Error)

    def test_inherits_runtimeerror(self):
        assert issubclass(LearnerError, RuntimeError)

    def test_isinstance_checks(self):
        exc = LearnerError("train failed")
        assert isinstance(exc, LearnM8Error)
        assert isinstance(exc, RuntimeError)


class TestAcquisitionError:

    def test_inherits_learnm8error(self):
        assert issubclass(AcquisitionError, LearnM8Error)

    def test_inherits_runtimeerror(self):
        assert issubclass(AcquisitionError, RuntimeError)

    def test_isinstance_checks(self):
        exc = AcquisitionError("selection failed")
        assert isinstance(exc, LearnM8Error)
        assert isinstance(exc, RuntimeError)


class TestOracleError:

    def test_inherits_learnm8error(self):
        assert issubclass(OracleError, LearnM8Error)

    def test_inherits_runtimeerror(self):
        assert issubclass(OracleError, RuntimeError)

    def test_catchable_as_runtimeerror(self):
        with pytest.raises(RuntimeError):
            raise OracleError("measurement failed")


class TestPersistenceError:

    def test_inherits_learnm8error(self):
        assert issubclass(PersistenceError, LearnM8Error)

    def test_inherits_oserror(self):
        assert issubclass(PersistenceError, OSError)

    def test_catchable_as_oserror(self):
        with pytest.raises(OSError):
            raise PersistenceError("disk full")

    def test_isinstance_checks(self):
        exc = PersistenceError("write failed")
        assert isinstance(exc, LearnM8Error)
        assert isinstance(exc, OSError)


class TestPruningError:

    def test_inherits_learnm8error(self):
        assert issubclass(PruningError, LearnM8Error)

    def test_inherits_runtimeerror(self):
        assert issubclass(PruningError, RuntimeError)

    def test_isinstance_checks(self):
        exc = PruningError("pruning failed")
        assert isinstance(exc, LearnM8Error)
        assert isinstance(exc, RuntimeError)


# ---------------------------------------------------------------------------
# Catch-all: every error subclass caught by except LearnM8Error
# ---------------------------------------------------------------------------


class TestCatchAll:

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigurationError,
            ValidationError,
            FeatureExtractionError,
            LearnerError,
            AcquisitionError,
            OracleError,
            PersistenceError,
            PruningError,
        ],
    )
    def test_all_caught_by_learnm8error(self, exc_class):
        with pytest.raises(LearnM8Error):
            if exc_class is ValidationError:
                raise exc_class("err", invalid_indices=[])
            else:
                raise exc_class("err")


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class TestLearnM8Warning:

    def test_inherits_userwarning(self):
        assert issubclass(LearnM8Warning, UserWarning)

    def test_captured_by_warnings_module(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("test", LearnM8Warning)
            assert len(w) == 1
            assert issubclass(w[0].category, LearnM8Warning)


class TestConvergenceWarning:

    def test_inherits_learnm8warning(self):
        assert issubclass(ConvergenceWarning, LearnM8Warning)

    def test_inherits_userwarning(self):
        assert issubclass(ConvergenceWarning, UserWarning)

    def test_captured_by_learnm8warning_filter(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", LearnM8Warning)
            warnings.warn("not converged", ConvergenceWarning)
            assert len(w) == 1
            assert issubclass(w[0].category, ConvergenceWarning)


class TestDataConversionWarning:

    def test_inherits_learnm8warning(self):
        assert issubclass(DataConversionWarning, LearnM8Warning)

    def test_inherits_userwarning(self):
        assert issubclass(DataConversionWarning, UserWarning)


# ---------------------------------------------------------------------------
# Import paths (backward compatibility)
# ---------------------------------------------------------------------------


class TestImportPaths:

    def test_import_from_top_level(self):
        from learnm8 import (  # noqa: F401
            LearnM8Error,
            ConfigurationError,
            ValidationError,
            FeatureExtractionError,
            LearnerError,
            AcquisitionError,
            OracleError,
            PersistenceError,
            PruningError,
            LearnM8Warning,
            ConvergenceWarning,
            DataConversionWarning,
        )

    def test_import_from_exceptions_module(self):
        from learnm8.exceptions import LearnM8Error as E  # noqa: F401
        from learnm8.exceptions import PruningError as P  # noqa: F401

    def test_backward_compat_acquisition_error(self):
        from learnm8.acquisition import AcquisitionError as AE
        from learnm8.acquisition.base import AcquisitionError as AE2
        assert AE is AE2

    def test_backward_compat_pruning_error(self):
        from learnm8.pruning import PruningError as PE
        from learnm8.pruning.base import PruningError as PE2
        assert PE is PE2

    def test_all_exceptions_same_identity(self):
        from learnm8 import AcquisitionError as A1
        from learnm8.exceptions import AcquisitionError as A2
        from learnm8.acquisition import AcquisitionError as A3
        assert A1 is A2 is A3

    def test_all_pruning_errors_same_identity(self):
        from learnm8 import PruningError as P1
        from learnm8.exceptions import PruningError as P2
        from learnm8.pruning import PruningError as P3
        assert P1 is P2 is P3

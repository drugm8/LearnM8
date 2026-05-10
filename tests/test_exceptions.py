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
)

pytestmark = [pytest.mark.unit]

EXCEPTION_SPECS = [
    (LearnM8Error, None),
    (ConfigurationError, None),
    (ValidationError, ("err",)),
    (FeatureExtractionError, None),
    (LearnerError, None),
    (AcquisitionError, None),
    (OracleError, None),
    (PersistenceError, None),
    (PruningError, None),
]


class TestExceptionHierarchy:

    @pytest.mark.parametrize("exc_class,args", EXCEPTION_SPECS)
    def test_inheritance(self, exc_class, args):
        assert issubclass(exc_class, LearnM8Error)

    @pytest.mark.parametrize("exc_class,args", EXCEPTION_SPECS)
    def test_isinstance(self, exc_class, args):
        exc = exc_class(*(args or ("test",)))
        assert isinstance(exc, LearnM8Error)
        assert isinstance(exc, Exception)

    @pytest.mark.parametrize("exc_class,args", EXCEPTION_SPECS)
    def test_catchable_as_learnm8error(self, exc_class, args):
        with pytest.raises(LearnM8Error):
            raise exc_class(*(args or ("test",)))

    def test_learnm8error_instantiation(self):
        assert str(LearnM8Error("msg")) == "msg"


class TestValidationErrorAttributes:

    def test_default_attributes_are_none(self):
        exc = ValidationError("bad data")
        assert exc.invalid_indices is None
        assert exc.invalid_smiles is None

    def test_invalid_indices_attribute(self):
        exc = ValidationError("bad rows", invalid_indices=[1, 3, 5])
        assert exc.invalid_indices == [1, 3, 5]
        assert exc.invalid_smiles is None

    def test_invalid_smiles_attribute(self):
        exc = ValidationError("bad smiles", invalid_indices=[0, 2],
                              invalid_smiles=["XXX", "YYY"])
        assert exc.invalid_indices == [0, 2]
        assert exc.invalid_smiles == ["XXX", "YYY"]

    def test_message_preserved(self):
        assert str(ValidationError("test message", invalid_indices=[1])) == "test message"


WARNING_SPECS = [
    (LearnM8Warning, UserWarning),
    (ConvergenceWarning, LearnM8Warning),
]


class TestWarningHierarchy:

    @pytest.mark.parametrize("warn_class,parent", WARNING_SPECS)
    def test_inheritance(self, warn_class, parent):
        assert issubclass(warn_class, parent)

    def test_captured_by_warnings_module(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("test", LearnM8Warning)
            assert len(w) == 1
            assert issubclass(w[0].category, LearnM8Warning)

    def test_captured_by_learnm8warning_filter(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", LearnM8Warning)
            warnings.warn("not converged", ConvergenceWarning)
            assert len(w) == 1
            assert issubclass(w[0].category, ConvergenceWarning)


class TestImportPaths:

    def test_import_from_top_level(self):
        from learnm8 import (  # noqa: F401
            LearnM8Error, ConfigurationError, ValidationError,
            FeatureExtractionError, LearnerError, AcquisitionError,
            OracleError, PersistenceError, PruningError,
            LearnM8Warning, ConvergenceWarning,
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

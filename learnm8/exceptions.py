"""Exception hierarchy for the LearnM8 active learning framework.

All library-specific exceptions inherit from :class:`LearnM8Error`, enabling
users to catch every LearnM8 error with a single ``except LearnM8Error`` clause
while retaining the ability to catch specific subtypes or their built-in parents
(e.g. ``except ValueError`` will also catch :class:`ConfigurationError`).

All library-specific warnings inherit from :class:`LearnM8Warning` and integrate
with Python's standard :mod:`warnings` filter system.
"""

from __future__ import annotations


class LearnM8Error(Exception):
    """Base exception for all LearnM8 library errors.

    Catch this to handle any error raised by LearnM8.

    Example:
        >>> try:
        ...     run_active_learning(...)
        ... except LearnM8Error as e:
        ...     print(f"LearnM8 failed: {e}")
    """


class ConfigurationError(LearnM8Error, ValueError):
    """Invalid parameters, incompatible settings, or unknown component names.

    Raised when user-supplied configuration is invalid, such as unknown learner
    names, incompatible acquisition/learner combinations, or out-of-range
    parameter values.
    """


class ValidationError(LearnM8Error, ValueError):
    """Invalid input data with optional structured error details.

    Carries information about which compounds failed validation so callers can
    inspect and fix specific entries.

    Attributes:
        invalid_indices: Indices of compounds that failed validation.
        invalid_smiles: SMILES strings that failed validation.

    Example:
        >>> try:
        ...     validate_compound_pool(df)
        ... except ValidationError as e:
        ...     print(f"Bad rows: {e.invalid_indices}")
    """

    def __init__(
        self,
        message: str,
        *,
        invalid_indices: list[int] | None = None,
        invalid_smiles: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.invalid_indices = invalid_indices
        self.invalid_smiles = invalid_smiles


class FeatureExtractionError(LearnM8Error, RuntimeError):
    """Featurizer failures during molecular feature extraction.

    Raised when feature extraction fails for reasons such as invalid molecular
    structures, unsupported featurizer configurations, or numerical errors
    during fingerprint computation.
    """


class LearnerError(LearnM8Error, RuntimeError):
    """Model training or prediction failures.

    Raised when a learner fails to train on provided data or when prediction
    fails due to model state, input shape mismatches, or numerical instability.
    """


class AcquisitionError(LearnM8Error, RuntimeError):
    """Acquisition strategy failures.

    Raised when an acquisition function encounters errors during compound
    selection, such as missing required columns, invalid score values, or
    numerical issues in acquisition score computation.
    """


class OracleError(LearnM8Error, RuntimeError):
    """Oracle measurement failures.

    Raised when an oracle fails to provide measurements for selected compounds,
    such as external service failures, missing properties, or computation errors
    in user-defined oracle functions.
    """


class PersistenceError(LearnM8Error, OSError):
    """File I/O and caching failures.

    Raised when saving or loading results, HDF5 cache operations, or other
    file system interactions fail.
    """


class PruningError(LearnM8Error, RuntimeError):
    """Design space pruning failures.

    Raised when pruning encounters invalid parameters, incompatible inputs,
    or computational errors during compound filtering.
    """


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class LearnM8Warning(UserWarning):
    """Base warning for all LearnM8 library warnings.

    Filter with standard :mod:`warnings` mechanisms::

        import warnings
        warnings.filterwarnings("ignore", category=LearnM8Warning)
    """


class ConvergenceWarning(LearnM8Warning):
    """Model convergence issues during training.

    Issued when a learner's training loop does not converge within the
    configured number of epochs or iterations.
    """


class DataConversionWarning(LearnM8Warning):
    """Auto-conversion notices for input data.

    Issued when LearnM8 automatically converts input data types, such as
    converting a pandas DataFrame to Polars or casting column types.
    """

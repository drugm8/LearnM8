"""Exception hierarchy for the LearnM8 active learning framework.

All library-specific exceptions inherit from :class:`LearnM8Error`, enabling
users to catch every LearnM8 error with a single ``except LearnM8Error`` clause
while retaining the ability to catch specific subtypes.

All library-specific warnings inherit from :class:`LearnM8Warning` and integrate
with Python's standard :mod:`warnings` filter system.
"""

from __future__ import annotations

MAX_LIST_ITEMS = 10


def _truncate_list(items, max_items: int = MAX_LIST_ITEMS) -> str:
    items = list(items)
    if len(items) <= max_items:
        return ", ".join(str(i) for i in items)
    shown = ", ".join(str(i) for i in items[:max_items])
    return f"{shown} (and {len(items) - max_items} more)"


class LearnM8Error(Exception):
    """Base exception for all LearnM8 library errors.

    Catch this to handle any error raised by LearnM8.
    """


class ConfigurationError(LearnM8Error):
    """Invalid parameters, incompatible settings, or unknown component names.

    Raised when user-supplied configuration is invalid, such as unknown learner
    names, incompatible acquisition/learner combinations, or out-of-range
    parameter values.
    """


class ValidationError(LearnM8Error):
    """Invalid input data with optional structured error details.

    Carries information about which compounds failed validation so callers can
    inspect and fix specific entries.

    Attributes:
        invalid_indices: Indices of compounds that failed validation.
        invalid_smiles: SMILES strings that failed validation.
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


class FeatureExtractionError(LearnM8Error):
    """Featurizer failures during molecular feature extraction.

    Raised when feature extraction fails for reasons such as invalid molecular
    structures, unsupported featurizer configurations, or numerical errors
    during fingerprint computation.
    """


class LearnerError(LearnM8Error):
    """Model training or prediction failures.

    Raised when a learner fails to train on provided data or when prediction
    fails due to model state, input shape mismatches, or numerical instability.
    """


class AcquisitionError(LearnM8Error):
    """Acquisition strategy failures.

    Raised when an acquisition function encounters errors during compound
    selection, such as missing required columns, invalid score values, or
    numerical issues in acquisition score computation.
    """


class OracleError(LearnM8Error):
    """Oracle measurement failures.

    Raised when an oracle fails to provide measurements for selected compounds,
    such as external service failures, missing properties, or computation errors
    in user-defined oracle functions.
    """


class PersistenceError(LearnM8Error):
    """File I/O and caching failures.

    Raised when saving or loading results, HDF5 cache operations, or other
    file system interactions fail.
    """


class PruningError(LearnM8Error):
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

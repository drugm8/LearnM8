"""Base class for scikit-fingerprints featurizer wrappers.

Provides common functionality for all scikit-fingerprints fingerprint
wrappers. Handles SMILES → Mol conversion, optional conformer generation
for 3D fingerprints, and parameter management.
"""

import logging
import warnings
from typing import Any

import numpy as np
from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer

from learnm8.core.interfaces import Featurizer
from learnm8.exceptions import FeatureExtractionError, LearnM8Warning

logger = logging.getLogger(__name__)

MAX_SMILES_LENGTH = 10000

# RDKit ETKDG convention seed (0xf00d == 61453). Used as the default
# random_state for ConformerGenerator so 3D fingerprints are deterministic
# across runs. Recorded in get_config() when requires_3d() is True so cache
# keys disambiguate different seeds.
DEFAULT_3D_RANDOM_STATE: int = 0xF00D

# Cache-irrelevant featurizer parameters (feature 025, Item 1). These are
# surfaced by scikit-fingerprints' ``get_params()`` but never change feature
# VALUES: ``n_jobs`` is a parallelism knob, ``verbose`` a logging knob, and
# ``batch_size`` a per-call working-set knob (feature 025, Item 6). They are
# denylisted (excluded) from the config hash so the cache is not fragmented by
# core count, verbosity, or chunk size. The filter is a denylist: any
# ``get_params()`` key NOT in this set is hashed by default — a fail-safe
# direction so an unknown / future feature-determining parameter still changes
# the hash rather than being silently ignored.
_CACHE_IRRELEVANT_PARAMS: frozenset[str] = frozenset(
    {'n_jobs', 'verbose', 'batch_size'}
)

# scikit-fingerprints classes whose ``_calculate_fingerprint`` feeds the
# SMILES *string* into the feature computation (feature 026, REQ-3). For these
# — and only these — the parent process must keep converting SMILES to ``Mol``
# first, because skfp's ``ensure_smiles`` canonicalises ``Mol`` input via
# ``MolToSmiles``. Handing them raw SMILES would silently change feature values
# for any non-canonical input:
#   - LingoFingerprint    slices substrings straight out of the SMILES text
#   - MHFPFingerprint     passes it to MHFPEncoder.EncodeSmilesBulk
#   - SECFPFingerprint    passes it to MHFPEncoder.EncodeSECFPSmiles
#
# This is a DENYLIST: anything absent takes the fast SMILES-through path. Audit
# it when upgrading scikit-fingerprints.
#
# Audit note — grepping skfp for ``ensure_smiles`` returns a fourth class,
# ``RDKit2DDescriptorsFingerprint``. It is deliberately NOT listed: it computes
# ``ensure_smiles(X)`` but descriptastorus' ``calculateMol(m, smiles)`` ignores
# the ``smiles`` argument outright (both RDKit2D and RDKit2DNormalized), so the
# canonicalisation cannot reach the output. The criterion is "the ensure_smiles
# RESULT affects feature values", not "ensure_smiles is called".
_SMILES_CONSUMING_FINGERPRINTS: frozenset[str] = frozenset(
    {
        'LingoFingerprint',
        'MHFPFingerprint',
        'SECFPFingerprint',
    }
)

# Row count at or above which a SMILES-consuming featurizer (REQ-3) warns about
# its unbounded parent-heap Mol allocation (feature 026, REQ-4). ~11 KB per
# RDKit Mol puts 1M rows at roughly 11 GB held in the calling process.
SMILES_CONSUMING_WARN_ROWS: int = 1_000_000

# One-shot guard for the orphaned-cache WARNING (feature 025, REQ-4).
_orphan_cache_warning_emitted: bool = False

# One-shot guard for the SMILES-consuming large-input WARNING (feature 026, REQ-4).
_smiles_consuming_warning_emitted: bool = False


def _warn_orphaned_cache_once() -> None:
    """Emit a single WARNING that the Item-1 hash-recipe change orphans old cache.

    Feature 025 Item 1 widens the config-hash denylist (``verbose`` and
    ``batch_size`` join ``n_jobs``), which changes ``get_config_hash()`` for
    every featurizer. Cache rows written by earlier LearnM8 versions therefore
    become unreachable. This fires once per process from :meth:`get_config` —
    the first hash computation after the upgrade — as a heads-up; the orphaned
    rows are left in place and may be deleted manually.
    """
    global _orphan_cache_warning_emitted
    if _orphan_cache_warning_emitted:
        return
    _orphan_cache_warning_emitted = True
    logger.warning(
        'Featurization cache config-hash recipe changed in this version '
        '(LearnM8 feature 025 / Item 1: n_jobs, verbose, and batch_size are '
        'now excluded from the hash). If you have a feature cache from an '
        'earlier LearnM8 version, its rows for this featurizer are now '
        'orphaned — they will not be reused and new rows are appended '
        'alongside them. Delete stale features_*.h5 files in your cache '
        'directory to reclaim space.'
    )


def _reset_orphan_cache_warning() -> None:
    """Reset the one-shot orphaned-cache WARNING guard (test convenience)."""
    global _orphan_cache_warning_emitted
    _orphan_cache_warning_emitted = False


def _warn_smiles_consuming_once(featurizer_name: str, n_smiles: int) -> None:
    """Warn once that a SMILES-consuming featurizer holds Mols in the parent.

    Feature 026 moves RDKit parsing into the loky workers for every featurizer
    that can accept SMILES directly, removing the parent-heap ``Mol`` list. The
    three fingerprints in :data:`_SMILES_CONSUMING_FINGERPRINTS` cannot make
    that move without changing their feature values (REQ-3), so at large row
    counts they still allocate the full ``Mol`` list in the calling process —
    roughly 11 KB per molecule. This flags that cost rather than leaving it to
    be rediscovered as an OOM on a cluster node.

    Fires at most once per process, following the
    :func:`_warn_orphaned_cache_once` precedent.

    Args:
        featurizer_name: Registry name of the featurizer, for the message.
        n_smiles: Number of SMILES in the triggering call.
    """
    global _smiles_consuming_warning_emitted
    if _smiles_consuming_warning_emitted:
        return
    _smiles_consuming_warning_emitted = True
    logger.warning(
        f'Featurizer {featurizer_name!r} consumes SMILES strings directly, so '
        f'its RDKit Mol objects must be built in this process rather than in '
        f'the featurization workers. This call converts {n_smiles:,} SMILES '
        f'(~{n_smiles * 11 / 1_000_000:.1f} GB of Mol objects at ~11 KB each) '
        f'and that allocation grows without bound with the input size. Prefer '
        f"a featurizer that parses worker-side (e.g. 'morgan', 'ecfp') for "
        f'large pools, or reduce the chunk size.'
    )


def _reset_smiles_consuming_warning() -> None:
    """Reset the one-shot SMILES-consuming WARNING guard (test convenience)."""
    global _smiles_consuming_warning_emitted
    _smiles_consuming_warning_emitted = False


class SkfpFeaturizer(Featurizer):
    """Base class for scikit-fingerprints wrappers.

    Provides common functionality for all scikit-fingerprints fingerprint
    wrappers. Handles SMILES → Mol conversion, optional conformer generation
    for 3D fingerprints, and parameter management.

    Key Features:
    - Automatic conformer generation for 3D fingerprints
    - Configuration-aware caching via get_config()
    - Sklearn-compatible API (fit/transform pattern)
    - Built-in parallelization via n_jobs
    - DoS protection (batch size and SMILES length limits)

    Example:
        >>> from skfp.fingerprints import ECFPFingerprint
        >>> fp = ECFPFingerprint(radius=3, fp_size=4096)
        >>> featurizer = SkfpFeaturizer(fp, auto_generate_conformers=False)
        >>> features = featurizer.transform(['CCO', 'CCC'])
    """

    def __init__(
        self,
        fingerprint_instance,
        auto_generate_conformers: bool = True,
        conformer_params: dict[str, Any] | None = None,
        n_jobs: int = -1,
        verbose: int = 0,
        random_state: int = DEFAULT_3D_RANDOM_STATE,
        *,
        storage_dtype: str,
        feature_type: str,
        fingerprint_name: str,
        fingerprint_params: dict[str, Any],
        description: str | None = None,
    ):
        """Initialize scikit-fingerprints wrapper.

        Args:
            fingerprint_instance: Initialized scikit-fingerprints fingerprint
                                object (e.g., ECFPFingerprint(radius=3))
            auto_generate_conformers: Auto-generate conformers for 3D
                                      fingerprints (default: True)
            conformer_params: Optional dict of parameters for ConformerGenerator
                            (e.g., {'num_conformers': 1, 'optimize_force_field': 'UFF'})
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Verbosity level for scikit-fingerprints logging (default: 0)
            random_state: Seed forwarded to ConformerGenerator for deterministic
                         3D conformer embedding (default ``0xf00d`` == 61453,
                         the RDKit ETKDG convention). Only consumed by 3D
                         fingerprints; recorded in :meth:`get_config` when
                         :meth:`requires_3d` is True so cache keys disambiguate
                         different seeds.
            storage_dtype: Storage dtype for feature cache ('packed_uint8',
                          'float32', 'csr_uint16', etc.)
            feature_type: Feature type identifier ('binary' or 'continuous')
            fingerprint_name: Canonical name for this fingerprint (e.g., 'morgan')
            fingerprint_params: Parameters used to construct the fingerprint
                               instance, for cache key generation
            description: Human-readable description for CLI display

        Note:
            If fingerprint.requires_conformers=True and
            auto_generate_conformers=False, users must provide molecules
            with pre-computed conformers.
        """
        self.fingerprint = fingerprint_instance
        self.auto_generate_conformers = auto_generate_conformers
        self.conformer_params = conformer_params or {}
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = int(random_state)
        self._storage_dtype = storage_dtype
        self._feature_type = feature_type
        self._fingerprint_name = fingerprint_name
        self._fingerprint_params = fingerprint_params
        self._description = description

        # REQ-5: the paths that still materialise Mol objects in this process
        # (3D conformer generation, and the SMILES-consuming fingerprints of
        # REQ-3) parse across n_jobs cores instead of one. Everything else
        # bypasses this transformer entirely and parses worker-side.
        self.mol_from_smiles = MolFromSmilesTransformer(n_jobs=n_jobs)

        if self.requires_3d() and auto_generate_conformers:
            try:
                self.conformer_gen = ConformerGenerator(
                    n_jobs=n_jobs,
                    random_state=self.random_state,
                    **self.conformer_params,
                )
            except TypeError as e:
                if 'random_state' not in str(e):
                    raise
                fallback_msg = (
                    'scikit-fingerprints<1.18.0 detected; ConformerGenerator '
                    'does not accept random_state. 3D fingerprints will be '
                    'NON-DETERMINISTIC and feature-cache reproducibility is '
                    'forfeited. Upgrade with: pip install -U scikit-fingerprints'
                )
                warnings.warn(fallback_msg, LearnM8Warning, stacklevel=2)
                # Also emit a structured log warning so the determinism loss is
                # visible in log pipelines that suppress Python warnings.
                logger.warning(fallback_msg)
                self.random_state = None
                self.conformer_gen = ConformerGenerator(
                    n_jobs=n_jobs, **self.conformer_params
                )
        else:
            self.conformer_gen = None

    def transform(self, smiles_list: list[str]) -> np.ndarray:
        """Transform SMILES to features using scikit-fingerprints.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Feature matrix of shape (n_compounds, n_features)

        Raises:
            ValueError: If SMILES length exceeds limits
            RuntimeError: If featurization fails

        Note:
            For 3D fingerprints, conformers are auto-generated if
            auto_generate_conformers=True (default).

            Dispatch is three-way (feature 026):

            1. ``requires_3d()`` — SMILES are converted to ``Mol`` here, given
               conformers, then fingerprinted (REQ-2).
            2. A fingerprint in :data:`_SMILES_CONSUMING_FINGERPRINTS` — the
               ``Mol`` round-trip is kept because those fingerprints read the
               canonicalised SMILES text itself (REQ-3).
            3. Everything else — SMILES go straight to the fingerprint, which
               parses them inside its own workers. No ``Mol`` list is
               materialised in this process (REQ-1).
        """

        for smiles in smiles_list:
            if len(smiles) > MAX_SMILES_LENGTH:
                raise FeatureExtractionError(
                    f'SMILES string length {len(smiles)} exceeds maximum allowed '
                    f'length of {MAX_SMILES_LENGTH} characters. '
                    f'This limit exists to prevent excessive memory usage. '
                    f'Check your input data for unusually long SMILES strings.'
                )

        if len(smiles_list) == 0:
            return np.empty((0, self.get_dimension()), dtype=np.float32)

        try:
            if self.requires_3d():
                # REQ-2: unchanged 3D path — SMILES -> Mol -> conformers -> fp.
                # The Mol conversion stays ahead of the conformer_gen guard so
                # the failure mode for invalid SMILES is exactly as before.
                mols = self.mol_from_smiles.transform(smiles_list)

                if self.conformer_gen is None:
                    raise FeatureExtractionError(
                        f'{self.get_name()} requires 3D conformers but conformer generation '
                        f'is disabled (auto_generate_conformers=False). '
                        f'Set auto_generate_conformers=True to enable automatic conformer '
                        f'generation, or provide molecules with pre-computed conformers. '
                        f"Alternatively, use a 2D featurizer (e.g., 'morgan', 'ecfp') "
                        f'that does not require conformers.'
                    )

                logger.debug(
                    f'Generating conformers for {len(smiles_list)} molecules '
                    f'(3D fingerprint: {self.get_name()})'
                )
                mols = self.conformer_gen.transform(mols)
                features = self.fingerprint.transform(mols)

            elif self.fingerprint.__class__.__name__ in _SMILES_CONSUMING_FINGERPRINTS:
                # REQ-3: these read the SMILES text itself, and skfp's
                # ensure_smiles canonicalises Mol input via MolToSmiles. Passing
                # raw SMILES would change their feature values for any
                # non-canonical input, so the round-trip is load-bearing here.
                if len(smiles_list) >= SMILES_CONSUMING_WARN_ROWS:
                    _warn_smiles_consuming_once(self.get_name(), len(smiles_list))
                mols = self.mol_from_smiles.transform(smiles_list)
                features = self.fingerprint.transform(mols)

            else:
                # REQ-1: hand SMILES straight through. Every 2D skfp fingerprint
                # calls ensure_mols(X) at the top of _calculate_fingerprint, so
                # the parse still happens — but inside each loky worker, on its
                # own batch_size slice, instead of as one list in this process.
                # At the chunk sizes estimate_batch_size picks (B ~ 16.3M) that
                # list alone was 167-223 GB of parent heap.
                features = self.fingerprint.transform(smiles_list)

        except FeatureExtractionError:
            raise
        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f'Feature extraction failed: {e}')
            raise FeatureExtractionError(
                f'Feature extraction failed for {self.get_name()} on '
                f'{len(smiles_list)} compounds: {e}. '
                f"Check SMILES validity with 'learnm8 validate your_file.csv'. "
                f'If using a 3D featurizer, check conformer generation settings.'
            ) from None

        if features is None or len(features) == 0:
            raise FeatureExtractionError(
                f'No valid features generated by {self.get_name()} for '
                f'{len(smiles_list)} input compounds. All SMILES may be invalid. '
                f"Run 'learnm8 validate your_file.csv' to check SMILES validity."
            )

        raw = np.asarray(features)

        # Detect corruption on the RAW featurizer output, before narrowing the
        # dtype — a NaN/Inf would be silently lost in an integer downcast.
        # Integer raw output (binary fingerprints) is finite by construction.
        if np.issubdtype(raw.dtype, np.floating):
            nonfinite_rows = np.where(~np.isfinite(raw).all(axis=1))[0]
            if nonfinite_rows.size > 0:
                example_idx = int(nonfinite_rows[0])
                if self._feature_type == 'binary':
                    # Binary fingerprints are 0/1 by construction; NaN/Inf here
                    # means genuine corruption. Fail fast before the cache.
                    raise FeatureExtractionError(
                        f'Non-finite values (NaN/Inf) in binary fingerprint '
                        f'{self.get_name()} for {nonfinite_rows.size} of '
                        f'{len(smiles_list)} compounds — first offender: '
                        f"'{smiles_list[example_idx]}'. Caching aborted to avoid "
                        f'poisoning the cache with invalid rows. '
                        f"Run 'learnm8 validate your_file.csv' to check SMILES validity."
                    )
                # Continuous/descriptor featurizers (e.g. mordred) legitimately
                # emit NaN for descriptors that cannot be computed; the
                # learner-side median imputer handles these. Warn, don't abort.
                logger.warning(
                    f'Non-finite values in {self.get_name()} features for '
                    f'{nonfinite_rows.size} of {len(smiles_list)} compounds '
                    f'(continuous featurizer; imputed downstream)'
                )

        # Narrow to the compute dtype (uint8 for binary fingerprints).
        compute_dtype = np.dtype(self.get_compute_dtype())
        features_array = raw.astype(compute_dtype, copy=False)

        return features_array

    def get_dimension(self) -> int:
        """Get feature dimension from scikit-fingerprints fingerprint.

        Returns:
            Integer dimension of feature vectors

        Note:
            Queries the fingerprint object for dimension via:
            - n_features_out (property for sklearn-compatible fingerprints)
            - n_features_out_ (attribute after transform)
            - fp_size (hashed fingerprints)
            - n_features (descriptor-based fingerprints)
        """
        if hasattr(self.fingerprint, 'n_features_out'):
            return self.fingerprint.n_features_out
        elif hasattr(self.fingerprint, 'n_features_out_'):
            return self.fingerprint.n_features_out_
        elif hasattr(self.fingerprint, 'fp_size'):
            return self.fingerprint.fp_size
        elif hasattr(self.fingerprint, 'n_features'):
            return self.fingerprint.n_features
        else:
            params = self.fingerprint.get_params()
            if 'fp_size' in params:
                return params['fp_size']
            elif 'n_features' in params:
                return params['n_features']
            else:
                raise AttributeError(
                    f'Cannot determine feature dimension for {self.get_name()}. '
                    f'The fingerprint object lacks any of the expected attributes: '
                    f'n_features_out, n_features_out_, fp_size, or n_features. '
                    f'Ensure the fingerprint class is a valid scikit-fingerprints object.'
                )

    @property
    def feature_type(self) -> str:
        return self._feature_type

    def get_storage_dtype(self) -> str:
        return self._storage_dtype

    def get_compute_dtype(self) -> str:
        """Return the in-memory dtype for the freshly-computed feature matrix.

        Binary fingerprints are 0/1 and lossless as ``uint8`` — holding them
        as ``uint8`` rather than ``float32`` is a 4x RAM saving on the cold
        featurization path (e.g. a 2048-bit Morgan matrix). Count and
        descriptor featurizers stay ``float32`` so the cache's write-time
        range validation runs on the true values, not a truncated cast.
        """
        return 'uint8' if self._feature_type == 'binary' else 'float32'

    def get_description(self) -> str:
        return self._description if self._description else self.get_name()

    def get_name(self) -> str:
        return self._fingerprint_name

    def requires_3d(self) -> bool:
        """Check if fingerprint requires 3D conformers.

        Returns:
            Boolean from fingerprint's requires_conformers attribute

        Note:
            Returns True for 3D fingerprints (USR, WHIM, E3FP, GETAWAY,
            MORSE, RDF, Autocorr, ElectroShape, USRCAT).
        """
        return getattr(self.fingerprint, 'requires_conformers', False)

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary for cache key generation.

        Returns:
            Dictionary with all configuration parameters

        Note:
            Includes fingerprint class, auto_generate_conformers flag,
            conformer generation parameters, and the fingerprint-specific
            parameters from ``get_params()``. The :data:`_CACHE_IRRELEVANT_PARAMS`
            denylist (``n_jobs``, ``verbose``, ``batch_size``) is filtered out
            of the ``get_params()`` sub-dict only — these knobs control
            parallelism, logging, and per-call chunk size, never feature
            values, so including them would needlessly fragment the cache. The
            manually-assembled keys (``random_state``, ``conformer_params``,
            ``auto_generate_conformers``) are feature-determining and are NOT
            filtered. Any unknown / future ``get_params()`` key is kept by
            default (fail-safe: a new feature-determining param changes the
            hash).
        """
        config: dict[str, Any] = {
            'fingerprint_class': self.fingerprint.__class__.__name__,
            'auto_generate_conformers': self.auto_generate_conformers,
        }

        if self.requires_3d() and self.random_state is not None:
            config['random_state'] = self.random_state

        if self.conformer_params:
            config['conformer_params'] = self.conformer_params

        params = self.fingerprint.get_params()
        # REQ-1: explicit denylist filter applied ONLY to the get_params()
        # sub-dict. Drops n_jobs / verbose / batch_size (parallelism, logging,
        # and per-call chunk-size knobs that never change feature values);
        # every other key — including unknown ones — is kept.
        config.update(
            {k: v for k, v in params.items() if k not in _CACHE_IRRELEVANT_PARAMS}
        )

        _warn_orphaned_cache_once()

        return config

    def validate_smiles(self, smiles_list: list[str]) -> list[bool]:
        """Validate SMILES using RDKit.

        Args:
            smiles_list: List of SMILES strings to validate

        Returns:
            List of booleans indicating validity of each SMILES
        """
        from rdkit import Chem

        return [Chem.MolFromSmiles(s) is not None for s in smiles_list]

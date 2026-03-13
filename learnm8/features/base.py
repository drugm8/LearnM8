"""Base class for scikit-fingerprints featurizer wrappers.

Provides common functionality for all scikit-fingerprints fingerprint
wrappers. Handles SMILES → Mol conversion, optional conformer generation
for 3D fingerprints, and parameter management.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator
from learnm8.core.interfaces import Featurizer
from learnm8.exceptions import FeatureExtractionError

logger = logging.getLogger(__name__)

MAX_SMILES_LENGTH = 10000


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
        conformer_params: Optional[Dict[str, Any]] = None,
        n_jobs: int = -1
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

        Note:
            If fingerprint.requires_conformers=True and
            auto_generate_conformers=False, users must provide molecules
            with pre-computed conformers.
        """
        self.fingerprint = fingerprint_instance
        self.auto_generate_conformers = auto_generate_conformers
        self.conformer_params = conformer_params or {}
        self.n_jobs = n_jobs

        self.mol_from_smiles = MolFromSmilesTransformer()

        if self.requires_3d() and auto_generate_conformers:
            self.conformer_gen = ConformerGenerator(
                n_jobs=n_jobs,
                **self.conformer_params
            )
        else:
            self.conformer_gen = None

    def transform(self, smiles_list: List[str]) -> np.ndarray:
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
        """

        for smiles in smiles_list:
            if len(smiles) > MAX_SMILES_LENGTH:
                raise FeatureExtractionError(
                    f"SMILES string length {len(smiles)} exceeds maximum allowed "
                    f"length of {MAX_SMILES_LENGTH} characters. "
                    f"This limit exists to prevent excessive memory usage. "
                    f"Check your input data for unusually long SMILES strings."
                )

        if len(smiles_list) == 0:
            return np.empty((0, self.get_dimension()), dtype=np.float32)

        try:
            mols = self.mol_from_smiles.transform(smiles_list)

            if self.requires_3d():
                if self.conformer_gen is None:
                    raise FeatureExtractionError(
                        f"{self.get_name()} requires 3D conformers but conformer generation "
                        f"is disabled (auto_generate_conformers=False). "
                        f"Set auto_generate_conformers=True to enable automatic conformer "
                        f"generation, or provide molecules with pre-computed conformers. "
                        f"Alternatively, use a 2D featurizer (e.g., 'morgan', 'ecfp') "
                        f"that does not require conformers."
                    )

                logger.debug(
                    f"Generating conformers for {len(smiles_list)} molecules "
                    f"(3D fingerprint: {self.get_name()})"
                )
                mols = self.conformer_gen.transform(mols)

            features = self.fingerprint.transform(mols)

        except FeatureExtractionError:
            raise
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise FeatureExtractionError(
                f"Feature extraction failed for {self.get_name()} on "
                f"{len(smiles_list)} compounds: {e}. "
                f"Check SMILES validity with 'learnm8 validate your_file.csv'. "
                f"If using a 3D featurizer, check conformer generation settings."
            ) from None

        if features is None or len(features) == 0:
            raise FeatureExtractionError(
                f"No valid features generated by {self.get_name()} for "
                f"{len(smiles_list)} input compounds. All SMILES may be invalid. "
                f"Run 'learnm8 validate your_file.csv' to check SMILES validity."
            )

        features_array = np.array(features, dtype=np.float32)

        if np.any(~np.isfinite(features_array)):
            logger.warning(
                f"Non-finite values detected in {self.get_name()} features"
            )

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
                    f"Cannot determine feature dimension for {self.get_name()}. "
                    f"The fingerprint object lacks any of the expected attributes: "
                    f"n_features_out, n_features_out_, fp_size, or n_features. "
                    f"Ensure the fingerprint class is a valid scikit-fingerprints object."
                )

    def requires_3d(self) -> bool:
        """Check if fingerprint requires 3D conformers.

        Returns:
            Boolean from fingerprint's requires_conformers attribute

        Note:
            Returns True for 3D fingerprints (USR, WHIM, E3FP, GETAWAY,
            MORSE, RDF, Autocorr, ElectroShape, USRCAT).
        """
        return getattr(self.fingerprint, 'requires_conformers', False)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for cache key generation.

        Returns:
            Dictionary with all configuration parameters

        Note:
            Includes fingerprint class, auto_generate_conformers flag,
            conformer generation parameters, and all fingerprint-specific
            parameters from get_params().
        """
        config = {
            'fingerprint_class': self.fingerprint.__class__.__name__,
            'auto_generate_conformers': self.auto_generate_conformers,
            'n_jobs': self.n_jobs,
        }

        if self.conformer_params:
            config['conformer_params'] = self.conformer_params

        params = self.fingerprint.get_params()
        config.update(params)

        return config

    def validate_smiles(self, smiles_list: List[str]) -> List[bool]:
        """Validate SMILES using RDKit.

        Args:
            smiles_list: List of SMILES strings to validate

        Returns:
            List of booleans indicating validity of each SMILES
        """
        from rdkit import Chem
        return [Chem.MolFromSmiles(s) is not None for s in smiles_list]

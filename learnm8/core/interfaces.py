"""Core interfaces for the LearnM8 active learning framework.

This module defines the abstract base classes that establish the contracts
for all major components in the LearnM8 system.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import polars as pl
import numpy as np
import hashlib
import json


class Oracle(ABC):
    """Interface for measuring molecular properties.
    
    Oracles are responsible for evaluating compounds and returning their
    measured properties. This could be experimental measurements, 
    computational simulations, or lookup from databases.
    """
    
    @abstractmethod
    def measure(self, compounds: pl.DataFrame, properties: List[str]) -> pl.DataFrame:
        """
        Measure properties for given compounds.

        Args:
            compounds: Polars DataFrame with 'ID' and 'SMILES' columns
            properties: List of property names to measure

        Returns:
            Polars DataFrame with 'ID' column and measured property columns.
            The returned DataFrame MUST preserve the row order of the input
            compounds DataFrame to ensure correct value-to-compound alignment.

        Raises:
            ValueError: If compounds DataFrame is malformed
            RuntimeError: If measurement fails

        Note:
            Implementations must preserve input row order. The calling code
            relies on positional correspondence between input compound IDs
            and returned measured values.
        """
        pass


class Learner(ABC):
    """Base class for all machine learning models.

    Learners work with feature matrices (numpy arrays) and are agnostic to
    the molecular domain. Feature extraction happens at the API/cycle level.

    Some learners (e.g., Chemprop) work directly with SMILES strings instead
    of pre-computed features. These learners override requires_smiles() to
    return True and use the smiles parameter in train/predict.
    """

    @abstractmethod
    def train(self,
              features: np.ndarray,
              targets: np.ndarray,
              smiles: Optional[List[str]] = None) -> None:
        """
        Train the model on feature matrix or SMILES.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)
            smiles: Optional SMILES strings (required by some learners)

        Raises:
            ValueError: If input shapes invalid
            RuntimeError: If training fails
        """
        pass

    @abstractmethod
    def predict(self,
                features: np.ndarray,
                smiles: Optional[List[str]] = None
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict on feature matrix or SMILES.

        Args:
            features: Feature matrix (n_samples, n_features)
            smiles: Optional SMILES strings (required by some learners)

        Returns:
            Tuple of (predictions, uncertainties).
            uncertainties can be None if model doesn't provide uncertainty estimates.

        Raises:
            RuntimeError: If model is not trained or prediction fails
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return a descriptive name for this learner.
        
        Returns:
            String identifier for the learner type and configuration
        """
        pass
    
    def supports_uncertainty(self) -> bool:
        """Return True if this learner can provide uncertainty estimates.

        This method can be overridden by subclasses for efficiency.
        The default implementation attempts a test prediction to check
        if uncertainty is returned.

        Returns:
            Boolean indicating uncertainty support
        """
        # Default implementation - can be overridden for efficiency
        # Subclasses should override this method with actual logic
        return False

    def requires_smiles(self) -> bool:
        """Return True if this learner needs SMILES strings.

        Override in subclasses that work directly with molecular structures
        instead of pre-computed features (e.g., graph neural networks).

        Returns:
            False by default (feature-based learners)
        """
        return False


class AcquisitionFunction(ABC):
    """Base class for compound selection strategies.
    
    Acquisition functions determine which compounds to select for labeling
    in each active learning cycle based on model predictions and optionally
    uncertainty estimates.
    """
    
    @abstractmethod
    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """
        Select compounds for labeling.

        Args:
            compounds: Polars DataFrame with 'ID', 'SMILES', 'prediction' columns
                      May also contain 'uncertainty' column if available
            n_select: Number of compounds to select

        Returns:
            Polars DataFrame subset with selected compounds

        Raises:
            ValueError: If required columns are missing or n_select is invalid
            RuntimeError: If selection fails
        """
        pass
    
    def requires_uncertainty(self) -> bool:
        """Return True if this acquisition function requires uncertainty estimates.
        
        Returns:
            Boolean indicating if uncertainty column is required
        """
        return False
    
    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function.

        Returns:
            String identifier for the acquisition function
        """
        return self.__class__.__name__


class Featurizer(ABC):
    """Base class for all molecular featurizers.

    Featurizers convert SMILES strings into numerical feature representations
    suitable for machine learning. This interface enables custom featurizers
    to be easily integrated into the LearnM8 framework.

    This version is designed for scikit-fingerprints integration with
    built-in 3D conformer support.

    Example:
        class MyCustomFeaturizer(Featurizer):
            def transform(self, smiles_list):
                return features_array

            def get_dimension(self):
                return 2048

            def get_name(self):
                return 'custom_fp'

            def requires_3d(self):
                return False
    """

    @abstractmethod
    def transform(self, smiles_list: List[str]) -> np.ndarray:
        """Transform SMILES strings to feature matrix.

        Args:
            smiles_list: List of SMILES strings to featurize

        Returns:
            Feature matrix of shape (n_compounds, n_features)

        Raises:
            ValueError: If all SMILES are invalid
            RuntimeError: If featurization fails

        Note:
            Invalid SMILES should raise exceptions (not return zero vectors).
            This allows LearnM8's validation system to catch data quality issues.

            For 3D fingerprints (requires_3d() == True), conformers will be
            auto-generated if auto_generate_conformers=True (default).
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the feature dimension produced by this featurizer.

        Returns:
            Integer dimension of feature vectors

        Note:
            This should return a constant value for the featurizer type.
            Used for initializing empty arrays and validation.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get a descriptive name for this featurizer.

        Returns:
            String identifier (e.g., 'morgan', 'maccs', 'usr', 'whim')

        Note:
            Used for logging, cache file naming, and error messages.
        """
        pass

    def requires_3d(self) -> bool:
        """Return True if this featurizer requires 3D conformers.

        Returns:
            Boolean indicating if 3D molecular conformers are required

        Note:
            3D fingerprints (USR, WHIM, E3FP, GETAWAY, MORSE, RDF, etc.)
            return True. These fingerprints encode shape, spatial, or
            geometry-dependent information.

            Matches scikit-fingerprints' `requires_conformers` attribute.

            If True and auto_generate_conformers=True, conformers will be
            automatically generated using RDKit's ETKDG algorithm.
        """
        return False

    def supports_caching(self) -> bool:
        """Return True if this featurizer's output can be cached.

        Returns:
            Boolean indicating if features are deterministic and cacheable

        Note:
            Non-deterministic featurizers (e.g., using random projections)
            should return False to disable HDF5 caching.

            3D fingerprints with auto-generated conformers are still cacheable
            if conformer generation uses a fixed random seed.
        """
        return True

    def supports_batching(self) -> bool:
        """Return True if this featurizer can process batches efficiently.

        Returns:
            Boolean indicating if batched processing is more efficient than
            iterating over individual molecules

        Note:
            Most featurizers benefit from batching. Return False only if
            your implementation requires molecule-by-molecule processing.
        """
        return True

    def get_description(self) -> str:
        """Get a human-readable description of this featurizer.

        Returns:
            String description for documentation and CLI display

        Note:
            Optional method for better user experience. Default returns name.
        """
        return self.get_name()

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dictionary for this featurizer.

        Returns:
            Dictionary containing all configuration parameters

        Note:
            Used for cache key generation and serialization.
            Default returns empty dict. Override to include parameters.

            Should include all parameters that affect featurization output
            (e.g., radius, fp_size, include_chirality, etc.)
        """
        return {}

    def get_config_hash(self) -> str:
        """Get a hash of the featurizer configuration.

        Returns:
            MD5 hash string of the sorted configuration parameters

        Note:
            Used in cache key generation to prevent incorrect cache hits
            when parameters change. Automatically sorts keys for consistency.

            Critical for preventing bugs where changing parameters (e.g.,
            radius=2 → radius=3) returns wrong cached features.

            Includes featurizer name to prevent collisions between different
            featurizers with identical configurations.
        """
        config = self.get_config()
        config['_featurizer_name'] = self.get_name()
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def validate_smiles(self, smiles_list: List[str]) -> List[bool]:
        """Validate SMILES strings before featurization.

        Args:
            smiles_list: List of SMILES strings to validate

        Returns:
            List of booleans indicating validity of each SMILES

        Note:
            Optional pre-validation to provide better error messages.
            Default implementation returns all True (assumes valid).
            Override to implement custom validation logic.
        """
        return [True] * len(smiles_list)
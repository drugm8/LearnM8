"""Uncertainty-based acquisition functions for the LearnM8 framework.

This module provides sophisticated acquisition strategies that utilize model
uncertainty estimates for exploration-exploitation balance in active learning.
"""

import logging
from typing import Optional, TYPE_CHECKING
import numpy as np
import pandas as pd

from .base import AcquisitionFunction, validate_uncertainty_inputs

# Optional imports for statistical functions
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    norm = None
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class UCBAcquisition(AcquisitionFunction):
    """Upper Confidence Bound acquisition function.
    
    UCB balances exploitation (high predicted values) with exploration
    (high uncertainty) by computing upper confidence bounds using
    prediction + beta * uncertainty.
    """
    
    def __init__(self, beta: float = 2.0, **kwargs):
        """Initialize UCB acquisition function.

        Args:
            beta: Confidence parameter controlling exploration vs exploitation.
                  Higher values favor exploration.
            **kwargs: Additional parameters for compatibility
        """
        super().__init__(**kwargs)
        if beta < 0:
            raise ValueError("beta must be non-negative")
        
        self.beta = beta
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select using Upper Confidence Bound.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction', 'uncertainty' columns
            n_select: Number of compounds to select
            
        Returns:
            DataFrame subset with selected compounds
            
        Raises:
            ValueError: If uncertainty estimates are not available
        """
        # Validate input
        self.validate_input(compounds, n_select)
        
        # Extract predictions and uncertainties
        predictions, uncertainties = validate_uncertainty_inputs(compounds)

        # Calculate UCB scores based on score direction
        # Note: uncertainties are already standard deviations, not variances
        if self.maximize:
            # For maximization: select upper confidence bound
            ucb_scores = predictions + self.beta * uncertainties
        else:
            # For minimization: select lower confidence bound
            ucb_scores = predictions - self.beta * uncertainties

        # Select top compounds (always ascending=False because we want highest UCB scores)
        if self.maximize:
            ascending = False
        else:
            ascending = True
        selected = self._safe_select_top_k(
            compounds, ucb_scores, n_select, ascending=ascending
        )
        
        logger.debug(f"UCBAcquisition selected {len(selected)} compounds with β={self.beta}")
        
        return selected
    
    def requires_uncertainty(self) -> bool:
        """Return True since UCB requires uncertainty estimates."""
        return True
    
    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        return f"UCB(β={self.beta})"


class ExpectedImprovementAcquisition(AcquisitionFunction):
    """Expected Improvement acquisition function.
    
    EI calculates the expected improvement over the current best observed value,
    providing a principled way to balance exploration and exploitation.
    """
    
    def __init__(self,
                 xi: float = 0.01, minimize: bool = None, score_direction: str = 'higher',
                 current_best: Optional[float] = None,
                 **kwargs):
        """Initialize Expected Improvement acquisition function.

        Args:
            xi: Exploration parameter. Small positive values encourage exploration.
            minimize: DEPRECATED. Use score_direction instead. If provided, overrides score_direction.
            score_direction: Direction to optimize ('higher' or 'lower'). Default 'higher'
            current_best: Current best observed value from labeled data. Required for correct EI calculation.
            **kwargs: Additional parameters for compatibility
        """
        # Handle backward compatibility with minimize parameter
        if minimize is not None:
            import warnings
            warnings.warn(
                "The 'minimize' parameter is deprecated. Use 'score_direction' instead.",
                DeprecationWarning, stacklevel=2
            )
            score_direction = 'lower' if minimize else 'higher'

        super().__init__(score_direction=score_direction, **kwargs)
        if xi < 0:
            raise ValueError("xi must be non-negative")

        self.xi = xi
        self.current_best = current_best
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select using Expected Improvement.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction', 'uncertainty' columns
            n_select: Number of compounds to select
            
        Returns:
            DataFrame subset with selected compounds
            
        Raises:
            ValueError: If uncertainty estimates are not available
            RuntimeError: If scipy is not available for normal distribution calculations
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for Expected Improvement. Install with: pip install scipy")
        
        # Validate input
        self.validate_input(compounds, n_select)
        
        # Extract predictions and uncertainties
        predictions, uncertainties = validate_uncertainty_inputs(compounds)

        # Require current_best from labeled data
        if self.current_best is None:
            raise ValueError(
                "Expected Improvement requires 'current_best' parameter with the best observed value "
                "from labeled training data. This should be passed via acquisition_params at the cycle level."
            )

        current_best = self.current_best

        # Calculate improvement based on score direction
        if self.maximize:
            improvement = predictions - current_best - self.xi
        else:
            improvement = current_best - predictions - self.xi
        
        # Calculate standard deviations
        std_devs = np.sqrt(uncertainties)
        
        # Calculate Expected Improvement
        with np.errstate(divide="ignore", invalid="ignore"):
            z_scores = improvement / std_devs
        
        # Calculate EI using normal distribution
        ei_scores = improvement * norm.cdf(z_scores) + std_devs * norm.pdf(z_scores)
        
        # Handle zero variance case
        zero_var_mask = uncertainties == 0
        ei_scores[zero_var_mask] = np.maximum(improvement[zero_var_mask], 0)
        
        # Select top compounds
        selected = self._safe_select_top_k(
            compounds, ei_scores, n_select, ascending=False
        )
        
        logger.debug(f"ExpectedImprovementAcquisition selected {len(selected)} compounds "
                    f"with ξ={self.xi}, current_best={current_best:.3f}")
        
        return selected
    
    def requires_uncertainty(self) -> bool:
        """Return True since EI requires uncertainty estimates."""
        return True
    
    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        direction = "max" if self.maximize else "min"
        return f"EI(ξ={self.xi},{direction})"


class ProbabilityImprovementAcquisition(AcquisitionFunction):
    """Probability of Improvement acquisition function.
    
    PI calculates the probability that a compound will improve over
    the current best observed value.
    """
    
    def __init__(self,
                 xi: float = 0.01, minimize: bool = None, score_direction: str = 'higher',
                 current_best: Optional[float] = None,
                 **kwargs):
        """Initialize Probability of Improvement acquisition function.

        Args:
            xi: Exploration parameter. Small positive values encourage exploration.
            minimize: DEPRECATED. Use score_direction instead. If provided, overrides score_direction.
            score_direction: Direction to optimize ('higher' or 'lower'). Default 'higher'
            current_best: Current best observed value from labeled data. Required for correct PI calculation.
            **kwargs: Additional parameters for compatibility
        """
        # Handle backward compatibility with minimize parameter
        if minimize is not None:
            import warnings
            warnings.warn(
                "The 'minimize' parameter is deprecated. Use 'score_direction' instead.",
                DeprecationWarning, stacklevel=2
            )
            score_direction = 'lower' if minimize else 'higher'

        super().__init__(score_direction=score_direction, **kwargs)
        if xi < 0:
            raise ValueError("xi must be non-negative")

        self.xi = xi
        self.current_best = current_best
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select using Probability of Improvement.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction', 'uncertainty' columns
            n_select: Number of compounds to select
            
        Returns:
            DataFrame subset with selected compounds
            
        Raises:
            ValueError: If uncertainty estimates are not available
            RuntimeError: If scipy is not available for normal distribution calculations
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for Probability of Improvement. Install with: pip install scipy")
        
        # Validate input
        self.validate_input(compounds, n_select)
        
        # Extract predictions and uncertainties
        predictions, uncertainties = validate_uncertainty_inputs(compounds)

        # Require current_best from labeled data
        if self.current_best is None:
            raise ValueError(
                "Probability of Improvement requires 'current_best' parameter with the best observed value "
                "from labeled training data. This should be passed via acquisition_params at the cycle level."
            )

        current_best = self.current_best

        # Calculate improvement based on score direction
        if self.maximize:
            improvement = predictions - current_best - self.xi
        else:
            improvement = current_best - predictions - self.xi
        
        # Calculate standard deviations
        std_devs = np.sqrt(uncertainties)
        
        # Calculate Probability of Improvement
        with np.errstate(divide="ignore"):
            z_scores = improvement / std_devs
        
        pi_scores = norm.cdf(z_scores)
        
        # Handle zero variance case
        zero_var_mask = uncertainties == 0
        pi_scores[zero_var_mask] = np.where(improvement[zero_var_mask] > 0, 1.0, 0.0)
        
        # Select top compounds
        selected = self._safe_select_top_k(
            compounds, pi_scores, n_select, ascending=False
        )
        
        logger.debug(f"ProbabilityImprovementAcquisition selected {len(selected)} compounds "
                    f"with ξ={self.xi}, current_best={current_best:.3f}")
        
        return selected
    
    def requires_uncertainty(self) -> bool:
        """Return True since PI requires uncertainty estimates."""
        return True
    
    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        direction = "max" if self.maximize else "min"
        return f"PI(ξ={self.xi},{direction})"


class ThompsonSamplingAcquisition(AcquisitionFunction):
    """Thompson Sampling acquisition function.
    
    Thompson sampling draws samples from the posterior predictive distribution
    and selects compounds with the highest sampled values, providing a
    stochastic exploration strategy.
    """
    
    def __init__(self, random_state: int = 42, **kwargs):
        """Initialize Thompson Sampling acquisition function.

        Args:
            random_state: Random seed for reproducible sampling
            **kwargs: Additional parameters for compatibility
        """
        super().__init__(**kwargs)
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select using Thompson Sampling.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction', 'uncertainty' columns
            n_select: Number of compounds to select
            
        Returns:
            DataFrame subset with selected compounds
            
        Raises:
            ValueError: If uncertainty estimates are not available
        """
        # Validate input
        self.validate_input(compounds, n_select)
        
        # Extract predictions and uncertainties
        predictions, uncertainties = validate_uncertainty_inputs(compounds)
        
        # Sample from posterior predictive distribution
        std_devs = np.sqrt(uncertainties)
        thompson_samples = self._rng.normal(predictions, std_devs)
        
        # Select top compounds based on samples and score direction
        selected = self._safe_select_top_k(
            compounds, thompson_samples, n_select, ascending=not self.maximize
        )
        
        logger.debug(f"ThompsonSamplingAcquisition selected {len(selected)} compounds "
                    f"with random_state={self.random_state}")
        
        return selected
    
    def requires_uncertainty(self) -> bool:
        """Return True since Thompson sampling requires uncertainty estimates."""
        return True
    
    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        return f"Thompson(seed={self.random_state})"


class EntropyAcquisition(AcquisitionFunction):
    """Entropy-based acquisition function for maximum information gain.
    
    This acquisition function selects compounds that are expected to provide
    the most information, measured by prediction entropy or uncertainty.
    """
    
    def __init__(self, entropy_type: str = 'uncertainty', **kwargs):
        """Initialize Entropy acquisition function.

        Args:
            entropy_type: Type of entropy measure ('uncertainty', 'variance')
            **kwargs: Additional parameters for compatibility
        """
        super().__init__(**kwargs)
        if entropy_type not in ['uncertainty', 'variance']:
            raise ValueError("entropy_type must be 'uncertainty' or 'variance'")
        
        self.entropy_type = entropy_type
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select using entropy-based information criterion.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction', 'uncertainty' columns
            n_select: Number of compounds to select
            
        Returns:
            DataFrame subset with selected compounds
            
        Raises:
            ValueError: If uncertainty estimates are not available
        """
        # Validate input
        self.validate_input(compounds, n_select)
        
        # Extract predictions and uncertainties
        predictions, uncertainties = validate_uncertainty_inputs(compounds)
        
        if self.entropy_type == 'uncertainty':
            # Use uncertainty directly as information measure
            entropy_scores = uncertainties
        else:  # variance
            # Use variance (square of uncertainty) as information measure
            entropy_scores = uncertainties ** 2
        
        # Select compounds with highest entropy/information
        selected = self._safe_select_top_k(
            compounds, entropy_scores, n_select, ascending=False
        )
        
        logger.debug(f"EntropyAcquisition selected {len(selected)} compounds "
                    f"using {self.entropy_type} entropy")
        
        return selected
    
    def requires_uncertainty(self) -> bool:
        """Return True since entropy acquisition requires uncertainty estimates."""
        return True
    
    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        return f"Entropy({self.entropy_type})"
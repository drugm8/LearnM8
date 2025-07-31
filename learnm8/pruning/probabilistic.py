"""Probabilistic pruning strategies for the LearnM8 framework.

This module provides pruning strategies based on statistical analysis of
model predictions and uncertainties to remove compounds unlikely to be valuable.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .base import DesignSpacePruner, PruningError, calculate_confidence_intervals

# Optional imports for statistical functions
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    norm = None
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProbabilisticPruner(DesignSpacePruner):
    """Probabilistic pruning based on compound value probability estimates.
    
    This pruner estimates the probability that each compound will be valuable
    (e.g., above a certain threshold) and removes compounds with low probability.
    """
    
    def __init__(self, 
                 value_threshold: float,
                 probability_threshold: float = 0.1,
                 minimize: bool = False):
        """Initialize probabilistic pruner.
        
        Args:
            value_threshold: Threshold for considering a compound "valuable"
            probability_threshold: Minimum probability for retention
            minimize: If True, seek values below threshold; if False, above threshold
        """
        self.value_threshold = value_threshold
        self.probability_threshold = probability_threshold
        self.minimize = minimize
        self.last_pruning_stats = {}
        
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for ProbabilisticPruner. Install with: pip install scipy")
    
    def prune(self, 
              compounds: pd.DataFrame, 
              predictions: np.ndarray, 
              uncertainties: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Prune compounds based on probability of being valuable.
        
        Args:
            compounds: DataFrame with compound information
            predictions: Model predictions
            uncertainties: Model uncertainties (standard deviations)
            
        Returns:
            Pruned DataFrame with compounds likely to be valuable
        """
        # Validate inputs
        self.validate_inputs(compounds, predictions, uncertainties)
        
        original_count = len(compounds)
        
        # Calculate probabilities
        if uncertainties is not None:
            probabilities = self._calculate_value_probabilities(predictions, uncertainties)
        else:
            # Fallback: use deterministic threshold without uncertainty
            if self.minimize:
                probabilities = (predictions <= self.value_threshold).astype(float)
            else:
                probabilities = (predictions >= self.value_threshold).astype(float)
        
        # Select compounds above probability threshold
        keep_mask = probabilities >= self.probability_threshold
        pruned_compounds = self._safe_prune_by_indices(compounds, keep_mask)
        
        # Update statistics
        pruned_count = len(pruned_compounds)
        self.last_pruning_stats = {
            'compounds_before_pruning': original_count,
            'compounds_after_pruning': pruned_count,
            'compounds_pruned': original_count - pruned_count,
            'pruning_fraction': self._calculate_pruning_fraction(original_count, pruned_count),
            'mean_probability': float(np.mean(probabilities)),
            'mean_kept_probability': float(np.mean(probabilities[keep_mask])) if np.any(keep_mask) else 0.0,
            'probability_threshold': self.probability_threshold,
            'value_threshold': self.value_threshold
        }
        
        logger.info(f"ProbabilisticPruner removed {original_count - pruned_count} compounds "
                   f"({self.last_pruning_stats['pruning_fraction']:.1%}), "
                   f"kept {pruned_count} with probability >= {self.probability_threshold}")
        
        return pruned_compounds
    
    def _calculate_value_probabilities(self, 
                                     predictions: np.ndarray, 
                                     uncertainties: np.ndarray) -> np.ndarray:
        """Calculate probability that each compound exceeds value threshold.
        
        Args:
            predictions: Model predictions (means)
            uncertainties: Model uncertainties (standard deviations)
            
        Returns:
            Array of probabilities
        """
        # Calculate z-scores for threshold
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.minimize:
                # P(value <= threshold) for minimization
                z_scores = (self.value_threshold - predictions) / uncertainties
            else:
                # P(value >= threshold) for maximization  
                z_scores = (predictions - self.value_threshold) / uncertainties
        
        # Calculate probabilities using normal CDF
        probabilities = norm.cdf(z_scores)
        
        # Handle zero uncertainty case
        zero_uncertainty_mask = uncertainties == 0
        if np.any(zero_uncertainty_mask):
            if self.minimize:
                deterministic_probs = (predictions <= self.value_threshold).astype(float)
            else:
                deterministic_probs = (predictions >= self.value_threshold).astype(float)
            
            probabilities[zero_uncertainty_mask] = deterministic_probs[zero_uncertainty_mask]
        
        return probabilities
    
    def requires_uncertainty(self) -> bool:
        """Return True since probabilistic pruning benefits from uncertainty."""
        return True
    
    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get statistics from the most recent pruning operation."""
        return self.last_pruning_stats.copy()
    
    def get_name(self) -> str:
        """Return a descriptive name for this pruning strategy."""
        direction = "min" if self.minimize else "max"
        return f"Probabilistic(threshold={self.value_threshold},{direction},p>={self.probability_threshold})"


class UncertaintyThresholdPruner(DesignSpacePruner):
    """Pruning strategy that removes compounds with high uncertainty.
    
    This pruner removes compounds where the model is very uncertain,
    focusing on regions where the model is more confident.
    """
    
    def __init__(self, 
                 uncertainty_threshold: Optional[float] = None,
                 retention_fraction: float = 0.8,
                 adaptive_threshold: bool = True):
        """Initialize uncertainty threshold pruner.
        
        Args:
            uncertainty_threshold: Fixed uncertainty threshold (if not adaptive)
            retention_fraction: Fraction of compounds to retain when adaptive
            adaptive_threshold: Whether to calculate threshold adaptively
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.retention_fraction = retention_fraction
        self.adaptive_threshold = adaptive_threshold
        self.last_pruning_stats = {}
        
        if not adaptive_threshold and uncertainty_threshold is None:
            raise ValueError("uncertainty_threshold must be provided when adaptive_threshold=False")
    
    def prune(self, 
              compounds: pd.DataFrame, 
              predictions: np.ndarray, 
              uncertainties: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Prune compounds with high uncertainty.
        
        Args:
            compounds: DataFrame with compound information
            predictions: Model predictions  
            uncertainties: Model uncertainties (required)
            
        Returns:
            Pruned DataFrame with low-uncertainty compounds
        """
        # Validate inputs
        self.validate_inputs(compounds, predictions, uncertainties)
        
        if uncertainties is None:
            raise PruningError("UncertaintyThresholdPruner requires uncertainty estimates")
        
        original_count = len(compounds)
        
        # Determine threshold
        if self.adaptive_threshold:
            # Calculate threshold to retain target fraction
            percentile = (1 - self.retention_fraction) * 100
            threshold = np.percentile(uncertainties, 100 - percentile)
        else:
            threshold = self.uncertainty_threshold
        
        # Select compounds below uncertainty threshold
        keep_mask = uncertainties <= threshold
        pruned_compounds = self._safe_prune_by_indices(compounds, keep_mask)
        
        # Update statistics
        pruned_count = len(pruned_compounds)
        self.last_pruning_stats = {
            'compounds_before_pruning': original_count,
            'compounds_after_pruning': pruned_count,
            'compounds_pruned': original_count - pruned_count,
            'pruning_fraction': self._calculate_pruning_fraction(original_count, pruned_count),
            'uncertainty_threshold': float(threshold),
            'mean_uncertainty': float(np.mean(uncertainties)),
            'mean_kept_uncertainty': float(np.mean(uncertainties[keep_mask])) if np.any(keep_mask) else 0.0,
            'uncertainty_std': float(np.std(uncertainties)),
            'adaptive_threshold': self.adaptive_threshold
        }
        
        logger.info(f"UncertaintyThresholdPruner removed {original_count - pruned_count} compounds "
                   f"({self.last_pruning_stats['pruning_fraction']:.1%}) "
                   f"with uncertainty > {threshold:.3f}")
        
        return pruned_compounds
    
    def requires_uncertainty(self) -> bool:
        """Return True since this strategy requires uncertainty estimates."""
        return True
    
    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get statistics from the most recent pruning operation."""
        return self.last_pruning_stats.copy()
    
    def get_name(self) -> str:
        """Return a descriptive name for this pruning strategy."""
        if self.adaptive_threshold:
            return f"UncertaintyThreshold(adaptive,retain={self.retention_fraction})"
        else:
            return f"UncertaintyThreshold(fixed={self.uncertainty_threshold})"


class PredictionThresholdPruner(DesignSpacePruner):
    """Simple prediction-based pruning strategy.
    
    This pruner removes compounds with predictions below (or above) a threshold,
    effectively focusing on the most promising regions of chemical space.
    """
    
    def __init__(self, 
                 prediction_threshold: Optional[float] = None,
                 retention_fraction: float = 0.5,
                 adaptive_threshold: bool = True,
                 maximize: bool = True):
        """Initialize prediction threshold pruner.
        
        Args:
            prediction_threshold: Fixed prediction threshold (if not adaptive)
            retention_fraction: Fraction of compounds to retain when adaptive
            adaptive_threshold: Whether to calculate threshold adaptively
            maximize: If True, keep high predictions; if False, keep low predictions
        """
        self.prediction_threshold = prediction_threshold
        self.retention_fraction = retention_fraction
        self.adaptive_threshold = adaptive_threshold
        self.maximize = maximize
        self.last_pruning_stats = {}
        
        if not adaptive_threshold and prediction_threshold is None:
            raise ValueError("prediction_threshold must be provided when adaptive_threshold=False")
    
    def prune(self, 
              compounds: pd.DataFrame, 
              predictions: np.ndarray, 
              uncertainties: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Prune compounds based on prediction threshold.
        
        Args:
            compounds: DataFrame with compound information
            predictions: Model predictions
            uncertainties: Model uncertainties (not used)
            
        Returns:
            Pruned DataFrame with compounds above/below threshold
        """
        # Validate inputs
        self.validate_inputs(compounds, predictions, uncertainties)
        
        original_count = len(compounds)
        
        # Determine threshold
        if self.adaptive_threshold:
            if self.maximize:
                # Keep top fraction
                percentile = (1 - self.retention_fraction) * 100
                threshold = np.percentile(predictions, percentile)
            else:
                # Keep bottom fraction
                percentile = self.retention_fraction * 100
                threshold = np.percentile(predictions, percentile)
        else:
            threshold = self.prediction_threshold
        
        # Select compounds based on threshold
        if self.maximize:
            keep_mask = predictions >= threshold
        else:
            keep_mask = predictions <= threshold
        
        pruned_compounds = self._safe_prune_by_indices(compounds, keep_mask)
        
        # Update statistics
        pruned_count = len(pruned_compounds)
        self.last_pruning_stats = {
            'compounds_before_pruning': original_count,
            'compounds_after_pruning': pruned_count,
            'compounds_pruned': original_count - pruned_count,
            'pruning_fraction': self._calculate_pruning_fraction(original_count, pruned_count),
            'prediction_threshold': float(threshold),
            'mean_prediction': float(np.mean(predictions)),
            'mean_kept_prediction': float(np.mean(predictions[keep_mask])) if np.any(keep_mask) else 0.0,
            'prediction_std': float(np.std(predictions)),
            'adaptive_threshold': self.adaptive_threshold,
            'maximize': self.maximize
        }
        
        direction = ">=" if self.maximize else "<="
        logger.info(f"PredictionThresholdPruner removed {original_count - pruned_count} compounds "
                   f"({self.last_pruning_stats['pruning_fraction']:.1%}) "
                   f"with prediction {direction} {threshold:.3f}")
        
        return pruned_compounds
    
    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get statistics from the most recent pruning operation."""
        return self.last_pruning_stats.copy()
    
    def get_name(self) -> str:
        """Return a descriptive name for this pruning strategy."""
        direction = "max" if self.maximize else "min"
        if self.adaptive_threshold:
            return f"PredictionThreshold(adaptive,{direction},retain={self.retention_fraction})"
        else:
            return f"PredictionThreshold(fixed={self.prediction_threshold},{direction})"


class ConfidenceIntervalPruner(DesignSpacePruner):
    """Pruning based on confidence intervals around predictions.
    
    This pruner removes compounds whose confidence intervals do not overlap
    with a target region of interest, focusing on compounds likely to be
    in the desired value range.
    """
    
    def __init__(self, 
                 target_min: float,
                 target_max: float,
                 confidence_level: float = 0.95,
                 overlap_requirement: str = 'any'):
        """Initialize confidence interval pruner.
        
        Args:
            target_min: Minimum value of target region
            target_max: Maximum value of target region
            confidence_level: Confidence level for intervals (0-1)
            overlap_requirement: 'any' for any overlap, 'complete' for complete containment
        """
        if target_min >= target_max:
            raise ValueError("target_min must be less than target_max")
        
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        
        if overlap_requirement not in ['any', 'complete']:
            raise ValueError("overlap_requirement must be 'any' or 'complete'")
        
        self.target_min = target_min
        self.target_max = target_max
        self.confidence_level = confidence_level
        self.overlap_requirement = overlap_requirement
        self.last_pruning_stats = {}
        
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for ConfidenceIntervalPruner. Install with: pip install scipy")
    
    def prune(self, 
              compounds: pd.DataFrame, 
              predictions: np.ndarray, 
              uncertainties: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Prune compounds based on confidence interval overlap with target region.
        
        Args:
            compounds: DataFrame with compound information
            predictions: Model predictions
            uncertainties: Model uncertainties (required)
            
        Returns:
            Pruned DataFrame with compounds whose confidence intervals overlap target
        """
        # Validate inputs
        self.validate_inputs(compounds, predictions, uncertainties)
        
        if uncertainties is None:
            raise PruningError("ConfidenceIntervalPruner requires uncertainty estimates")
        
        original_count = len(compounds)
        
        # Calculate confidence intervals
        lower_bounds, upper_bounds = calculate_confidence_intervals(
            predictions, uncertainties, self.confidence_level
        )
        
        # Determine overlap with target region
        if self.overlap_requirement == 'any':
            # Keep if confidence interval has any overlap with target region
            keep_mask = (lower_bounds <= self.target_max) & (upper_bounds >= self.target_min)
        else:  # 'complete'
            # Keep if confidence interval is completely within target region
            keep_mask = (lower_bounds >= self.target_min) & (upper_bounds <= self.target_max)
        
        pruned_compounds = self._safe_prune_by_indices(compounds, keep_mask)
        
        # Update statistics
        pruned_count = len(pruned_compounds)
        
        # Calculate overlap statistics
        overlapping_compounds = np.sum(keep_mask)
        mean_interval_width = np.mean(upper_bounds - lower_bounds)
        
        self.last_pruning_stats = {
            'compounds_before_pruning': original_count,
            'compounds_after_pruning': pruned_count,
            'compounds_pruned': original_count - pruned_count,
            'pruning_fraction': self._calculate_pruning_fraction(original_count, pruned_count),
            'target_min': self.target_min,
            'target_max': self.target_max,
            'confidence_level': self.confidence_level,
            'overlap_requirement': self.overlap_requirement,
            'mean_interval_width': float(mean_interval_width),
            'overlapping_compounds': int(overlapping_compounds),
            'overlap_fraction': float(overlapping_compounds / original_count)
        }
        
        logger.info(f"ConfidenceIntervalPruner removed {original_count - pruned_count} compounds "
                   f"({self.last_pruning_stats['pruning_fraction']:.1%}) "
                   f"without {self.overlap_requirement} overlap with target region "
                   f"[{self.target_min:.3f}, {self.target_max:.3f}]")
        
        return pruned_compounds
    
    def requires_uncertainty(self) -> bool:
        """Return True since this strategy requires uncertainty estimates."""
        return True
    
    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get statistics from the most recent pruning operation."""
        return self.last_pruning_stats.copy()
    
    def get_name(self) -> str:
        """Return a descriptive name for this pruning strategy."""
        return (f"ConfidenceInterval({self.overlap_requirement},"
                f"target=[{self.target_min:.2f},{self.target_max:.2f}],"
                f"conf={self.confidence_level})")
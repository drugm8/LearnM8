"""Adaptive pruning strategies for the LearnM8 framework.

This module provides pruning strategies that adapt their behavior over active
learning cycles based on performance metrics and experimental progress.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from .base import StatefulPruner, PruningError

logger = logging.getLogger(__name__)


class CycleBudgetPruner(StatefulPruner):
    """Budget-aware pruning that adjusts aggressiveness based on remaining cycles.
    
    This pruner becomes more aggressive as the budget is consumed, ensuring
    efficient use of oracle evaluations across the entire experiment.
    """
    
    def __init__(self, 
                 total_cycles: int,
                 initial_retention_fraction: float = 0.8,
                 final_retention_fraction: float = 0.3,
                 strategy: str = 'linear'):
        """Initialize cycle budget pruner.
        
        Args:
            total_cycles: Total number of cycles in the experiment
            initial_retention_fraction: Retention fraction for early cycles
            final_retention_fraction: Retention fraction for late cycles
            strategy: Adaptation strategy ('linear', 'exponential', 'step')
        """
        super().__init__()
        
        self.total_cycles = total_cycles
        self.initial_retention_fraction = initial_retention_fraction
        self.final_retention_fraction = final_retention_fraction
        self.strategy = strategy
        
        if strategy not in ['linear', 'exponential', 'step']:
            raise ValueError("strategy must be 'linear', 'exponential', or 'step'")
        
        self.last_pruning_stats = {}
    
    def prune(self, 
              compounds: pd.DataFrame, 
              predictions: np.ndarray, 
              uncertainties: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Prune compounds with budget-aware retention fraction.
        
        Args:
            compounds: DataFrame with compound information
            predictions: Model predictions
            uncertainties: Model uncertainties (not used)
            
        Returns:
            Pruned DataFrame with budget-aware retention
        """
        # Validate inputs
        self.validate_inputs(compounds, predictions, uncertainties)
        
        original_count = len(compounds)
        
        # Calculate current retention fraction based on cycle progress
        current_retention = self._calculate_retention_fraction()
        
        # Apply prediction-based threshold
        threshold = np.percentile(predictions, (1 - current_retention) * 100)
        keep_mask = predictions >= threshold
        
        pruned_compounds = self._safe_prune_by_indices(compounds, keep_mask)
        
        # Update statistics
        pruned_count = len(pruned_compounds)
        cycle_stats = {
            'compounds_before_pruning': original_count,
            'compounds_after_pruning': pruned_count,
            'compounds_pruned': original_count - pruned_count,
            'pruning_fraction': self._calculate_pruning_fraction(original_count, pruned_count),
            'retention_fraction': current_retention,
            'cycle_progress': self.cycle_count / self.total_cycles,
            'prediction_threshold': float(threshold)
        }
        
        self.last_pruning_stats = cycle_stats
        self.update_cycle_state(cycle_stats)
        
        logger.info(f"CycleBudgetPruner removed {original_count - pruned_count} compounds "
                   f"({cycle_stats['pruning_fraction']:.1%}) "
                   f"with retention {current_retention:.3f} "
                   f"(cycle {self.cycle_count}/{self.total_cycles})")
        
        return pruned_compounds
    
    def _calculate_retention_fraction(self) -> float:
        """Calculate retention fraction based on cycle progress.
        
        Returns:
            Current retention fraction
        """
        if self.total_cycles <= 0:
            return self.initial_retention_fraction
        
        progress = min(self.cycle_count / self.total_cycles, 1.0)
        
        if self.strategy == 'linear':
            # Linear interpolation
            retention = (self.initial_retention_fraction + 
                        progress * (self.final_retention_fraction - self.initial_retention_fraction))
        
        elif self.strategy == 'exponential':
            # Exponential decay
            decay_rate = np.log(self.final_retention_fraction / self.initial_retention_fraction)
            retention = self.initial_retention_fraction * np.exp(decay_rate * progress)
        
        elif self.strategy == 'step':
            # Step function (aggressive pruning in second half)
            if progress < 0.5:
                retention = self.initial_retention_fraction
            else:
                retention = self.final_retention_fraction
        
        else:
            retention = self.initial_retention_fraction
        
        return retention
    
    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get statistics from the most recent pruning operation."""
        stats = self.last_pruning_stats.copy()
        stats.update({
            'total_cycles': self.total_cycles,
            'strategy': self.strategy,
            'initial_retention': self.initial_retention_fraction,
            'final_retention': self.final_retention_fraction
        })
        return stats
    
    def get_name(self) -> str:
        """Return a descriptive name for this pruning strategy."""
        return f"CycleBudget({self.strategy},{self.initial_retention_fraction}->{self.final_retention_fraction})"


class PerformanceBasedPruner(StatefulPruner):
    """Performance-based adaptive pruning strategy.
    
    This pruner adjusts its aggressiveness based on the actual performance
    improvements observed in recent cycles, becoming more aggressive when
    performance stagnates.
    """
    
    def __init__(self, 
                 base_retention_fraction: float = 0.6,
                 performance_window: int = 3,
                 improvement_threshold: float = 0.01,
                 aggressive_retention: float = 0.3,
                 conservative_retention: float = 0.8):
        """Initialize performance-based pruner.
        
        Args:
            base_retention_fraction: Default retention fraction
            performance_window: Number of cycles to consider for performance
            improvement_threshold: Minimum improvement to avoid aggressive pruning
            aggressive_retention: Retention when performance stagnates
            conservative_retention: Retention when performance improves
        """
        super().__init__()
        
        self.base_retention_fraction = base_retention_fraction
        self.performance_window = performance_window
        self.improvement_threshold = improvement_threshold
        self.aggressive_retention = aggressive_retention
        self.conservative_retention = conservative_retention
        
        self.performance_metrics = []
        self.last_pruning_stats = {}
    
    def prune(self, 
              compounds: pd.DataFrame, 
              predictions: np.ndarray, 
              uncertainties: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Prune compounds based on recent performance.
        
        Args:
            compounds: DataFrame with compound information
            predictions: Model predictions
            uncertainties: Model uncertainties (not used)
            
        Returns:
            Pruned DataFrame with performance-based retention
        """
        # Validate inputs
        self.validate_inputs(compounds, predictions, uncertainties)
        
        original_count = len(compounds)
        
        # Determine retention fraction based on performance
        current_retention = self._calculate_performance_based_retention()
        
        # Apply prediction-based threshold
        threshold = np.percentile(predictions, (1 - current_retention) * 100)
        keep_mask = predictions >= threshold
        
        pruned_compounds = self._safe_prune_by_indices(compounds, keep_mask)
        
        # Update statistics
        pruned_count = len(pruned_compounds)
        cycle_stats = {
            'compounds_before_pruning': original_count,
            'compounds_after_pruning': pruned_count,
            'compounds_pruned': original_count - pruned_count,
            'pruning_fraction': self._calculate_pruning_fraction(original_count, pruned_count),
            'retention_fraction': current_retention,
            'performance_based': True,
            'recent_improvement': self._calculate_recent_improvement()
        }
        
        self.last_pruning_stats = cycle_stats
        self.update_cycle_state(cycle_stats)
        
        logger.info(f"PerformanceBasedPruner removed {original_count - pruned_count} compounds "
                   f"({cycle_stats['pruning_fraction']:.1%}) "
                   f"with performance-based retention {current_retention:.3f}")
        
        return pruned_compounds
    
    def update_performance_metric(self, metric_value: float) -> None:
        """Update performance metric for adaptation.
        
        Args:
            metric_value: Performance metric value (higher is better)
        """
        self.performance_metrics.append(metric_value)
        
        # Keep only recent metrics
        if len(self.performance_metrics) > self.performance_window * 2:
            self.performance_metrics = self.performance_metrics[-self.performance_window * 2:]
    
    def _calculate_performance_based_retention(self) -> float:
        """Calculate retention fraction based on recent performance.
        
        Returns:
            Performance-based retention fraction
        """
        recent_improvement = self._calculate_recent_improvement()
        
        if recent_improvement >= self.improvement_threshold:
            # Good performance, be conservative
            return self.conservative_retention
        elif recent_improvement <= -self.improvement_threshold:
            # Poor performance, be aggressive
            return self.aggressive_retention
        else:
            # Stable performance, use base retention
            return self.base_retention_fraction
    
    def _calculate_recent_improvement(self) -> float:
        """Calculate recent performance improvement.
        
        Returns:
            Recent improvement rate
        """
        if len(self.performance_metrics) < self.performance_window:
            return 0.0  # Not enough data
        
        recent_metrics = self.performance_metrics[-self.performance_window:]
        earlier_metrics = self.performance_metrics[-2*self.performance_window:-self.performance_window]
        
        if len(earlier_metrics) < self.performance_window:
            return 0.0  # Not enough data for comparison
        
        recent_mean = np.mean(recent_metrics)
        earlier_mean = np.mean(earlier_metrics)
        
        if earlier_mean == 0:
            return 0.0
        
        return (recent_mean - earlier_mean) / abs(earlier_mean)
    
    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get statistics from the most recent pruning operation."""
        stats = self.last_pruning_stats.copy()
        stats.update({
            'base_retention': self.base_retention_fraction,
            'performance_window': self.performance_window,
            'improvement_threshold': self.improvement_threshold,
            'performance_metrics_count': len(self.performance_metrics)
        })
        return stats
    
    def get_name(self) -> str:
        """Return a descriptive name for this pruning strategy."""
        return f"PerformanceBased(base={self.base_retention_fraction},window={self.performance_window})"
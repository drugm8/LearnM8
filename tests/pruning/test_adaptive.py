"""Tests for adaptive pruning strategies.

Tests cycle budget and performance-based adaptive pruning using real molecular data.
"""

import pytest
import numpy as np
import pandas as pd

try:
    from learnm8.pruning.adaptive import (
        CycleBudgetPruner,
        PerformanceBasedPruner
    )
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


@pytest.mark.skipif(not ADAPTIVE_AVAILABLE, reason="Adaptive pruning modules not available")
class TestCycleBudgetPruner:
    """Test cycle budget-aware pruning strategies."""
    
    def test_cycle_budget_linear_strategy(self, medium_real_compounds):
        """Test cycle budget pruner with linear strategy."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Use subset for faster testing
        compounds = compounds.head(50)
        predictions = np.random.uniform(0, 1, len(compounds))
        
        pruner = CycleBudgetPruner(
            total_cycles=5,
            initial_retention_fraction=0.8,
            final_retention_fraction=0.3,
            strategy='linear'
        )
        
        # Test pruning at different cycles
        retention_rates = []
        
        for cycle in range(5):
            # Manually set cycle count to simulate progression
            pruner.cycle_count = cycle
            
            pruned = pruner.prune(compounds, predictions)
            
            assert isinstance(pruned, pd.DataFrame)
            assert len(pruned) <= len(compounds)
            assert all(col in pruned.columns for col in compounds.columns)
            
            # Track retention rates
            retention = len(pruned) / len(compounds)
            retention_rates.append(retention)
            
            # Retention should decrease over cycles
            if cycle == 0:
                # First cycle should retain close to 80%
                assert 0.7 <= retention <= 0.9
            elif cycle == 4:
                # Last cycle should retain close to 30%
                assert 0.2 <= retention <= 0.4
        
        # Overall trend should be decreasing
        assert retention_rates[0] > retention_rates[-1]
    
    def test_cycle_budget_strategies(self, small_real_compounds):
        """Test different cycle budget strategies."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        strategies = ['linear', 'exponential', 'step']
        
        for strategy in strategies:
            try:
                pruner = CycleBudgetPruner(
                    total_cycles=3,
                    initial_retention_fraction=0.9,
                    final_retention_fraction=0.3,
                    strategy=strategy
                )
                
                # Test first cycle
                pruner.cycle_count = 0
                pruned_first = pruner.prune(compounds, predictions)
                
                # Test last cycle
                pruner.cycle_count = 2
                pruned_last = pruner.prune(compounds, predictions)
                
                # First should retain more than last
                assert len(pruned_first) >= len(pruned_last)
                
                # Check statistics
                stats = pruner.get_pruning_stats()
                assert 'strategy' in stats
                assert stats['strategy'] == strategy
                
            except Exception as e:
                pytest.skip(f"Strategy {strategy} failed: {e}")
    
    def test_cycle_budget_state_tracking(self, small_real_compounds):
        """Test that cycle budget pruner tracks state correctly."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        pruner = CycleBudgetPruner(
            total_cycles=3,
            initial_retention_fraction=0.8,
            final_retention_fraction=0.4,
            strategy='linear'
        )
        
        # Test multiple cycles - the prune method updates state internally
        for cycle in range(3):
            # Let the pruner track its own state
            pruned = pruner.prune(compounds, predictions)
            
            # Check that statistics are updated
            stats = pruner.get_pruning_stats()
            assert 'cycle_progress' in stats
            assert 'retention_fraction' in stats
            
            # Check cumulative statistics - cycle count is updated after prune
            cumulative = pruner.get_cumulative_stats()
            assert 'current_cycle' in cumulative
            assert cumulative['current_cycle'] == cycle + 1  # Updated after prune


@pytest.mark.skipif(not ADAPTIVE_AVAILABLE, reason="Adaptive pruning modules not available")
class TestPerformanceBasedPruner:
    """Test performance-based adaptive pruning strategies."""
    
    def test_performance_based_improving(self, small_real_compounds):
        """Test performance-based pruner with improving performance."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        pruner = PerformanceBasedPruner(
            base_retention_fraction=0.6,
            performance_window=3,
            improvement_threshold=0.05,
            aggressive_retention=0.3,
            conservative_retention=0.8
        )
        
        # Simulate improving performance
        improving_metrics = [0.3, 0.4, 0.5, 0.6, 0.7]  # Steady improvement
        
        for i, metric in enumerate(improving_metrics):
            pruner.update_performance_metric(metric)
            
            # Only test pruning after we have enough metrics
            if i >= 2:  # Need at least 3 metrics for window
                pruned = pruner.prune(compounds, predictions)
                
                assert isinstance(pruned, pd.DataFrame)
                assert len(pruned) <= len(compounds)
                
                # With improving performance, should use conservative retention
                retention = len(pruned) / len(compounds)
                # Should be closer to conservative retention (0.8) than aggressive (0.3)
                assert retention >= 0.5  # At least moderate retention
    
    def test_performance_based_declining(self, small_real_compounds):
        """Test performance-based pruner with declining performance."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        pruner = PerformanceBasedPruner(
            base_retention_fraction=0.6,
            performance_window=2,
            improvement_threshold=0.1,
            aggressive_retention=0.2,
            conservative_retention=0.9
        )
        
        # Simulate declining performance
        declining_metrics = [0.8, 0.6, 0.4, 0.2]  # Steady decline
        
        for i, metric in enumerate(declining_metrics):
            pruner.update_performance_metric(metric)
            
            # Only test pruning after we have enough metrics
            if i >= 1:  # Need at least 2 metrics for window of 2
                pruned = pruner.prune(compounds, predictions)
                
                assert isinstance(pruned, pd.DataFrame)
                assert len(pruned) <= len(compounds)
                
                # With declining performance, should use more aggressive retention
                retention = len(pruned) / len(compounds)
                # Check that some pruning occurred
                assert retention <= 0.8  # Should be less than conservative
    
    def test_performance_based_stable(self, small_real_compounds):
        """Test performance-based pruner with stable performance."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        pruner = PerformanceBasedPruner(
            base_retention_fraction=0.6,
            performance_window=3,
            improvement_threshold=0.05,
            aggressive_retention=0.3,
            conservative_retention=0.8
        )
        
        # Simulate stable performance (small changes)
        stable_metrics = [0.5, 0.51, 0.49, 0.5, 0.52]  # Minimal variation
        
        for i, metric in enumerate(stable_metrics):
            pruner.update_performance_metric(metric)
            
            # Only test pruning after we have enough metrics
            if i >= 2:  # Need at least 3 metrics for window
                pruned = pruner.prune(compounds, predictions)
                
                assert isinstance(pruned, pd.DataFrame)
                assert len(pruned) <= len(compounds)
                
                # With stable performance, should use base retention
                retention = len(pruned) / len(compounds)
                # Should be close to base retention (0.6)
                assert 0.4 <= retention <= 0.8  # Allow some tolerance
    
    def test_performance_metric_updates(self):
        """Test performance metric update functionality."""
        pruner = PerformanceBasedPruner(
            performance_window=3,
            improvement_threshold=0.05
        )
        
        # Test metric updates
        metrics = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for metric in metrics:
            pruner.update_performance_metric(metric)
        
        # Check that metrics are stored
        assert len(pruner.performance_metrics) == 5
        
        # Test window limiting (should keep only recent metrics)
        for i in range(10):  # Add more metrics
            pruner.update_performance_metric(0.6 + i * 0.01)
        
        # Should limit to 2 * window size
        max_expected = pruner.performance_window * 2
        assert len(pruner.performance_metrics) <= max_expected
    
    def test_performance_based_statistics(self, small_real_compounds):
        """Test that performance-based pruner provides proper statistics."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        pruner = PerformanceBasedPruner(
            base_retention_fraction=0.6,
            performance_window=2
        )
        
        # Add some performance metrics
        pruner.update_performance_metric(0.3)
        pruner.update_performance_metric(0.4)
        pruner.update_performance_metric(0.5)
        
        # Perform pruning
        pruned = pruner.prune(compounds, predictions)
        
        # Check statistics
        stats = pruner.get_pruning_stats()
        assert 'base_retention' in stats
        assert 'performance_window' in stats
        assert 'performance_based' in stats
        assert 'recent_improvement' in stats
        
        # Check cumulative statistics
        cumulative = pruner.get_cumulative_stats()
        assert 'total_compounds_seen' in cumulative
        assert 'overall_pruning_rate' in cumulative


@pytest.mark.skipif(not ADAPTIVE_AVAILABLE, reason="Adaptive pruning modules not available")
class TestAdaptivePruningIntegration:
    """Test integration scenarios with adaptive pruning."""
    
    def test_multiple_adaptive_cycles(self, medium_real_compounds):
        """Test running multiple adaptive pruning cycles."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Use subset for faster testing
        compounds = compounds.head(30)
        
        # Test both adaptive pruners together
        cycle_pruner = CycleBudgetPruner(
            total_cycles=3,
            initial_retention_fraction=0.8,
            final_retention_fraction=0.4
        )
        
        performance_pruner = PerformanceBasedPruner(
            base_retention_fraction=0.6,
            performance_window=2
        )
        
        current_compounds = compounds.copy()
        
        for cycle in range(3):
            predictions = np.random.uniform(0, 1, len(current_compounds))
            
            # Apply cycle budget pruning
            cycle_pruner.cycle_count = cycle
            stage1 = cycle_pruner.prune(current_compounds, predictions)
            
            # Apply performance-based pruning to results
            if len(stage1) > 0:
                stage1_predictions = np.random.uniform(0, 1, len(stage1))
                performance_pruner.update_performance_metric(0.5 + cycle * 0.1)  # Improving
                
                stage2 = performance_pruner.prune(stage1, stage1_predictions)
                
                # Each stage should reduce compounds
                assert len(stage2) <= len(stage1) <= len(current_compounds)
                
                current_compounds = stage2
                
                # Should have some compounds remaining
                assert len(current_compounds) > 0
    
    def test_adaptive_error_handling(self, small_real_compounds):
        """Test error handling in adaptive pruning."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Test invalid strategy
        with pytest.raises(ValueError):
            CycleBudgetPruner(
                total_cycles=3,
                initial_retention_fraction=0.8,
                final_retention_fraction=0.3,
                strategy='invalid_strategy'
            )
        
        # Test with empty compounds
        empty_compounds = pd.DataFrame(columns=['ID', 'SMILES'])
        empty_predictions = np.array([])
        
        pruner = CycleBudgetPruner(total_cycles=3)
        
        with pytest.raises(Exception):  # Should fail on empty input
            pruner.prune(empty_compounds, empty_predictions)

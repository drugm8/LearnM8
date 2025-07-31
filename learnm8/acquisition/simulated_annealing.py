"""Simulated annealing acquisition function for the LearnM8 framework.

This module implements a simulated annealing-based acquisition strategy that 
balances exploration and exploitation through a temperature-based probabilistic
selection process. The algorithm starts with high temperature allowing random
exploration and gradually cools down to become more greedy/exploitative.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, TYPE_CHECKING

from .base import AcquisitionFunction

if TYPE_CHECKING:
    from ..core.data_manager import DataManager

logger = logging.getLogger(__name__)


class SimulatedAnnealingAcquisition(AcquisitionFunction):
    """Simulated annealing acquisition function for compound selection.
    
    This acquisition strategy uses simulated annealing to balance exploration 
    and exploitation in compound selection. The algorithm:
    
    1. Starts with high temperature allowing random exploration
    2. Gradually cools down following a cooling schedule  
    3. Uses Metropolis criterion to accept/reject compounds
    4. Returns the best compounds found during the annealing process
    
    The energy function is based on prediction values, where higher predictions
    correspond to lower energy (for maximization problems).
    """
    
    def __init__(self, 
                 data_manager: Optional['DataManager'] = None,
                 initial_temp: float = 1.0,
                 final_temp: float = 0.01,
                 max_iterations: int = 1000,
                 cooling_schedule: str = 'exponential',
                 score_direction: str = 'higher',
                 random_state: int = 42,
                 **kwargs):
        """Initialize simulated annealing acquisition function.
        
        Args:
            data_manager: Optional DataManager for feature extraction (not used by SimulatedAnnealingAcquisition)
            initial_temp: Starting temperature for annealing process
            final_temp: Final temperature for annealing process
            max_iterations: Maximum number of annealing iterations
            cooling_schedule: Cooling schedule ('exponential' or 'linear')
            score_direction: Direction of score optimization ('higher' or 'lower')
            random_state: Random seed for reproducible selection
            **kwargs: Additional parameters for compatibility
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(data_manager=data_manager, **kwargs)
        # Validate parameters
        if initial_temp <= 0:
            raise ValueError("initial_temp must be positive")
        if final_temp <= 0:
            raise ValueError("final_temp must be positive")
        if final_temp >= initial_temp:
            raise ValueError("final_temp must be less than initial_temp")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if cooling_schedule not in ['exponential', 'linear']:
            raise ValueError(f"cooling_schedule must be 'exponential' or 'linear', got '{cooling_schedule}'")
        if score_direction not in ['higher', 'lower']:
            raise ValueError(f"score_direction must be 'higher' or 'lower', got '{score_direction}'")
        
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.max_iterations = max_iterations
        self.cooling_schedule = cooling_schedule
        self.score_direction = score_direction
        self.random_state = random_state
        
        # Setup random number generator
        self._rng = np.random.RandomState(random_state)
        
        # Keep track of maximization vs minimization
        self.maximize = score_direction == 'higher'
    
    def _calculate_energy(self, prediction: float, uncertainty: Optional[float] = None) -> float:
        """Calculate energy for simulated annealing.
        
        Args:
            prediction: Model prediction value
            uncertainty: Model uncertainty (optional, not used in basic version)
            
        Returns:
            Energy value (lower is better for annealing)
        """
        if self.maximize:
            # For maximization: higher predictions = lower energy
            return -prediction
        else:
            # For minimization: lower predictions = lower energy
            return prediction
    
    def _get_temperature(self, iteration: int) -> float:
        """Calculate current temperature based on cooling schedule.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Current temperature value
        """
        # Calculate progress (0 to 1)
        progress = iteration / self.max_iterations
        
        if self.cooling_schedule == 'exponential':
            # Exponential cooling: T(t) = T_initial * (T_final/T_initial)^progress
            ratio = self.final_temp / self.initial_temp
            return self.initial_temp * (ratio ** progress)
        elif self.cooling_schedule == 'linear':
            # Linear cooling: T(t) = T_initial * (1 - progress) + T_final * progress
            return self.initial_temp * (1 - progress) + self.final_temp * progress
        else:
            # Should not reach here due to validation in __init__
            raise ValueError(f"Unknown cooling schedule: {self.cooling_schedule}")
    
    def _metropolis_accept(self, current_energy: float, candidate_energy: float, temperature: float) -> bool:
        """Determine whether to accept a candidate based on Metropolis criterion.
        
        Args:
            current_energy: Energy of current compound
            candidate_energy: Energy of candidate compound
            temperature: Current temperature
            
        Returns:
            True if candidate should be accepted, False otherwise
        """
        # Always accept if candidate is better (lower energy)
        if candidate_energy <= current_energy:
            return True
        
        # Accept worse candidates with probability exp(-ΔE/T)
        if temperature <= 0:
            return False
        
        delta_energy = candidate_energy - current_energy
        acceptance_prob = np.exp(-delta_energy / temperature)
        
        return bool(self._rng.random() < acceptance_prob)
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select compounds using simulated annealing.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction' columns
            n_select: Number of compounds to select
            
        Returns:
            DataFrame subset with selected compounds including 'acquisition_score' column
            
        Raises:
            ValueError: If required columns are missing or n_select is invalid
        """
        # Validate input
        self.validate_input(compounds, n_select)
        
        # Handle edge case where n_select >= available compounds
        actual_n_select = min(n_select, len(compounds))
        
        if actual_n_select == len(compounds):
            # Return all compounds with energy-based scores
            selected = compounds.copy()
            energies = [self._calculate_energy(pred) for pred in compounds['prediction']]
            selected['acquisition_score'] = -np.array(energies)  # Convert back to scores
            return selected
        
        # Initialize annealing process
        visited_indices = []
        visited_energies = []
        
        # Start with a random compound
        current_idx = self._rng.randint(len(compounds))
        current_energy = self._calculate_energy(compounds.iloc[current_idx]['prediction'])
        
        # Track the best compound found so far
        best_idx = current_idx
        best_energy = current_energy
        
        # Main annealing loop
        for iteration in range(self.max_iterations):
            # Calculate current temperature
            temperature = self._get_temperature(iteration)
            
            # Propose a new compound (random selection for simplicity)
            candidate_idx = self._rng.randint(len(compounds))
            candidate_energy = self._calculate_energy(compounds.iloc[candidate_idx]['prediction'])
            
            # Decide whether to accept the candidate
            if self._metropolis_accept(current_energy, candidate_energy, temperature):
                current_idx = candidate_idx
                current_energy = candidate_energy
                
                # Update best compound if this is better
                if candidate_energy < best_energy:
                    best_idx = candidate_idx
                    best_energy = candidate_energy
            
            # Record this compound in our history
            visited_indices.append(current_idx)
            visited_energies.append(current_energy)
        
        # Select the best n_select compounds from our annealing history
        # Get unique compounds and their best energies
        unique_compounds = {}
        for idx, energy in zip(visited_indices, visited_energies):
            if idx not in unique_compounds or energy < unique_compounds[idx]:
                unique_compounds[idx] = energy
        
        # Sort by energy (best first) and select top n_select
        sorted_compounds = sorted(unique_compounds.items(), key=lambda x: x[1])
        selected_indices = [idx for idx, _ in sorted_compounds[:actual_n_select]]
        
        # If we don't have enough unique compounds, add more from the pool
        if len(selected_indices) < actual_n_select:
            # Add compounds not yet selected, sorted by energy
            remaining_indices = set(range(len(compounds))) - set(selected_indices)
            remaining_with_energy = [(idx, self._calculate_energy(compounds.iloc[idx]['prediction'])) 
                                   for idx in remaining_indices]
            remaining_sorted = sorted(remaining_with_energy, key=lambda x: x[1])
            
            # Add best remaining compounds
            needed = actual_n_select - len(selected_indices)
            selected_indices.extend([idx for idx, _ in remaining_sorted[:needed]])
        
        # Create result DataFrame
        selected = compounds.iloc[selected_indices].copy()
        
        # Add acquisition scores (negative energy for intuitive scoring)
        acquisition_scores = [-self._calculate_energy(pred) for pred in selected['prediction']]
        selected['acquisition_score'] = acquisition_scores
        
        logger.debug(f"SimulatedAnnealingAcquisition selected {len(selected)} compounds "
                    f"using {self.cooling_schedule} cooling schedule "
                    f"(temp: {self.initial_temp:.3f} → {self.final_temp:.3f})")
        
        return selected
    
    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        return f"SimulatedAnnealing({self.cooling_schedule}_{self.score_direction})"
    
    def requires_uncertainty(self) -> bool:
        """Return True if this acquisition function requires uncertainty estimates."""
        return False  # Basic version doesn't use uncertainty
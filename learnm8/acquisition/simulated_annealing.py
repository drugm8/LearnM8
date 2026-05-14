"""Simulated annealing acquisition function for the LearnM8 framework.

This module implements a simulated annealing-based acquisition strategy that
balances exploration and exploitation through a temperature-based probabilistic
selection process. The algorithm starts with high temperature allowing random
exploration and gradually cools down to become more greedy/exploitative.

Supports three neighbor generation strategies:
- 'random': Uniform random candidate from entire pool (default, Metropolis
  filter — not a true local-move neighbourhood).
- 'score_band': Local moves to compounds within +/- band_width ranks of the
  current compound in predicted-score order. True local-move SA with O(1) moves
  after one O(n log n) sort — no feature extraction, scales to 100M+ pools.
- 'knn_features': Local moves to k-nearest neighbours in feature space (true
  chemical-space SA). Requires featurizer_obj + cache_dir. NOTE: this builds a
  NearestNeighbors index over the *entire* pool on every select() call, so it
  is only viable for small pools — prefer 'score_band' or 'random' at scale.

A future 'ann_features' strategy (approximate-NN index, e.g. HNSW/faiss, queried
lazily for visited points only) would give true chemical-space locality at 10M+
scale; it is intentionally not implemented here to avoid a new dependency.
"""

import logging

import numpy as np
import polars as pl

from .base import AcquisitionFunction

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

    Neighbor strategies:
    - 'random': Uniform random candidate from entire pool (Metropolis filter,
      default). Not a true local-move neighbourhood.
    - 'score_band': Candidate drawn from compounds within +/- band_width ranks
      of the current compound in predicted-score order. True local-move SA;
      one O(n log n) sort then O(1) per move; scales to 100M+ pools.
    - 'knn_features': Local moves to k-nearest neighbours in feature space
      (chemical-space SA). Requires featurizer_obj + cache_dir; uses cosine
      distance. Builds a NearestNeighbors index over the WHOLE pool on every
      select() call — small-pool use only; prefer 'score_band' at scale.
    """

    _VALID_NEIGHBOR_STRATEGIES = ('random', 'score_band', 'knn_features')

    def __init__(self,
                 initial_temp: float = 1.0,
                 final_temp: float = 0.01,
                 max_iterations: int = 1000,
                 cooling_schedule: str = 'exponential',
                 score_direction: str = 'higher',
                 random_state: int = 42,
                 neighbor_strategy: str = 'random',
                 n_neighbors: int = 10,
                 band_width: int = 50,
                 **kwargs):
        """Initialize simulated annealing acquisition function.

        Args:
            initial_temp: Starting temperature for annealing process
            final_temp: Final temperature for annealing process
            max_iterations: Maximum number of annealing iterations
            cooling_schedule: Cooling schedule ('exponential' or 'linear')
            score_direction: Direction of score optimization ('higher' or 'lower')
            random_state: Random seed for reproducible selection
            neighbor_strategy: Candidate generation strategy
                ('random', 'score_band', or 'knn_features')
            n_neighbors: Number of nearest neighbors for the 'knn_features' strategy
            band_width: Rank-window half-width for the 'score_band' strategy —
                a candidate is drawn from compounds within +/- band_width ranks
                of the current compound in predicted-score order.
            **kwargs: Additional parameters (featurizer_obj, cache_dir, etc.)

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(score_direction=score_direction, **kwargs)
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
        if neighbor_strategy not in self._VALID_NEIGHBOR_STRATEGIES:
            raise ValueError(
                f"neighbor_strategy must be one of "
                f"{self._VALID_NEIGHBOR_STRATEGIES}, got '{neighbor_strategy}'"
            )
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if band_width < 1:
            raise ValueError("band_width must be >= 1")

        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.max_iterations = max_iterations
        self.cooling_schedule = cooling_schedule
        self.score_direction = score_direction
        self.random_state = random_state
        self.neighbor_strategy = neighbor_strategy
        self.n_neighbors = n_neighbors
        self.band_width = band_width

        self._rng = np.random.default_rng(random_state)
        self.maximize = score_direction == 'higher'

        self._featurizer_obj = kwargs.pop('featurizer_obj', None)
        self._cache_dir = kwargs.pop('cache_dir', None)

    def _calculate_energy(self, prediction: float, uncertainty: float | None = None) -> float:
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

    def _build_neighbor_map(self, compounds: pl.DataFrame) -> list[np.ndarray] | None:
        """Build kNN neighbor map from feature space.

        Extracts features from SMILES and builds a sklearn NearestNeighbors
        index using cosine distance. Returns a list of neighbor index arrays.

        Args:
            compounds: DataFrame with 'SMILES' column

        Returns:
            List where element i is an array of neighbor indices for compound i,
            or None if featurizer_obj is not available.
        """
        if self._featurizer_obj is None:
            return None

        from sklearn.neighbors import NearestNeighbors

        from learnm8.features.extraction import extract_features

        smiles_list = compounds.get_column('SMILES').to_list()
        features = extract_features(
            smiles_list,
            self._featurizer_obj,
            cache_dir=self._cache_dir,
        )

        k = min(self.n_neighbors + 1, features.shape[0])
        nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
        nn.fit(features)
        neighbor_indices = nn.kneighbors(features, return_distance=False)

        return [row[row != i] for i, row in enumerate(neighbor_indices)]

    @staticmethod
    def _build_score_band_index(
        predictions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the rank index for the 'score_band' neighbour strategy.

        One O(n log n) sort of the prediction vector. Returns
        ``(idx_at_rank, rank_of_idx)`` where ``idx_at_rank[r]`` is the compound
        index at rank ``r`` (ascending prediction) and ``rank_of_idx[i]`` is the
        rank of compound ``i``. Together these give O(1) local moves in
        predicted-score space without any feature extraction.
        """
        idx_at_rank = np.argsort(predictions, kind='stable')
        rank_of_idx = np.empty_like(idx_at_rank)
        rank_of_idx[idx_at_rank] = np.arange(idx_at_rank.size)
        return idx_at_rank, rank_of_idx

    def _score_band_candidate(
        self,
        current_idx: int,
        idx_at_rank: np.ndarray,
        rank_of_idx: np.ndarray,
    ) -> int:
        """Draw a candidate within +/- band_width ranks of ``current_idx``."""
        n = idx_at_rank.size
        r = int(rank_of_idx[current_idx])
        lo = max(0, r - self.band_width)
        hi = min(n - 1, r + self.band_width)
        if hi <= lo:
            return current_idx
        # Draw a rank in [lo, hi] excluding r itself.
        candidate_rank = int(self._rng.integers(lo, hi))
        if candidate_rank >= r:
            candidate_rank += 1
        return int(idx_at_rank[candidate_rank])

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """Select compounds using simulated annealing.

        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction' columns
            n_select: Number of compounds to select

        Returns:
            DataFrame subset with selected compounds including 'acquisition_score' column

        Raises:
            ValueError: If required columns are missing or n_select is invalid
        """
        self.validate_input(compounds, n_select)

        actual_n_select = min(n_select, len(compounds))

        if actual_n_select == len(compounds):
            selected = compounds.clone()
            energies = np.array([self._calculate_energy(pred)
                                for pred in compounds.get_column('prediction').to_numpy()])
            selected = selected.with_columns(
                pl.Series('acquisition_score', -energies)
            )
            return selected

        all_ids = compounds.get_column('ID').to_numpy()
        predictions = compounds.get_column('prediction').to_numpy()

        # Resolve the effective strategy locally — never mutate self, so a
        # reused instance is not silently downgraded across select() calls.
        effective_strategy = self.neighbor_strategy

        neighbor_map = None
        score_band_index: tuple[np.ndarray, np.ndarray] | None = None
        if effective_strategy == 'knn_features':
            neighbor_map = self._build_neighbor_map(compounds)
            if neighbor_map is None:
                logger.warning(
                    "neighbor_strategy='knn_features' requires featurizer_obj; "
                    "falling back to random candidate generation"
                )
                effective_strategy = 'random'
        elif effective_strategy == 'score_band':
            score_band_index = self._build_score_band_index(predictions)

        visited_indices = []
        visited_energies = []

        current_idx = int(self._rng.integers(len(compounds)))
        current_energy = self._calculate_energy(predictions[current_idx])

        best_energy = current_energy

        for iteration in range(self.max_iterations):
            temperature = self._get_temperature(iteration)

            if neighbor_map is not None:
                nbrs = neighbor_map[current_idx]
                if len(nbrs) > 0:
                    candidate_idx = int(self._rng.choice(nbrs))
                else:
                    candidate_idx = current_idx
            elif score_band_index is not None:
                candidate_idx = self._score_band_candidate(
                    current_idx, score_band_index[0], score_band_index[1]
                )
            else:
                candidate_idx = int(self._rng.integers(len(compounds)))

            candidate_energy = self._calculate_energy(predictions[candidate_idx])

            if self._metropolis_accept(current_energy, candidate_energy, temperature):
                current_idx = candidate_idx
                current_energy = candidate_energy

                if candidate_energy < best_energy:
                    best_energy = candidate_energy

            visited_indices.append(current_idx)
            visited_energies.append(current_energy)

        unique_compounds = {}
        for idx, energy in zip(visited_indices, visited_energies, strict=True):
            if idx not in unique_compounds or energy < unique_compounds[idx]:
                unique_compounds[idx] = energy

        sorted_compounds = sorted(unique_compounds.items(), key=lambda x: x[1])
        selected_indices = [idx for idx, _ in sorted_compounds[:actual_n_select]]

        if len(selected_indices) < actual_n_select:
            remaining_indices = set(range(len(compounds))) - set(selected_indices)
            remaining_with_energy = [(idx, self._calculate_energy(predictions[idx]))
                                    for idx in remaining_indices]
            remaining_sorted = sorted(remaining_with_energy, key=lambda x: x[1])

            needed = actual_n_select - len(selected_indices)
            selected_indices.extend([idx for idx, _ in remaining_sorted[:needed]])

        selected_ids = all_ids[selected_indices]

        selected = compounds.filter(pl.col('ID').is_in(selected_ids.tolist()))

        acquisition_scores = {}
        for idx in selected_indices:
            acquisition_scores[all_ids[idx]] = -self._calculate_energy(predictions[idx])

        selected = selected.with_columns(
            pl.col('ID').replace_strict(acquisition_scores).alias('acquisition_score')
        )

        logger.debug(
            f"SimulatedAnnealingAcquisition selected {len(selected)} compounds "
            f"using {self.cooling_schedule} cooling schedule "
            f"(neighbor_strategy={effective_strategy}, "
            f"temp: {self.initial_temp:.3f} → {self.final_temp:.3f})"
        )

        return selected

    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        suffix = f"_{self.neighbor_strategy}" if self.neighbor_strategy != 'random' else ''
        return f"SimulatedAnnealing({self.cooling_schedule}_{self.score_direction}{suffix})"

    def requires_uncertainty(self) -> bool:
        """Return True if this acquisition function requires uncertainty estimates."""
        return False  # Basic version doesn't use uncertainty

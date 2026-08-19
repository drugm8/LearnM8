"""Simulated annealing acquisition function for the LearnM8 framework.

This module implements a simulated annealing-based acquisition strategy that
balances exploration and exploitation through a temperature-based probabilistic
selection process. The algorithm starts with high temperature allowing random
exploration and gradually cools down to become more greedy/exploitative.

The temperature schedule and step budget are DERIVED from each cycle's
predictions rather than fixed, because energy is ``-prediction`` and therefore
carries the target's units: a hard-coded temperature is only meaningful for a
target whose energy scale happens to match it. See
:class:`SimulatedAnnealingAcquisition` for the derivation and
:func:`_calibrate_temperature` for the calibration itself.

Scaling: the walk runs ``R`` independent chains of length ``L``, vectorized
over chains in numpy, so the Python-level loop is ``L`` iterations regardless
of ``n_select`` or pool size. ``supports_streaming()`` remains False —
annealing is a path-dependent walk over the full pool and cannot be expressed
as the pointwise ``score_chunk`` contract, so SA runs the legacy cycle path.

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
import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from .base import AcquisitionFunction

logger = logging.getLogger(__name__)

# --- Annealing schedule derivation constants (feature 027) ------------------
# The energy function is -prediction (or +prediction), so temperature carries
# the target's units. Every constant below is therefore either a dimensionless
# acceptance ratio or a step count — never a temperature.

SA_TARGET_UPHILL_ACCEPTANCE = 0.8  # Standard chi_0 (Ben-Ameur 2004; Kirkpatrick 1983)
SA_FINAL_UPHILL_ACCEPTANCE = 0.01  # Effectively-frozen chain at end of anneal
SA_CALIBRATION_SAMPLES = 1000  # Cheap; +/-0.01 on a mean acceptance estimate
SA_CALIBRATION_TOL = 0.01  # Fixed-point convergence on chi
SA_CALIBRATION_MAX_ITERS = 50  # Fixed-point iteration cap
SA_CHAIN_LENGTH = 2000  # Cooling-grid resolution; bounds the Python loop
SA_PILOT_STEPS = 300  # Pilot walk length; measures chi-bar for the step budget
SA_PILOT_CHAINS = 16  # Pilot chains; 300x16 = 4800 samples of the acceptance rate
SA_BUDGET_HEADROOM = 1.25  # Head-room for non-uniform revisiting in the tail
SA_MAX_TOTAL_STEPS = 20_000_000  # ~160 MB int64 visit matrix ceiling
SA_BACKFILL_WARN = 0.10  # Decision D3: WARNING above this backfill share
SA_BACKFILL_ERROR = 0.50  # Decision D3: AcquisitionError above this share

# With no uphill move in the neighbourhood every proposal is accepted whatever
# the temperature, so the value is arbitrary — it only has to be finite and
# positive so the cooling grid and the acceptance rule stay well-defined.
_DEGENERATE_TEMP = 1.0

# Keeps log(chi) finite when the fixed point transiently lands on chi = 0 or 1.
_CHI_EPS = 1e-12


def _calibrate_temperature(dE_plus: np.ndarray, chi_target: float) -> float:
    """Solve for the temperature whose REALIZED mean uphill acceptance is ``chi_target``.

    Ben-Ameur (2004) fixed point. The closed form ``T = E[dE] / -ln(chi)``
    seeds the iteration only: ``exp`` is convex, so by Jensen's inequality
    ``E[exp(-dE/T)] > exp(-E[dE]/T)`` and the closed form systematically
    overshoots ``chi_target`` by an amount set by the spread of ``dE``. On a
    right-skewed target that gap is too large to meet the +/-0.05 acceptance
    tolerance the feature is specified against (spec REQ-3, C2).

    Args:
        dE_plus: Strictly positive uphill energy gaps sampled from the
            proposal neighbourhood actually in use.
        chi_target: Desired mean uphill acceptance ratio, in ``(0, 1)``.

    Returns:
        Temperature in the units of the energy function. Scale-equivariant:
        ``_calibrate_temperature(c * dE, chi) == c * _calibrate_temperature(dE, chi)``.
    """
    if dE_plus.size == 0:
        # REQ-5: every proposal is accepted; skip the fixed point entirely.
        logger.debug(
            "Annealing calibration found no uphill move in the sampled "
            "neighbourhood (all predictions equal); using degenerate "
            "temperature %.3f and accepting every proposal.",
            _DEGENERATE_TEMP,
        )
        return _DEGENERATE_TEMP

    log_target = math.log(chi_target)
    temperature = float(dE_plus.mean() / -log_target)  # closed form SEEDS ONLY

    for _ in range(SA_CALIBRATION_MAX_ITERS):
        chi = float(np.exp(-dE_plus / temperature).mean())
        if abs(chi - chi_target) < SA_CALIBRATION_TOL:
            break
        # chi and chi_target both lie in (0, 1), so both logs are negative and
        # the ratio is positive: too much acceptance lowers T, too little
        # raises it.
        chi = min(max(chi, _CHI_EPS), 1.0 - _CHI_EPS)
        updated = temperature * (math.log(chi) / log_target)
        if not math.isfinite(updated) or updated <= 0.0:
            break  # keep the last well-defined temperature
        temperature = updated

    return temperature


def _cooling_grid(
    initial_temp: float,
    final_temp: float,
    chain_length: int,
    cooling_schedule: str,
) -> np.ndarray:
    """Return the temperature at each of ``chain_length`` steps.

    The same grid the chains will walk, so the acceptance estimate that sizes
    the step budget is taken against the schedule actually in use (REQ-4).
    """
    # linspace so the last step lands exactly on final_temp, matching the
    # iteration/max_iterations convention of the scalar _get_temperature.
    progress = np.linspace(0.0, 1.0, chain_length)
    if cooling_schedule == 'linear':
        return initial_temp * (1.0 - progress) + final_temp * progress
    return initial_temp * ((final_temp / initial_temp) ** progress)


def _derive_step_budget(
    n_select: int,
    pool_size: int,
    mean_acceptance: float,
) -> int:
    """Derive the total Metropolis step budget. No hard-coded multiplier (REQ-4).

    Two effects set the budget, and both come from measurement:

    1. Only an accepted move relocates the chain, so ``n_accepted ~
       total_steps * chi_bar``. ``chi_bar`` is MEASURED on a pilot walk
       (:meth:`SimulatedAnnealingAcquisition._pilot_acceptance_rate`) rather
       than estimated from uniformly-drawn proposal pairs. The distinction is
       not academic: a chain descends into the low-energy tail, where a
       ``'random'`` proposal is almost always a large uphill jump, so the
       uniform-origin estimate overstates acceptance ~3x and under-budgets
       the walk by the same factor.
    2. A random walk revisits compounds it has already seen, so collecting
       ``U`` *unique* compounds from a pool of ``N`` needs
       ``A = ln(1 - U/N) / ln(1 - 1/N)`` accepted moves, not ``U``. Ignoring
       this under-budgets precisely at high pool exhaustion, where residual
       greedy backfill would then silently paper over the shortfall.

    Args:
        n_select: Number of unique compounds required, ``U``.
        pool_size: Size of the selection pool, ``N``.
        mean_acceptance: Measured overall acceptance rate, ``chi_bar``.

    Returns:
        Total step budget across all chains, clamped to ``SA_MAX_TOTAL_STEPS``.
    """
    mean_acceptance = max(mean_acceptance, _CHI_EPS)

    # Expected-unique-visits inversion. Guard U < N so the log stays finite;
    # select() already short-circuits the n_select >= pool_size case.
    unique_target = min(n_select, pool_size - 1) if pool_size > 1 else 1
    if pool_size > 1 and unique_target >= 1:
        accepted_needed = math.log1p(-unique_target / pool_size) / math.log1p(
            -1.0 / pool_size
        )
    else:
        accepted_needed = float(unique_target)

    required = accepted_needed / mean_acceptance * SA_BUDGET_HEADROOM
    total_steps = max(1, math.ceil(required))

    if total_steps > SA_MAX_TOTAL_STEPS:
        logger.warning(
            "Derived annealing budget %d steps exceeds SA_MAX_TOTAL_STEPS "
            "(%d); clamping. n_select=%d on a pool of %d needs ~%.0f accepted "
            "moves at a mean acceptance of %.3f. The batch may be completed by "
            "greedy backfill — check the acquisition_backfill_fraction metric.",
            total_steps,
            SA_MAX_TOTAL_STEPS,
            n_select,
            pool_size,
            accepted_needed,
            mean_acceptance,
        )
        total_steps = SA_MAX_TOTAL_STEPS

    return total_steps


def _pad_neighbor_map(neighbor_map: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a ragged kNN neighbour map into a gatherable ``(padded, lengths)`` pair.

    Built once per ``select()`` so the per-step proposal is a single fancy-index
    gather rather than a Python loop over chains.

    Args:
        neighbor_map: Element ``i`` is the neighbour index array of compound ``i``.

    Returns:
        ``(padded, lengths)`` where ``padded[i, :lengths[i]]`` are the
        neighbours of ``i``. A compound with no neighbours gets itself at
        column 0 and length 1, matching the scalar path's stay-put fallback.
    """
    raw_lengths = np.fromiter(
        (nbrs.size for nbrs in neighbor_map), dtype=np.int64, count=len(neighbor_map)
    )
    k_max = max(int(raw_lengths.max(initial=0)), 1)
    padded = np.zeros((len(neighbor_map), k_max), dtype=np.int64)
    for i, nbrs in enumerate(neighbor_map):
        if nbrs.size:
            padded[i, : nbrs.size] = nbrs
        else:
            padded[i, 0] = i
    return padded, np.maximum(raw_lengths, 1)


@dataclass(frozen=True)
class AnnealingSchedule:
    """Derived annealing schedule for one ``select()`` call.

    The single return value of the derivation step, so calibration is testable
    without running a walk. Temperatures are in the units of the energy
    function (i.e. the target's units), so they are NOT comparable across
    cycles — read schedule diagnostics as ratios (spec C3).

    Attributes:
        initial_temp: T_0, calibrated so mean uphill acceptance equals
            ``SA_TARGET_UPHILL_ACCEPTANCE``, or the caller's override.
        final_temp: T_final, calibrated against ``SA_FINAL_UPHILL_ACCEPTANCE``,
            or the caller's override.
        n_chains: R, the number of independent chains advanced in lockstep.
        chain_length: L, the number of Python-level iterations. Bounded by
            ``SA_CHAIN_LENGTH`` regardless of ``n_select`` or pool size.
        mean_acceptance: chi-bar, the estimated overall acceptance probability
            averaged over the cooling grid. Used to derive the step budget.
        calibration_samples: Number of uphill deltas that survived sampling.
        degenerate: True when no uphill move existed in the sampled
            neighbourhood (all predictions equal), so every proposal is
            accepted and fixed-point iteration was skipped.
    """

    initial_temp: float
    final_temp: float
    n_chains: int
    chain_length: int
    mean_acceptance: float
    calibration_samples: int
    degenerate: bool


class SimulatedAnnealingAcquisition(AcquisitionFunction):
    """Simulated annealing acquisition function for compound selection.

    This acquisition strategy uses simulated annealing to balance exploration
    and exploitation in compound selection. The algorithm:

    1. Derives the whole schedule from this cycle's predictions (see below)
    2. Starts ``n_chains`` independent chains at independent random positions
    3. Cools all chains together along the configured cooling schedule,
       accepting or rejecting each proposal by the Metropolis criterion
    4. Returns the ``n_select`` LOWEST-ENERGY DISTINCT STATES VISITED across
       all chains, ordered best-first by ``acquisition_score``

    If the walk visits fewer than ``n_select`` distinct compounds, the batch is
    completed by greedy backfill and the share is reported on
    ``last_backfill_fraction`` — above ``SA_BACKFILL_WARN`` that logs a
    WARNING, and above ``SA_BACKFILL_ERROR`` it raises. Silent degradation to
    greedy is not possible.

    The energy function is based on prediction values, where higher predictions
    correspond to lower energy (for maximization problems). Energy therefore
    carries the target's units, and so does temperature.

    Schedule derivation (feature 027):
    ``initial_temp``, ``final_temp`` and ``max_iterations`` each default to
    ``None``, meaning "derive from the data at ``select()`` time":

    - ``initial_temp`` / ``final_temp`` are solved by Ben-Ameur fixed-point
      iteration so that the mean uphill acceptance equals
      ``SA_TARGET_UPHILL_ACCEPTANCE`` / ``SA_FINAL_UPHILL_ACCEPTANCE``, using
      energy gaps sampled through the proposal neighbourhood actually in use.
    - ``max_iterations`` is derived from ``n_select``, pool size and the
      measured acceptance rate.

    Supplying a number for any of the three uses it verbatim and disables
    derivation for that parameter only. Note that an explicit
    ``max_iterations`` is the TOTAL step budget across all chains
    (``n_chains * chain_length <= max_iterations``), not a per-chain length.

    Derivation is repeated every ``select()`` call, so absolute temperatures
    are not comparable across cycles — read schedule diagnostics as ratios.

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
                 initial_temp: float | None = None,
                 final_temp: float | None = None,
                 max_iterations: int | None = None,
                 cooling_schedule: str = 'exponential',
                 score_direction: str = 'higher',
                 random_state: int = 42,
                 neighbor_strategy: str = 'random',
                 n_neighbors: int = 10,
                 band_width: int = 50,
                 **kwargs):
        """Initialize simulated annealing acquisition function.

        Args:
            initial_temp: Starting temperature. ``None`` (default) derives it
                per cycle by Ben-Ameur calibration against
                ``SA_TARGET_UPHILL_ACCEPTANCE``. A number is used verbatim and
                disables derivation for this parameter only. Temperature is in
                the units of the target, so a fixed value is only meaningful
                if you know that scale.
            final_temp: End temperature. ``None`` (default) derives it against
                ``SA_FINAL_UPHILL_ACCEPTANCE``; a number is used verbatim.
            max_iterations: TOTAL Metropolis step budget across all chains,
                not per-chain length. ``None`` (default) derives it from
                ``n_select``, pool size and the measured acceptance rate.
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
        # REQ-19: validate only what the caller supplied numerically. None
        # means "derive at select() time" and cannot be range-checked here.
        if initial_temp is not None and initial_temp <= 0:
            raise ValueError("initial_temp must be positive")
        if final_temp is not None and final_temp <= 0:
            raise ValueError("final_temp must be positive")
        if (
            initial_temp is not None
            and final_temp is not None
            and final_temp >= initial_temp
        ):
            raise ValueError("final_temp must be less than initial_temp")
        if max_iterations is not None and max_iterations <= 0:
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

        # Share of the last batch completed by greedy backfill rather than by
        # the walk. None until select() has run (REQ-11).
        self.last_backfill_fraction: float | None = None

        self._featurizer_obj = kwargs.pop('featurizer_obj', None)
        self._cache_dir = kwargs.pop('cache_dir', None)

    def _energies(self, predictions: np.ndarray) -> np.ndarray:
        """Vectorized energy function — the single implementation of the sign rule.

        Energy carries the target's units, which is exactly why the temperature
        schedule must be calibrated against sampled energy gaps rather than
        assumed (feature 027).

        Args:
            predictions: Model prediction values

        Returns:
            Energy values (lower is better for annealing)
        """
        # For maximization: higher predictions = lower energy. For
        # minimization: the prediction value is the energy.
        return -predictions if self.maximize else predictions

    def _calculate_energy(self, prediction: float, uncertainty: float | None = None) -> float:
        """Calculate energy for a single prediction.

        Thin scalar wrapper over :meth:`_energies` so the sign rule has one
        implementation rather than two.

        Args:
            prediction: Model prediction value
            uncertainty: Model uncertainty (optional, not used in basic version)

        Returns:
            Energy value (lower is better for annealing)
        """
        return float(self._energies(np.asarray(prediction, dtype=np.float64)))

    def _sample_deltas(
        self,
        predictions: np.ndarray,
        *,
        neighbor_pad: tuple[np.ndarray, np.ndarray] | None = None,
        score_band_index: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Sample two-sided energy gaps from the proposal distribution in use.

        Draws ``SA_CALIBRATION_SAMPLES`` origins and generates each partner
        through :meth:`_propose` — literally the same call :meth:`select` will
        make — so the calibrated temperature matches the neighbourhood actually
        being explored. Using the global prediction standard deviation instead
        would mis-scale ``'score_band'`` by orders of magnitude (REQ-2, C2).

        Args:
            predictions: Prediction vector for the whole pool
            neighbor_pad: ``(padded, lengths)``, for ``'knn_features'``
            score_band_index: ``(idx_at_rank, rank_of_idx)``, for ``'score_band'``

        Returns:
            ``E(partner) - E(origin)`` for every sampled pair, both signs.
            Temperature calibration uses the positive half.
        """
        n_pool = predictions.shape[0]
        origins = self._rng.integers(n_pool, size=SA_CALIBRATION_SAMPLES)
        partners = self._propose(
            origins,
            n_pool,
            neighbor_pad=neighbor_pad,
            score_band_index=score_band_index,
        )
        energies = self._energies(predictions)
        deltas: np.ndarray = energies[partners] - energies[origins]
        return deltas

    def _sample_uphill_deltas(
        self,
        predictions: np.ndarray,
        *,
        neighbor_pad: tuple[np.ndarray, np.ndarray] | None = None,
        score_band_index: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """The strictly positive half of :meth:`_sample_deltas`.

        Returns:
            Uphill gaps only. May be empty when the sampled neighbourhood is
            flat, which is the degenerate case of REQ-5.
        """
        deltas = self._sample_deltas(
            predictions,
            neighbor_pad=neighbor_pad,
            score_band_index=score_band_index,
        )
        return deltas[deltas > 0]

    def _pilot_acceptance_rate(
        self,
        energies: np.ndarray,
        initial_temp: float,
        final_temp: float,
        *,
        neighbor_pad: tuple[np.ndarray, np.ndarray] | None = None,
        score_band_index: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> float:
        """Measure chi-bar on a short pilot walk across the full cooling grid.

        The quantity that sizes the step budget is the acceptance rate of a
        chain that has *descended*, not of one sampled uniformly. Under
        ``'random'`` proposals the two differ by ~3x, because once the chain
        reaches the low-energy tail every uniform draw is a large uphill jump.
        Measuring beats estimating here for the same reason it does for
        temperature: the value depends on the data, not on a constant.

        Args:
            energies: Energy of every compound in the pool
            initial_temp: T_0 of the pilot's cooling grid
            final_temp: T_final of the pilot's cooling grid
            neighbor_pad: ``(padded, lengths)``, for ``'knn_features'``
            score_band_index: ``(idx_at_rank, rank_of_idx)``, for ``'score_band'``

        Returns:
            Fraction of proposals accepted over ``SA_PILOT_STEPS`` steps of
            ``SA_PILOT_CHAINS`` chains.
        """
        n_pool = energies.size
        temperatures = _cooling_grid(
            initial_temp, final_temp, SA_PILOT_STEPS, self.cooling_schedule
        )
        current = self._rng.integers(n_pool, size=SA_PILOT_CHAINS)
        n_accepted = 0

        for step in range(SA_PILOT_STEPS):
            candidate = self._propose(
                current,
                n_pool,
                neighbor_pad=neighbor_pad,
                score_band_index=score_band_index,
            )
            accepted = self._acceptance_mask(
                energies[candidate] - energies[current], float(temperatures[step])
            )
            n_accepted += int(accepted.sum())
            current = np.where(accepted, candidate, current)

        return n_accepted / (SA_PILOT_STEPS * SA_PILOT_CHAINS)

    def _derive_schedule(
        self,
        predictions: np.ndarray,
        *,
        n_select: int,
        neighbor_pad: tuple[np.ndarray, np.ndarray] | None = None,
        score_band_index: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> AnnealingSchedule:
        """Derive the whole annealing schedule from this cycle's predictions.

        Each of ``initial_temp``, ``final_temp`` and ``max_iterations`` is used
        verbatim when the caller supplied a number, and derived when it is
        ``None`` (REQ-1). Derivation is recomputed on every call: it costs no
        cross-cycle state, and it is the only cadence that holds acceptance at
        the target ratio as the training set concentrates on the tail across
        cycles (spec C3, REQ-6).

        Args:
            predictions: Prediction vector for the whole pool
            n_select: Number of unique compounds the walk must supply
            neighbor_pad: ``(padded, lengths)``, for ``'knn_features'``
            score_band_index: ``(idx_at_rank, rank_of_idx)``, for ``'score_band'``

        Returns:
            The derived :class:`AnnealingSchedule`.
        """
        deltas = self._sample_deltas(
            predictions,
            neighbor_pad=neighbor_pad,
            score_band_index=score_band_index,
        )
        uphill = deltas[deltas > 0]
        degenerate = uphill.size == 0

        initial_temp = (
            self.initial_temp
            if self.initial_temp is not None
            else _calibrate_temperature(uphill, SA_TARGET_UPHILL_ACCEPTANCE)
        )
        final_temp = (
            self.final_temp
            if self.final_temp is not None
            else _calibrate_temperature(uphill, SA_FINAL_UPHILL_ACCEPTANCE)
        )

        if final_temp > initial_temp:
            # Reachable only by overriding exactly one of the two temperatures.
            logger.warning(
                "Annealing schedule is inverted (initial_temp=%.4g < "
                "final_temp=%.4g): the chain will HEAT rather than cool. This "
                "happens when one temperature is supplied numerically and the "
                "other is derived from data on a different scale. Supply both "
                "or neither.",
                initial_temp,
                final_temp,
            )

        # chi-bar is MEASURED on a pilot walk, not estimated from uniform
        # proposal pairs — see _derive_step_budget for why the two differ.
        mean_acceptance = self._pilot_acceptance_rate(
            self._energies(predictions),
            initial_temp,
            final_temp,
            neighbor_pad=neighbor_pad,
            score_band_index=score_band_index,
        )

        if self.max_iterations is not None:
            total_steps = self.max_iterations
        else:
            total_steps = _derive_step_budget(
                n_select=n_select,
                pool_size=predictions.shape[0],
                mean_acceptance=mean_acceptance,
            )

        # REQ-7: the Python loop is L iterations, bounded by SA_CHAIN_LENGTH
        # regardless of n_select or pool size. R chains advance per iteration.
        chain_length = max(1, min(SA_CHAIN_LENGTH, total_steps))
        if self.max_iterations is not None:
            # REQ-7a: an explicit max_iterations is a hard cap on TOTAL work
            # across chains, so round down and keep R*L <= max_iterations.
            n_chains = max(1, total_steps // chain_length)
        else:
            # A derived budget is a minimum requirement, so round up.
            n_chains = max(1, math.ceil(total_steps / chain_length))

        return AnnealingSchedule(
            initial_temp=float(initial_temp),
            final_temp=float(final_temp),
            n_chains=int(n_chains),
            chain_length=int(chain_length),
            mean_acceptance=mean_acceptance,
            calibration_samples=int(uphill.size),
            degenerate=degenerate,
        )

    def _get_temperature(self, iteration: int) -> float:
        """Scalar cooling curve for explicitly-parameterised instances.

        The walk itself uses the precomputed :func:`_cooling_grid` over the
        derived :class:`AnnealingSchedule`; this scalar form is retained for
        callers that supplied all three parameters numerically.

        Args:
            iteration: Current iteration number

        Returns:
            Current temperature value

        Raises:
            ValueError: If any of the three schedule parameters is ``None``,
                i.e. is derived per-cycle and has no instance-level value.
        """
        if (
            self.initial_temp is None
            or self.final_temp is None
            or self.max_iterations is None
        ):
            raise ValueError(
                "_get_temperature requires initial_temp, final_temp and "
                "max_iterations to be supplied numerically; this instance "
                "derives them per cycle. Read the temperatures from the "
                "AnnealingSchedule returned by _derive_schedule instead."
            )

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

    def _acceptance_mask(self, delta_energy: np.ndarray, temperature: float) -> np.ndarray:
        """Vectorized Metropolis rule — the single implementation (REQ-18).

        Draws one uniform per entry so that R chains can be advanced in one
        numpy operation, which is what keeps the Python loop length constant
        in ``n_select`` and pool size.

        Args:
            delta_energy: ``E(candidate) - E(current)`` per chain. Non-positive
                entries are downhill (or flat) and are accepted unconditionally.
            temperature: Current temperature, in the units of the energy.

        Returns:
            Boolean acceptance mask with the same shape as ``delta_energy``.
        """
        delta_energy = np.asarray(delta_energy, dtype=np.float64)
        downhill = delta_energy <= 0.0

        if temperature <= 0.0:
            # A frozen chain accepts nothing uphill.
            return downhill

        # exp(-dE/T) for uphill entries; the exponent is clipped at 0 so
        # downhill entries cannot overflow exp() at low temperature.
        probabilities = np.exp(np.minimum(0.0, -delta_energy / temperature))
        accepted: np.ndarray = downhill | (
            self._rng.random(delta_energy.shape) < probabilities
        )
        return accepted

    def _metropolis_accept(self, current_energy: float, candidate_energy: float, temperature: float) -> bool:
        """Scalar Metropolis criterion — thin wrapper over :meth:`_acceptance_mask`.

        Args:
            current_energy: Energy of current compound
            candidate_energy: Energy of candidate compound
            temperature: Current temperature

        Returns:
            True if candidate should be accepted, False otherwise
        """
        delta = np.array([candidate_energy - current_energy], dtype=np.float64)
        return bool(self._acceptance_mask(delta, temperature)[0])

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

    def _propose(
        self,
        cur: np.ndarray,
        n_pool: int,
        *,
        neighbor_pad: tuple[np.ndarray, np.ndarray] | None = None,
        score_band_index: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Propose one candidate per chain, vectorized over chains.

        The single implementation of the proposal rule: calibration, the pilot
        walk and the chains all call this, so all three sample from an
        identical distribution by construction rather than by agreement
        between two copies of the logic.

        Args:
            cur: Current pool index of each chain, shape ``(R,)``.
            n_pool: Size of the selection pool.
            neighbor_pad: ``(padded, lengths)`` from :func:`_pad_neighbor_map`,
                for ``'knn_features'``.
            score_band_index: ``(idx_at_rank, rank_of_idx)``, for ``'score_band'``.

        Returns:
            Candidate pool indices, shape ``(R,)``, all in ``[0, n_pool)``.
        """
        if neighbor_pad is not None:
            padded, lengths = neighbor_pad
            columns = self._rng.integers(lengths[cur])
            neighbours: np.ndarray = padded[cur, columns]
            return neighbours

        if score_band_index is not None:
            idx_at_rank, rank_of_idx = score_band_index
            n = idx_at_rank.size
            r = rank_of_idx[cur]
            lo = np.maximum(0, r - self.band_width)
            hi = np.minimum(n - 1, r + self.band_width)
            # Draw a rank in [lo, hi-1], then skip over r to land in
            # [lo, hi] \ {r} — the scalar exclude-self offset, vectorized.
            has_window = hi > lo
            draw = self._rng.integers(lo, np.where(has_window, hi, lo + 1))
            draw = draw + (draw >= r)
            candidate_rank = np.where(has_window, draw, r)
            in_band: np.ndarray = idx_at_rank[candidate_rank]
            return in_band

        return self._rng.integers(n_pool, size=cur.size)

    def _run_chains(
        self,
        predictions: np.ndarray,
        schedule: AnnealingSchedule,
        *,
        neighbor_pad: tuple[np.ndarray, np.ndarray] | None = None,
        score_band_index: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, int, int]:
        """Run ``R`` independent chains of length ``L``, vectorized over chains.

        The Python-level loop is ``L`` iterations regardless of ``n_select`` or
        pool size (REQ-7); all ``R`` chains advance inside each iteration in
        numpy. Chains start at ``R`` independent uniform positions so they
        explore different basins (REQ-8) — which is also what makes the batch
        more diverse than a single walk of the same total length.

        Args:
            predictions: Prediction vector for the whole pool
            schedule: The derived schedule supplying ``R``, ``L`` and the
                temperature endpoints
            neighbor_pad: ``(padded, lengths)``, for ``'knn_features'``
            score_band_index: ``(idx_at_rank, rank_of_idx)``, for ``'score_band'``

        Returns:
            ``(visited_indices, n_accepted, n_proposed)``. ``visited_indices``
            holds the ``R`` starting positions plus every ACCEPTED landing
            position (REQ-16) — a rejected step does not re-log the position
            the chain was already sitting on.
        """
        energies = self._energies(predictions)
        n_pool = predictions.shape[0]
        n_chains = schedule.n_chains
        chain_length = schedule.chain_length

        temperatures = _cooling_grid(
            schedule.initial_temp,
            schedule.final_temp,
            chain_length,
            self.cooling_schedule,
        )
        current = self._rng.integers(n_pool, size=n_chains)  # REQ-8
        visits = np.full((chain_length + 1, n_chains), -1, dtype=np.int64)
        visits[0] = current
        n_accepted = 0

        for step in range(chain_length):
            candidate = self._propose(
                current,
                n_pool,
                neighbor_pad=neighbor_pad,
                score_band_index=score_band_index,
            )
            accepted = self._acceptance_mask(
                energies[candidate] - energies[current], float(temperatures[step])
            )
            current = np.where(accepted, candidate, current)
            # REQ-16: record a visit only on acceptance. The old code appended
            # unconditionally, logging the same index once per rejected step.
            visits[step + 1] = np.where(accepted, candidate, -1)
            n_accepted += int(accepted.sum())

        visited = visits.ravel()
        return visited[visited >= 0], n_accepted, n_chains * chain_length

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """Select compounds using simulated annealing.

        Derives the annealing schedule from ``compounds['prediction']``, runs
        ``n_chains`` vectorized chains, and returns the ``n_select``
        lowest-energy distinct states they visited.

        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction' columns
            n_select: Number of compounds to select

        Returns:
            DataFrame subset with selected compounds including an
            'acquisition_score' column, ordered BEST-FIRST by that score so a
            downstream ``.head(k)`` truncation drops the worst compounds.

        Raises:
            ValueError: If required columns are missing or n_select is invalid
        """
        self.validate_input(compounds, n_select)

        actual_n_select = min(n_select, len(compounds))

        if actual_n_select == len(compounds):
            # REQ-14: no annealing was attempted, so nothing was degraded.
            self.last_backfill_fraction = 0.0
            energies = self._energies(compounds.get_column('prediction').to_numpy())
            # Best-first here too, so select() has ONE ordering contract (REQ-15).
            order = np.argsort(energies, kind='stable')
            return compounds[order].with_columns(
                pl.Series('acquisition_score', -energies[order])
            )

        predictions = compounds.get_column('prediction').to_numpy()

        # Resolve the effective strategy locally — never mutate self, so a
        # reused instance is not silently downgraded across select() calls.
        effective_strategy = self.neighbor_strategy

        # Built once and shared by calibration, the pilot and the walk, so all
        # three sample from an identical proposal distribution (REQ-2).
        neighbor_pad: tuple[np.ndarray, np.ndarray] | None = None
        score_band_index: tuple[np.ndarray, np.ndarray] | None = None
        if effective_strategy == 'knn_features':
            neighbor_map = self._build_neighbor_map(compounds)
            if neighbor_map is None:
                logger.warning(
                    "neighbor_strategy='knn_features' requires featurizer_obj; "
                    "falling back to random candidate generation"
                )
                effective_strategy = 'random'
            else:
                neighbor_pad = _pad_neighbor_map(neighbor_map)
        elif effective_strategy == 'score_band':
            score_band_index = self._build_score_band_index(predictions)

        schedule = self._derive_schedule(
            predictions,
            n_select=actual_n_select,
            neighbor_pad=neighbor_pad,
            score_band_index=score_band_index,
        )
        visited, n_accepted, n_proposed = self._run_chains(
            predictions,
            schedule,
            neighbor_pad=neighbor_pad,
            score_band_index=score_band_index,
        )

        energies = self._energies(predictions)

        # Energy is a pure function of pool index, so np.unique IS the dedupe —
        # there is no per-index minimum to take.
        candidates = np.unique(visited)
        ranked = candidates[np.argsort(energies[candidates], kind='stable')]
        selected_indices = ranked[:actual_n_select]

        n_backfilled = actual_n_select - selected_indices.size
        if n_backfilled > 0:
            # The walk could not supply a full batch. Complete it greedily, but
            # record the share so the caller can see how much of the batch is
            # not actually an annealing result (REQ-11).
            unvisited = np.ones(len(compounds), dtype=bool)
            unvisited[selected_indices] = False
            remaining = np.flatnonzero(unvisited)
            best_remaining = remaining[
                np.argsort(energies[remaining], kind='stable')[:n_backfilled]
            ]
            selected_indices = np.concatenate([selected_indices, best_remaining])

        # REQ-15: return best-first by acquisition_score. Backfilled compounds
        # are not necessarily worse than walked ones, so the whole batch is
        # re-sorted. core/cycle.py truncates over-long batches with
        # .head(batch_size), which under pool order dropped arbitrary rather
        # than worst compounds.
        selected_indices = selected_indices[
            np.argsort(energies[selected_indices], kind='stable')
        ]

        self.last_backfill_fraction = float(n_backfilled) / float(actual_n_select)
        self._enforce_backfill_policy(actual_n_select, n_backfilled, schedule)

        selected: pl.DataFrame = compounds[selected_indices]
        selected = selected.with_columns(
            pl.Series('acquisition_score', -energies[selected_indices])
        )

        logger.debug(
            "SimulatedAnnealingAcquisition selected %d compounds using %s "
            "cooling (neighbor_strategy=%s, T %.4g -> %.4g, R=%d chains x L=%d "
            "steps, accepted %d/%d = %.3f, unique visits %d, backfill %.1f%%)",
            len(selected),
            self.cooling_schedule,
            effective_strategy,
            schedule.initial_temp,
            schedule.final_temp,
            schedule.n_chains,
            schedule.chain_length,
            n_accepted,
            n_proposed,
            n_accepted / max(n_proposed, 1),
            candidates.size,
            100.0 * self.last_backfill_fraction,
        )

        return selected

    def _enforce_backfill_policy(
        self, n_select: int, n_backfilled: int, schedule: AnnealingSchedule
    ) -> None:
        """Surface residual greedy backfill, loudly (REQ-12, REQ-13).

        Args:
            n_select: Size of the requested batch
            n_backfilled: How many of those came from greedy backfill
            schedule: The derived schedule, reported so the cause is diagnosable

        Raises:
            AcquisitionError: If the backfilled share exceeds
                ``SA_BACKFILL_ERROR``. At that point the batch is more greedy
                than annealed and returning it would misattribute the result.
        """
        fraction = n_backfilled / n_select
        if fraction <= SA_BACKFILL_WARN:
            return

        detail = (
            f"{n_backfilled} of {n_select} compounds ({fraction:.1%}) came from "
            f"greedy backfill, not from the annealing walk. Derived schedule: "
            f"T {schedule.initial_temp:.4g} -> {schedule.final_temp:.4g}, "
            f"{schedule.n_chains} chains x {schedule.chain_length} steps, "
            f"mean acceptance {schedule.mean_acceptance:.3f}"
            f"{', degenerate landscape' if schedule.degenerate else ''}"
        )

        if fraction > SA_BACKFILL_ERROR:
            from learnm8.exceptions import AcquisitionError

            raise AcquisitionError(
                # WHAT
                f"Simulated annealing backfill exceeded "
                f"{SA_BACKFILL_ERROR:.0%}: {detail}. "
                # WHY
                "WHY: the annealing walk visited too few distinct compounds to "
                "fill the batch, so the result would be mostly greedy top-k "
                "mislabelled as simulated annealing. "
                # HOW
                "HOW TO FIX: raise max_iterations (it is the TOTAL step budget "
                "across chains), or leave it None so the budget is derived; "
                "use neighbor_strategy='score_band' for local moves that keep "
                "acceptance high at scale; or reduce n_select."
            )

        logger.warning(
            "Simulated annealing backfill above %.0f%%: %s. The batch is "
            "partly greedy top-k. Raise max_iterations or leave it None so "
            "the budget is derived, or use neighbor_strategy='score_band'.",
            SA_BACKFILL_WARN * 100,
            detail,
        )

    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        suffix = f"_{self.neighbor_strategy}" if self.neighbor_strategy != 'random' else ''
        return f"SimulatedAnnealing({self.cooling_schedule}_{self.score_direction}{suffix})"

    def requires_uncertainty(self) -> bool:
        """Return True if this acquisition function requires uncertainty estimates."""
        return False  # Basic version doesn't use uncertainty

"""Test configuration and fixture registration.

Fixtures are organized into domain-specific modules in tests/fixtures/:
- molecules.py: Molecular data (synthetic and real compounds)
- mocks.py: Mock learners and oracles for testing
- master_dataframe.py: Master DataFrame fixtures for active learning cycles
- performance.py: Performance sequences for pruning/adaptation tests
- utilities.py: Logging and helper utilities

All fixtures are automatically available to tests via pytest's fixture discovery.
"""

from typing import Final

import numpy as np
import pytest

# Register fixture plugins from fixtures/ directory
pytest_plugins = [
    "tests.fixtures.molecules",
    "tests.fixtures.mocks",
    "tests.fixtures.master_dataframe",
    "tests.fixtures.performance",
    "tests.fixtures.utilities",
    "tests.fixtures.features",
    "tests.fixtures.trained_models",
]


RNG_SEED: Final[int] = 19_2026


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic NumPy generator seeded with RNG_SEED for property tests."""
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def synthetic_mu_sigma(rng: np.random.Generator):
    """Factory yielding (mu, sigma) pairs of arbitrary length, including σ=0 corners.

    Returns a callable: f(n: int, include_zero_sigma: bool = True) -> (mu, sigma).
    """
    def _make(n: int, include_zero_sigma: bool = True):
        mu = rng.normal(loc=0.0, scale=5.0, size=n)
        sigma = np.abs(rng.normal(loc=1.0, scale=0.5, size=n))
        if include_zero_sigma and n >= 5:
            sigma[0] = 0.0
            sigma[1] = 0.0
            sigma[2] = 1e-300
            sigma[3] = 1e-12
        return mu, sigma

    return _make


SPEED_MARKERS = {"unit", "integration", "slow"}


def pytest_xdist_auto_num_workers(config):
    return 4


def pytest_collection_modifyitems(config, items):
    """Enforce that every test has a speed marker (unit/integration/slow)."""
    unmarked = []
    for item in items:
        marker_names = {m.name for m in item.iter_markers()}
        if not marker_names & SPEED_MARKERS:
            unmarked.append(item.nodeid)
    if unmarked:
        msg = (
            f"{len(unmarked)} test(s) missing required speed markers "
            f"(unit/integration/slow):\n"
            + "\n".join(f"  - {nid}" for nid in unmarked[:20])
            + (f"\n  ... and {len(unmarked) - 20} more" if len(unmarked) > 20 else "")
        )
        pytest.exit(msg, returncode=4)

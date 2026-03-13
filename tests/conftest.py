"""Test configuration and fixture registration.

Fixtures are organized into domain-specific modules in tests/fixtures/:
- molecules.py: Molecular data (synthetic and real compounds)
- mocks.py: Mock learners and oracles for testing
- master_dataframe.py: Master DataFrame fixtures for active learning cycles
- performance.py: Performance sequences for pruning/adaptation tests
- utilities.py: Logging and helper utilities

All fixtures are automatically available to tests via pytest's fixture discovery.
"""

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


import pytest

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

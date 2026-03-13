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
]


import warnings

REQUIRED_MARKERS = {"unit", "integration", "slow", "molecular"}


def pytest_configure(config):
    """Configure pytest markers and settings."""
    config.addinivalue_line(
        "markers", "unit: unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests"
    )
    config.addinivalue_line(
        "markers", "molecular: tests requiring RDKit"
    )
    config.addinivalue_line(
        "markers", "slow: slow running tests"
    )


def pytest_collection_modifyitems(config, items):
    """Warn about tests missing required markers."""
    unmarked = []
    for item in items:
        marker_names = {m.name for m in item.iter_markers()}
        if not marker_names & REQUIRED_MARKERS:
            unmarked.append(item.nodeid)
    if unmarked:
        warnings.warn(
            f"{len(unmarked)} test(s) missing required markers "
            f"(unit/integration/slow/molecular):\n"
            + "\n".join(f"  - {nid}" for nid in unmarked[:10])
            + (f"\n  ... and {len(unmarked) - 10} more" if len(unmarked) > 10 else ""),
            UserWarning,
            stacklevel=1,
        )

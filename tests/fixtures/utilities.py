"""Utility fixtures for LearnM8 tests.

Provides logging and helper utilities for test configuration.

Logging fixtures:
- caplog_info: Sets logging to INFO level
- caplog_debug: Sets logging to DEBUG level

Simple wrappers around pytest's caplog fixture.
"""

import logging
import pytest


@pytest.fixture
def caplog_info(caplog):
    """Capture INFO level logs from learnm8 logger."""
    # Enable propagation for test capture
    logger = logging.getLogger('learnm8')
    logger.propagate = True

    caplog.set_level(logging.INFO, logger='learnm8')
    return caplog


@pytest.fixture
def caplog_debug(caplog):
    """Capture DEBUG level logs from learnm8 logger."""
    # Enable propagation for test capture
    logger = logging.getLogger('learnm8')
    logger.propagate = True

    caplog.set_level(logging.DEBUG, logger='learnm8')
    return caplog

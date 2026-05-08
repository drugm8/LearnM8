"""Transient OSError retry path (T025, FR-010)."""

from __future__ import annotations

import errno
import logging

import pytest

from learnm8.features.cache import _read_with_one_retry


@pytest.mark.unit
def test_retry_succeeds_on_transient_eagain(caplog: pytest.LogCaptureFixture):
    calls = {'n': 0}

    def reader():
        calls['n'] += 1
        if calls['n'] == 1:
            raise OSError(errno.EAGAIN, 'temporary')
        return 'ok'

    with caplog.at_level(logging.WARNING):
        result = _read_with_one_retry(reader)

    assert result == 'ok'
    assert calls['n'] == 2
    warnings = [r for r in caplog.records if 'Transient OSError' in r.message]
    assert len(warnings) == 1


@pytest.mark.unit
def test_retry_succeeds_on_transient_eintr():
    calls = {'n': 0}

    def reader():
        calls['n'] += 1
        if calls['n'] == 1:
            raise OSError(errno.EINTR, 'interrupted')
        return 42

    assert _read_with_one_retry(reader) == 42
    assert calls['n'] == 2


@pytest.mark.unit
def test_retry_succeeds_on_transient_eio():
    calls = {'n': 0}

    def reader():
        calls['n'] += 1
        if calls['n'] == 1:
            raise OSError(errno.EIO, 'i/o error')
        return [1, 2, 3]

    assert _read_with_one_retry(reader) == [1, 2, 3]
    assert calls['n'] == 2


@pytest.mark.unit
def test_persistent_transient_error_propagates_after_one_retry():
    calls = {'n': 0}

    def reader():
        calls['n'] += 1
        raise OSError(errno.EAGAIN, 'still transient')

    with pytest.raises(OSError) as exc:
        _read_with_one_retry(reader)
    assert exc.value.errno == errno.EAGAIN
    assert calls['n'] == 2


@pytest.mark.unit
def test_non_transient_oserror_propagates_immediately():
    calls = {'n': 0}

    def reader():
        calls['n'] += 1
        raise OSError(errno.ENOENT, 'no such file')

    with pytest.raises(OSError) as exc:
        _read_with_one_retry(reader)
    assert exc.value.errno == errno.ENOENT
    assert calls['n'] == 1


@pytest.mark.unit
def test_non_oserror_propagates_immediately():
    def reader():
        raise ValueError('not an OSError')

    with pytest.raises(ValueError):
        _read_with_one_retry(reader)

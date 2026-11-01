"""configure_learnm8_logging: the log file follows the current output_dir."""

import logging

import pytest

from learnm8.utils.logging import configure_learnm8_logging


@pytest.fixture
def clean_learnm8_logger():
    logger = logging.getLogger('learnm8')
    saved_handlers, saved_level = list(logger.handlers), logger.level
    logger.handlers.clear()
    yield logger
    for h in logger.handlers:
        h.close()
    logger.handlers[:] = saved_handlers
    logger.setLevel(saved_level)


def _file_handlers(logger):
    return [h for h in logger.handlers if isinstance(h, logging.FileHandler)]


@pytest.mark.unit
def test_console_first_then_output_dir_writes_log_file(tmp_path, clean_learnm8_logger):
    """The CLI configures the console first; a later run must still get a log file."""
    configure_learnm8_logging(console_type='simple')
    configure_learnm8_logging(output_dir=tmp_path)
    logging.getLogger('learnm8.test').info('hello')
    assert 'hello' in (tmp_path / 'learnm8.log').read_text()
    assert len(_file_handlers(clean_learnm8_logger)) == 1


@pytest.mark.unit
def test_second_output_dir_moves_log_file(tmp_path, clean_learnm8_logger):
    first, second = tmp_path / 'a', tmp_path / 'b'
    first.mkdir()
    second.mkdir()
    configure_learnm8_logging(output_dir=first, console_type='none')
    configure_learnm8_logging(output_dir=second, console_type='none')
    logging.getLogger('learnm8.test').info('second run')
    assert 'second run' in (second / 'learnm8.log').read_text()
    assert 'second run' not in (first / 'learnm8.log').read_text()
    assert len(_file_handlers(clean_learnm8_logger)) == 1


@pytest.mark.unit
def test_same_output_dir_keeps_one_handler(tmp_path, clean_learnm8_logger):
    configure_learnm8_logging(output_dir=tmp_path, console_type='none')
    configure_learnm8_logging(output_dir=tmp_path, console_type='none')
    assert len(_file_handlers(clean_learnm8_logger)) == 1

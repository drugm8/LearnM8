"""Centralized logging configuration with Rich formatting support."""

import logging
import os
import sys
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler


def detect_jupyter_environment() -> bool:
    """
    Detect if code is running in a Jupyter notebook environment.

    Returns:
        bool: True if running in Jupyter, False otherwise
    """
    jupyter_indicators = [
        'IPY_PARENT',
        'JUPYTER_COLUMNS',
        'JPY_PARENT_PID',
        'JUPYTER_RUNTIME_DIR',
    ]

    for indicator in jupyter_indicators:
        if indicator in os.environ:
            return True

    try:
        import IPython
        ipy = IPython.get_ipython()
        if ipy is not None:
            if hasattr(ipy, 'kernel'):
                return True
            if ipy.__class__.__name__ == 'ZMQInteractiveShell':
                return True
    except ImportError:
        pass

    try:
        import ipykernel.kernelapp
        if hasattr(ipykernel.kernelapp, 'IPKernelApp') and ipykernel.kernelapp.IPKernelApp.initialized():
            return True
    except (ImportError, AttributeError):
        pass

    return False

def configure_learnm8_logging(
    output_dir: Path | None = None,
    level: str = "INFO",
    console_type: Literal['auto', 'rich', 'simple', 'none'] = 'auto',
    show_time: bool = False,
    force_reconfigure: bool = False
) -> logging.Logger:
    """
    Configure LearnM8 logging with file and console handlers.

    This is the ONLY function needed for all logging configuration.
    It's idempotent (safe to call multiple times) and follows Python
    logging best practices.

    Args:
        output_dir: Directory for log file. If None, no file logging.
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        console_type:
            - 'auto': Rich in terminal, simple in Jupyter (default)
            - 'rich': Force Rich handler
            - 'simple': Force StreamHandler
            - 'none': No console output
        show_time: Show timestamps in console
        force_reconfigure: Clear and reconfigure even if already set up

    Returns:
        Configured 'learnm8' logger

    Example:
        # In API
        configure_learnm8_logging(
            output_dir=Path('results'),
            level='INFO'
        )

        # In CLI
        configure_learnm8_logging(
            console_type='rich',
            level='INFO'
        )
    """
    logger = logging.getLogger('learnm8')

    if logger.handlers and not force_reconfigure:
        logger.debug("LearnM8 logging already configured")
        return logger

    logger.handlers.clear()

    logger.setLevel(getattr(logging, level.upper()))

    if output_dir:
        log_file = output_dir / 'learnm8.log'
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    if console_type != 'none':
        if console_type == 'auto':
            use_rich = not detect_jupyter_environment()
        elif console_type == 'rich':
            use_rich = True
        else:
            use_rich = False

        if use_rich:
            console = Console()
            console_handler = RichHandler(
                console=console,
                show_time=show_time,
                show_path=False,
                markup=True,
                rich_tracebacks=True
            )
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)

        logger.addHandler(console_handler)

    logger.propagate = False

    return logger


def get_logger(name: str = "learnm8") -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(name)


def log_warning(logger: logging.Logger, message: str) -> None:
    """Log warning message with appropriate styling."""
    logger.warning(f"[yellow]Warning:[/yellow] {message}")

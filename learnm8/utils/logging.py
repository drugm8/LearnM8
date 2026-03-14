"""Centralized logging configuration with Rich formatting support."""

import logging
import sys
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler


def is_running_in_jupyter() -> bool:
    """Detect if code is running in Jupyter notebook/lab."""
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return False
        return ipython.__class__.__name__ == 'ZMQInteractiveShell'
    except ImportError:
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
            use_rich = not is_running_in_jupyter()
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



def log_error_to_stderr(message: str) -> None:
    """Log error messages to stderr without Rich formatting."""
    logging.getLogger('learnm8').error(message)


def log_warning(logger: logging.Logger, message: str) -> None:
    """Log warning message with appropriate styling."""
    logger.warning(f"[yellow]Warning:[/yellow] {message}")


def log_success(logger: logging.Logger, message: str) -> None:
    """Log success message with appropriate styling."""
    logger.info(f"[bold green]✓[/bold green] {message}")


def log_file_operation(logger: logging.Logger, operation: str, path: str) -> None:
    """Log file operations (save, load, etc.)."""
    logger.info(f"{operation}: [dim]{path}[/dim]")


def setup_logging(
    level: str = "INFO",
    show_time: bool = True,
    show_path: bool = False,
    console: Console | None = None
) -> logging.Logger:
    """
    Backward compatibility wrapper for configure_learnm8_logging().

    DEPRECATED: Use configure_learnm8_logging() instead.
    """
    return configure_learnm8_logging(
        level=level,
        console_type='rich',
        show_time=show_time
    )


def setup_logging_for_environment(
    logger: logging.Logger,
    output_dir: Path | None = None,
    level: str = "INFO",
    show_time: bool = True
):
    """
    Backward compatibility wrapper for configure_learnm8_logging().

    DEPRECATED: Use configure_learnm8_logging() instead.

    Note: This function previously returned a list of handlers, but the new
    implementation does not support this. It now returns None for compatibility.
    """
    configure_learnm8_logging(
        output_dir=output_dir,
        level=level,
        console_type='auto',
        show_time=show_time
    )
    return []

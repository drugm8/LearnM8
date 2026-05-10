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


def get_console_config() -> dict:
    """
    Get appropriate Rich console configuration based on environment.

    Returns:
        dict: Configuration parameters for Rich Console
    """
    in_jupyter = detect_jupyter_environment()

    config = {
        'width': 100,
        'legacy_windows': False,
    }

    if in_jupyter:
        config.update({
            'force_jupyter': True,
            'force_terminal': False,
        })
    else:
        config.update({
            'force_terminal': True,
            'force_jupyter': False,
        })

    return config


def get_change_indicator_style() -> str:
    """
    Get appropriate change indicator style based on environment.

    Returns:
        str: 'emoji' for Jupyter environments, 'arrow' for terminal
    """
    return 'emoji' if detect_jupyter_environment() else 'arrow'


def format_change_indicator(
    diff: float,
    is_improvement: bool,
    style: str = 'auto',
    stagnation_threshold: float = 0.01
) -> tuple[str, str]:
    """
    Format change indicator based on environment and style preference.

    Args:
        diff: Numeric difference (positive or negative)
        is_improvement: Whether the change represents an improvement
        style: 'arrow', 'emoji', or 'auto' for automatic detection
        stagnation_threshold: Absolute threshold below which change is considered stagnant (default 1%)

    Returns:
        tuple: (symbol, color) for the change indicator
               - Green ↑/📈 for improvement
               - Red ↓/📉 for worsening
               - Yellow →/➡️ for stagnation (|diff| < threshold)
    """
    if style == 'auto':
        style = get_change_indicator_style()

    if abs(diff) < stagnation_threshold:
        if style == 'emoji':
            return "➡️", "yellow"
        else:
            return "→", "yellow"

    if style == 'emoji':
        if is_improvement:
            symbol = "📈" if diff > 0 else "📉"
            color = "green"
        else:
            symbol = "📉" if diff > 0 else "📈"
            color = "red"
    else:
        symbol = "↑" if diff > 0 else "↓"
        color = "green" if is_improvement else "red"

    return symbol, color


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

"""Centralized logging configuration with Rich formatting support."""

import logging
import sys
from pathlib import Path
from typing import Optional, List

from rich.console import Console
from rich.logging import RichHandler


def is_running_in_jupyter() -> bool:
    """Detect if code is running in Jupyter notebook/lab."""
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return False
        # Check for Jupyter notebook/lab environments
        return ipython.__class__.__name__ == 'ZMQInteractiveShell'
    except ImportError:
        return False


def setup_logging_for_environment(
    logger: logging.Logger,
    output_dir: Optional[Path] = None,
    level: str = "INFO",
    show_time: bool = True
) -> List[logging.Handler]:
    """
    Set up logging that works optimally in both CLI and Jupyter environments.
    
    Args:
        logger: Logger instance to configure
        output_dir: Directory for log files (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        show_time: Whether to show timestamps
        
    Returns:
        List of handlers added to the logger
    """
    handlers = []
    log_level = getattr(logging, level.upper())
    
    # Always add file handler if output directory provided
    if output_dir:
        log_file = output_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        handlers.append(file_handler)
    
    # Add appropriate console handler based on environment
    if is_running_in_jupyter():
        # Use simple StreamHandler for Jupyter notebooks
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(log_level)
        # Simpler format for notebook cells
        jupyter_formatter = logging.Formatter('%(message)s')
        stream_handler.setFormatter(jupyter_formatter)
        logger.addHandler(stream_handler)
        handlers.append(stream_handler)
    else:
        # Use Rich handler for CLI environments
        console = Console()
        rich_handler = RichHandler(
            console=console,
            show_time=show_time,
            show_path=False,
            markup=True,
            rich_tracebacks=True
        )
        rich_handler.setLevel(log_level)
        logger.addHandler(rich_handler)
        handlers.append(rich_handler)
    
    logger.setLevel(log_level)
    return handlers


def setup_logging(
    level: str = "INFO",
    show_time: bool = True,
    show_path: bool = False,
    console: Optional[Console] = None
) -> logging.Logger:
    """
    Set up centralized logging with Rich formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        show_time: Whether to show timestamps
        show_path: Whether to show file paths
        console: Custom console instance (optional)
        
    Returns:
        Configured logger instance
    """
    if console is None:
        console = Console()
    
    # Configure Rich handler
    rich_handler = RichHandler(
        console=console,
        show_time=show_time,
        show_path=show_path,
        markup=True,
        rich_tracebacks=True
    )
    
    # Set up logging configuration
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=[rich_handler]
    )
    
    # Create and return logger
    logger = logging.getLogger("learnm8")
    return logger


def get_logger(name: str = "learnm8") -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(name)



def log_error_to_stderr(message: str) -> None:
    """Log error messages to stderr without Rich formatting."""
    print(f"Error: {message}", file=sys.stderr)


def log_warning(logger: logging.Logger, message: str) -> None:
    """Log warning message with appropriate styling."""
    logger.warning(f"[yellow]Warning:[/yellow] {message}")


def log_success(logger: logging.Logger, message: str) -> None:
    """Log success message with appropriate styling."""
    logger.info(f"[bold green]✓[/bold green] {message}")


def log_file_operation(logger: logging.Logger, operation: str, path: str) -> None:
    """Log file operations (save, load, etc.)."""
    logger.info(f"{operation}: [dim]{path}[/dim]")
"""Centralized logging configuration with Rich formatting support."""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


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


def log_experiment_start(logger: logging.Logger, config: dict) -> None:
    """Log experiment initialization with key parameters."""
    logger.info(f"[bold blue]Starting experiment:[/bold blue] {config.get('target_column', 'unknown')}")
    logger.info(f"Compounds: {config.get('n_compounds', 'unknown')} | "
                f"Cycles: {config.get('n_cycles', 'unknown')} | "
                f"Strategy: {config.get('selection_strategy', 'unknown')}")


def log_cycle_start(logger: logging.Logger, cycle: int, total_cycles: int) -> None:
    """Log cycle start with progress indication."""
    logger.info(f"[yellow]Cycle {cycle + 1}/{total_cycles}[/yellow]")


def log_selection(logger: logging.Logger, count: int, strategy: str) -> None:
    """Log compound selection results."""
    logger.info(f"Selected [bold]{count}[/bold] compounds using [cyan]{strategy}[/cyan]")


def log_metrics(logger: logging.Logger, metrics: dict) -> None:
    """Log performance metrics in a compact format."""
    if 'rmse' in metrics:
        rmse = metrics['rmse']
        logger.info(f"RMSE: [bold]{rmse:.4f}[/bold]")
    
    if 'top_k_overlap' in metrics:
        overlap = metrics['top_k_overlap']
        logger.info(f"Top-K Overlap: [bold]{overlap:.2f}%[/bold]")
    
    if 'enrichment_factor' in metrics:
        ef = metrics['enrichment_factor']
        logger.info(f"Enrichment Factor: [bold]{ef:.2f}[/bold]")


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
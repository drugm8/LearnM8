"""Environment detection utilities for LearnM8."""

import logging
import os

logger = logging.getLogger(__name__)


def detect_jupyter_environment() -> bool:
    """
    Detect if code is running in a Jupyter notebook environment.
    
    Returns:
        bool: True if running in Jupyter, False otherwise
    """
    # Primary check: Jupyter environment variables
    jupyter_indicators = [
        'IPY_PARENT',           # IPython kernel process
        'JUPYTER_COLUMNS',      # Jupyter terminal settings
        'JPY_PARENT_PID',       # JupyterLab process ID
        'JUPYTER_RUNTIME_DIR',  # Jupyter runtime directory
    ]

    # Check environment variables first (most reliable)
    for indicator in jupyter_indicators:
        if indicator in os.environ:
            return True

    # Secondary check: IPython in notebook context
    try:
        import IPython
        ipy = IPython.get_ipython()
        if ipy is not None:
            # Check if it's specifically a notebook kernel
            if hasattr(ipy, 'kernel'):
                return True
            # Check for ZMQInteractiveShell (notebook/qtconsole)
            if ipy.__class__.__name__ == 'ZMQInteractiveShell':
                return True
    except ImportError:
        pass

    # Tertiary check: notebook-specific modules (less reliable)
    try:
        import ipykernel.kernelapp
        # Only return True if we can find a running kernel
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
        # Jupyter-optimized configuration
        config.update({
            'force_jupyter': True,
            'force_terminal': False,
        })
        logger.debug("Using Jupyter-optimized Rich console configuration")
    else:
        # Terminal configuration
        config.update({
            'force_terminal': True,
            'force_jupyter': False,
        })
        logger.debug("Using terminal-optimized Rich console configuration")

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

    # Check for stagnation first
    if abs(diff) < stagnation_threshold:
        if style == 'emoji':
            return "➡️", "yellow"
        else:
            return "→", "yellow"

    if style == 'emoji':
        # Jupyter-friendly emoji indicators
        if is_improvement:
            symbol = "📈" if diff > 0 else "📉"
            color = "green"
        else:
            symbol = "📉" if diff > 0 else "📈"
            color = "red"
    else:
        # Terminal arrow indicators (default)
        symbol = "↑" if diff > 0 else "↓"
        color = "green" if is_improvement else "red"

    return symbol, color

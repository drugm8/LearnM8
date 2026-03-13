import logging
import re

logger = logging.getLogger(__name__)


def validate_n_jobs(n_jobs: int) -> int:
    """Validate and normalize the n_jobs parameter for parallel processing.

    Args:
        n_jobs: Number of parallel jobs. -1 for all cores, positive integer
            for a specific number of workers.

    Returns:
        Validated n_jobs value (always -1 or a positive integer).

    Raises:
        TypeError: If n_jobs is not an integer.
        ValueError: If n_jobs is 0 or less than -1.
    """
    if not isinstance(n_jobs, int):
        raise TypeError(f"n_jobs must be an integer, got {type(n_jobs).__name__}")
    if n_jobs == 0:
        raise ValueError("n_jobs must be -1 (all cores) or a positive integer, got 0")
    if n_jobs < -1:
        raise ValueError(f"n_jobs must be -1 (all cores) or a positive integer, got {n_jobs}")
    return n_jobs


def validate_device(device: str) -> str:
    """Validate device specification and check hardware availability.

    Args:
        device: Device string. Valid values: 'auto', 'cpu', 'cuda',
            'cuda:N' (specific GPU index), 'mps'.

    Returns:
        Validated device string.

    Raises:
        TypeError: If device is not a string.
        ValueError: If device is unknown or the requested hardware
            is not available.
    """
    if not isinstance(device, str):
        raise TypeError(f"device must be a string, got {type(device).__name__}")

    valid_patterns = {'auto', 'cpu', 'cuda', 'mps'}
    cuda_n_pattern = re.compile(r'^cuda:\d+$')

    if device in valid_patterns:
        if device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA device requested but no GPU available. Use 'cpu' or 'auto'."
                )
        elif device == 'mps':
            import torch
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                raise ValueError(
                    "MPS device requested but not available. Use 'cpu' or 'auto'."
                )
        return device

    if cuda_n_pattern.match(device):
        import torch
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA device requested but no GPU available. Use 'cpu' or 'auto'."
            )
        device_index = int(device.split(':')[1])
        gpu_count = torch.cuda.device_count()
        if device_index >= gpu_count:
            raise ValueError(
                f"CUDA device {device_index} requested but only {gpu_count} "
                f"GPUs available (indices 0-{gpu_count - 1})"
            )
        return device

    raise ValueError(
        f"Unknown device '{device}'. Valid: 'auto', 'cpu', 'cuda', 'cuda:N', 'mps'"
    )


def parse_device_for_lightning(device: str) -> tuple:
    """Convert a device string to PyTorch Lightning accelerator/devices format.

    Args:
        device: Validated device string (e.g. 'auto', 'cpu', 'cuda', 'cuda:0', 'mps').

    Returns:
        Tuple of (accelerator, devices) suitable for PyTorch Lightning Trainer.

    Raises:
        ValueError: If device string is not recognized.
    """
    if device == 'auto':
        return ('auto', 'auto')
    elif device == 'cpu':
        return ('cpu', 1)
    elif device == 'cuda':
        return ('gpu', 1)
    elif device.startswith('cuda:'):
        device_index = int(device.split(':')[1])
        return ('gpu', [device_index])
    elif device == 'mps':
        return ('mps', 1)
    else:
        raise ValueError(
            f"Unknown device '{device}'. Valid: 'auto', 'cpu', 'cuda', 'cuda:N', 'mps'"
        )

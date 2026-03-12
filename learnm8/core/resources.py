import logging
import re

logger = logging.getLogger(__name__)


def validate_n_jobs(n_jobs: int) -> int:
    if not isinstance(n_jobs, int):
        raise TypeError(f"n_jobs must be an integer, got {type(n_jobs).__name__}")
    if n_jobs == 0:
        raise ValueError("n_jobs must be -1 (all cores) or a positive integer, got 0")
    if n_jobs < -1:
        raise ValueError(f"n_jobs must be -1 (all cores) or a positive integer, got {n_jobs}")
    return n_jobs


def validate_device(device: str) -> str:
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

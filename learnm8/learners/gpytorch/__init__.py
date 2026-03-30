"""GPyTorch-based Gaussian Process learners for LearnM8.

Provides GPU-accelerated GP regression with LOVE-enabled fast variance
computation for active learning over molecular fingerprints.
"""

import contextlib

with contextlib.suppress(ImportError):
    from .gpu_gp import GPyTorchGPLearner

with contextlib.suppress(ImportError):
    from .svgp import SVGPLearner

__all__ = ["GPyTorchGPLearner", "SVGPLearner"]

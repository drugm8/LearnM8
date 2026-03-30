from __future__ import annotations

import os

import pytest
import sklearn
import torch


@pytest.fixture(autouse=True)
def restore_sklearn_config():
    original = sklearn.get_config()
    yield
    sklearn.set_config(**original)


@pytest.fixture(scope="session", autouse=True)
def cuda_determinism():
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    yield

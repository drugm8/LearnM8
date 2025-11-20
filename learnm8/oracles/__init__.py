# learnm8/oracles/__init__.py
"""Oracle implementations for compound measurement."""

from .csv_oracle import CSVOracle
from .python_oracle import PythonOracle

__all__ = ['CSVOracle', 'PythonOracle']
"""Core active learning functionality - Modular Architecture (v1.0.0).

Modular architecture with clear separation of concerns:
- validation: Early compound validation
- initialization: Master DataFrame and initial sampling
- config: Cycle configuration
- cycle: Unified cycle execution
- persistence: CSV export
- dataframe_ops: Vectorized operations
"""

# Core interfaces (maintained for component creation)
from .config import CycleConfig, parse_cycle_schedule, parse_cycle_spec
from .cycle import execute_cycle
from .dataframe_ops import (
    add_predictions,
    batch_update,
    get_compounds_by_status,
    update_status,
)
from .initialization import initialize_master_dataframe_empty
from .interfaces import Learner, Oracle
from .persistence import save_results

# Modular core functions
from .validation import ValidationResult, validate_compound_pool

__all__ = [
    # Configuration
    'CycleConfig',
    'Learner',
    # Core interfaces
    'Oracle',
    'ValidationResult',
    # DataFrame operations
    'add_predictions',
    'batch_update',
    # Cycle execution
    'execute_cycle',
    'get_compounds_by_status',
    # Initialization
    'initialize_master_dataframe_empty',
    'parse_cycle_schedule',
    'parse_cycle_spec',
    # Persistence
    'save_results',
    'update_status',
    # Validation
    'validate_compound_pool',
]

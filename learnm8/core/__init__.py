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
from .interfaces import Oracle, Learner

# New modular core functions
try:
    from .validation import validate_compound_pool, ValidationResult
except ImportError:
    pass

try:
    from .initialization import initialize_master_dataframe, select_initial_sample
except ImportError:
    pass

try:
    from .config import CycleConfig, parse_cycle_schedule, parse_cycle_spec
except ImportError:
    pass

try:
    from .cycle import execute_cycle
except ImportError:
    pass

try:
    from .persistence import save_results
except ImportError:
    pass

try:
    from .dataframe_ops import add_predictions, update_status, get_compounds_by_status, batch_update
except ImportError:
    pass


# Build __all__ dynamically based on successful imports
__all__ = [
    # Core interfaces (always available)
    'Oracle', 'Learner',
]

# Add modular functions only if imports succeeded
if 'validate_compound_pool' in dir():
    __all__.extend(['validate_compound_pool', 'ValidationResult'])

if 'initialize_master_dataframe' in dir():
    __all__.extend(['initialize_master_dataframe', 'select_initial_sample'])

if 'CycleConfig' in dir():
    __all__.extend(['CycleConfig', 'parse_cycle_schedule', 'parse_cycle_spec'])

if 'execute_cycle' in dir():
    __all__.append('execute_cycle')

if 'save_results' in dir():
    __all__.append('save_results')

if 'add_predictions' in dir():
    __all__.extend(['add_predictions', 'update_status', 'get_compounds_by_status', 'batch_update'])
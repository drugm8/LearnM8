"""Python-based oracle that executes user-defined oracle functions."""

import importlib.util
import inspect
import logging
import polars as pl
from pathlib import Path
from collections.abc import Callable
from learnm8.core.interfaces import Oracle
from learnm8.exceptions import OracleError

logger = logging.getLogger(__name__)


class PythonOracle(Oracle):
    """Oracle that executes user-defined Python functions."""
    
    def __init__(self, module_path: str = None, oracle_path: str = None, function_name: str = None, **kwargs):
        """
        Initialize the Python oracle.
        
        Args:
            module_path: Path to Python file containing oracle function (new parameter name)
            oracle_path: Path to Python file (legacy parameter for backward compatibility)
            function_name: Name of oracle function (if None, auto-detect)
            **kwargs: Additional parameters for compatibility
        """
        # Handle both new and legacy parameter names
        if module_path is not None:
            self.oracle_path = Path(module_path)
        elif oracle_path is not None:
            self.oracle_path = Path(oracle_path)
        else:
            raise OracleError(
                "PythonOracle requires a path to the oracle Python file. "
                "Provide module_path='path/to/oracle.py' pointing to a Python file "
                "that defines an oracle function accepting a list of compound IDs."
            )

        if not self.oracle_path.exists():
            raise FileNotFoundError(
                f"Oracle Python file not found: {self.oracle_path}. "
                f"Verify the file path is correct and the file exists."
            )
        
        # Load the oracle function
        self.oracle_function = self._load_oracle_function(function_name)
        
    def _load_oracle_function(self, function_name: str = None) -> Callable:
        """Load oracle function from Python file."""
        # Load module from file
        spec = importlib.util.spec_from_file_location("oracle_module", self.oracle_path)
        oracle_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle_module)
        
        # Find oracle function
        if function_name:
            if not hasattr(oracle_module, function_name):
                available_funcs = [name for name, obj in inspect.getmembers(oracle_module, inspect.isfunction)
                                   if not name.startswith('_')]
                raise OracleError(
                    f"Function '{function_name}' not found in {self.oracle_path}. "
                    f"Available functions: {available_funcs or 'none'}. "
                    f"Check the function_name parameter or define a function with that name."
                )
            oracle_function = getattr(oracle_module, function_name)
        else:
            # Auto-detect oracle function
            functions = [obj for name, obj in inspect.getmembers(oracle_module, inspect.isfunction)
                        if not name.startswith('_')]

            if len(functions) == 0:
                raise OracleError(
                    f"No public functions found in {self.oracle_path}. "
                    f"Define an oracle function that accepts a list of compound IDs "
                    f"and returns a DataFrame with 'ID' and target columns."
                )
            elif len(functions) == 1:
                oracle_function = functions[0]
                logger.info("Using oracle function: %s", oracle_function.__name__)
            else:
                # Look for common oracle function names
                common_names = ['oracle', 'oracle_function', 'measure', 'evaluate']
                candidates = [f for f in functions if f.__name__ in common_names]

                if len(candidates) == 1:
                    oracle_function = candidates[0]
                    logger.info("Using oracle function: %s", oracle_function.__name__)
                else:
                    function_names = [f.__name__ for f in functions]
                    raise OracleError(
                        f"Multiple functions found in {self.oracle_path}: {function_names}. "
                        f"Specify which function to use with function_name parameter. "
                        f"Common names that are auto-detected: oracle, oracle_function, measure, evaluate."
                    )
        
        # Validate function signature
        sig = inspect.signature(oracle_function)
        if len(sig.parameters) != 1:
            raise OracleError(
                f"Oracle function '{oracle_function.__name__}' must take exactly 1 parameter "
                f"(compound_ids: List[str]), but has signature: {sig}. "
                f"The function should accept a list of compound ID strings and return a DataFrame."
            )
        
        return oracle_function
        
    def measure(self, compounds: pl.DataFrame, properties: list[str]) -> pl.DataFrame:
        """
        Execute oracle function to measure compound properties.

        Args:
            compounds: DataFrame with 'ID' column
            properties: List of property names (not used for Python oracles)

        Returns:
            DataFrame with 'ID' and measured property columns
        """
        # Extract compound IDs
        compound_ids = compounds.get_column('ID').to_list()

        # Call oracle function
        try:
            result = self.oracle_function(compound_ids)
        except OracleError:
            raise
        except (ValueError, RuntimeError, TypeError, OSError, KeyError) as e:
            raise OracleError(f"Oracle function failed: {e}") from e

        # Validate result - support both pandas and polars return values
        if isinstance(result, pl.DataFrame):
            pass  # Already Polars, no conversion needed
        elif hasattr(result, '__dataframe__'):  # pandas DataFrame
            # Convert pandas to Polars
            result = pl.from_pandas(result)
        else:
            raise OracleError(
                f"Oracle function must return a pandas or polars DataFrame, "
                f"got {type(result).__name__}. "
                f"Return a DataFrame with 'ID' column and target property columns."
            )

        if 'ID' not in result.columns:
            raise OracleError(
                f"Oracle function result must contain an 'ID' column. "
                f"Available columns: {list(result.columns)}. "
                f"Ensure your oracle function returns a DataFrame with 'ID' matching the input compound IDs."
            )

        # Check that all requested compounds have results
        result_ids = set(result.get_column('ID').to_list())
        request_ids = set(compound_ids)
        missing_ids = request_ids - result_ids

        if missing_ids:
            logger.warning("Oracle did not return results for %d compounds", len(missing_ids))

        return result
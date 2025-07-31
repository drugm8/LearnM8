"""Python-based oracle that executes user-defined oracle functions."""

import importlib.util
import inspect
import pandas as pd
from pathlib import Path
from typing import List, Callable
from learnm8.core.interfaces import Oracle


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
            raise ValueError("Either module_path or oracle_path must be provided")
        
        if not self.oracle_path.exists():
            raise FileNotFoundError(f"Oracle file not found: {self.oracle_path}")
        
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
                raise ValueError(f"Function '{function_name}' not found in {self.oracle_path}")
            oracle_function = getattr(oracle_module, function_name)
        else:
            # Auto-detect oracle function
            functions = [obj for name, obj in inspect.getmembers(oracle_module, inspect.isfunction)
                        if not name.startswith('_')]
            
            if len(functions) == 0:
                raise ValueError(f"No functions found in {self.oracle_path}")
            elif len(functions) == 1:
                oracle_function = functions[0]
                print(f"Using oracle function: {oracle_function.__name__}")
            else:
                # Look for common oracle function names
                common_names = ['oracle', 'oracle_function', 'measure', 'evaluate']
                candidates = [f for f in functions if f.__name__ in common_names]
                
                if len(candidates) == 1:
                    oracle_function = candidates[0]
                    print(f"Using oracle function: {oracle_function.__name__}")
                else:
                    function_names = [f.__name__ for f in functions]
                    raise ValueError(
                        f"Multiple functions found in {self.oracle_path}: {function_names}. "
                        f"Please specify function_name parameter."
                    )
        
        # Validate function signature
        sig = inspect.signature(oracle_function)
        if len(sig.parameters) != 1:
            raise ValueError(
                f"Oracle function must take exactly 1 parameter (compound_ids: List[str]). "
                f"Found: {sig}"
            )
        
        return oracle_function
        
    def measure(self, compounds: pd.DataFrame, properties: List[str]) -> pd.DataFrame:
        """
        Execute oracle function to measure compound properties.
        
        Args:
            compounds: DataFrame with 'ID' column
            properties: List of property names (not used for Python oracles)
            
        Returns:
            DataFrame with 'ID' and measured property columns
        """
        # Extract compound IDs
        compound_ids = compounds['ID'].tolist()
        
        # Call oracle function
        try:
            result = self.oracle_function(compound_ids)
        except Exception as e:
            raise RuntimeError(f"Oracle function failed: {e}")
        
        # Validate result
        if not isinstance(result, pd.DataFrame):
            raise ValueError(f"Oracle function must return pandas DataFrame, got {type(result)}")
        
        if 'ID' not in result.columns:
            raise ValueError("Oracle function result must contain 'ID' column")
        
        # Check that all requested compounds have results
        result_ids = set(result['ID'])
        request_ids = set(compound_ids)
        missing_ids = request_ids - result_ids
        
        if missing_ids:
            print(f"Warning: Oracle did not return results for {len(missing_ids)} compounds")
        
        return result
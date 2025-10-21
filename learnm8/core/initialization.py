"""Initialization functions for active learning experiments.

Provides clean initialization with:
- Actual target column names (not 'target_value')
- Vectorized operations for performance
- Flexible initial sampling (random or diverse)
- Compatibility with new architecture (Phases 1 and 2)

This module replaces the old initialize_master_dataframe() in data_structures.py
with a cleaner, more consistent implementation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Literal
import logging
from pathlib import Path
from .data_structures import STATUS_LABELED, STATUS_UNLABELED, VALID_STATUSES
from learnm8.features.extraction import extract_features

logger = logging.getLogger(__name__)


def initialize_master_dataframe(
    valid_compounds: pd.DataFrame,
    initial_labeled_ids: List[str],
    initial_measurements: pd.Series,
    target_col: str
) -> pd.DataFrame:
    """Create master DataFrame from validated compounds.

    This is the NEW version that replaces the old one in data_structures.py.

    Key differences from old version:
    - Takes valid_compounds (already validated) instead of raw compound_pool
    - Uses actual target_col name throughout (not 'target_value')
    - Marks initial compounds with labeled_cycle = -1 (not pd.NA)
    - Does NOT create 'last_prediction' or 'last_uncertainty' columns (removed in Phase 2)
    - Uses vectorized initialization (not iterative loop)

    Args:
        valid_compounds: DataFrame with 'ID' and 'SMILES' columns (already validated)
        initial_labeled_ids: List of compound IDs to mark as initially labeled
        initial_measurements: Series mapping compound ID to measured value
        target_col: Name of target column for measurements

    Returns:
        Master DataFrame with columns: ID, SMILES, status, labeled_cycle,
        selected_cycle, pruned_cycle, target_col

    Example:
        >>> master_df = initialize_master_dataframe(
        ...     valid_compounds=validated_df,
        ...     initial_labeled_ids=['compound_1', 'compound_2'],
        ...     initial_measurements=pd.Series({'compound_1': 0.5, 'compound_2': 0.8}),
        ...     target_col='Activity'
        ... )
    """
    if 'ID' not in valid_compounds.columns or 'SMILES' not in valid_compounds.columns:
        raise ValueError("valid_compounds must contain 'ID' and 'SMILES' columns")

    master_df = valid_compounds[['ID', 'SMILES']].copy()

    initial_set = set(initial_labeled_ids)
    master_df['status'] = pd.Categorical(
        [STATUS_LABELED if cid in initial_set else STATUS_UNLABELED for cid in master_df['ID']],
        categories=VALID_STATUSES
    )

    master_df['labeled_cycle'] = pd.Series(dtype='Int64')
    master_df['selected_cycle'] = pd.Series(dtype='Int64')
    master_df['pruned_cycle'] = pd.Series(dtype='Int64')
    master_df[target_col] = pd.Series(dtype='float64')

    mask = master_df['ID'].isin(initial_set)
    id_to_value = initial_measurements.to_dict()
    master_df.loc[mask, target_col] = master_df.loc[mask, 'ID'].map(id_to_value)
    master_df.loc[mask, 'labeled_cycle'] = -1
    master_df.loc[mask, 'selected_cycle'] = -1

    logger.info(f"Initialized master DataFrame with {len(master_df)} compounds, {len(initial_labeled_ids)} initially labeled")

    return master_df


def select_initial_sample(
    compounds_df: pd.DataFrame,
    n_initial: int,
    strategy: Literal['random', 'diverse'] = 'random',
    featurizer_type: str = 'morgan',
    cache_dir: Optional[Path] = None,
    random_state: int = 42
) -> List[str]:
    """Select initial training set using random or diverse sampling strategies.

    Args:
        compounds_df: DataFrame with 'ID' and 'SMILES' columns
        n_initial: Number of compounds to select for initial training set
        strategy: Sampling strategy - 'random' or 'diverse' (requires BitBIRCH)
        featurizer_type: Type of molecular featurizer for diverse sampling
        cache_dir: Directory for feature cache (diverse sampling only)
        random_state: Random seed for reproducibility

    Returns:
        List of compound IDs selected for initial training set

    Examples:
        >>> # Random sampling
        >>> ids = select_initial_sample(df, n_initial=10, strategy='random')

        >>> # Diverse sampling (requires BitBIRCH)
        >>> ids = select_initial_sample(
        ...     df, n_initial=10, strategy='diverse',
        ...     featurizer_type='morgan', cache_dir=Path('.cache')
        ... )
    """
    if n_initial > len(compounds_df):
        logger.warning(f"n_initial ({n_initial}) > available compounds ({len(compounds_df)}), using all compounds")
        n_initial = len(compounds_df)

    if strategy == 'random':
        selected_ids = compounds_df.sample(n=n_initial, random_state=random_state)['ID'].tolist()
        logger.info(f"Selected {n_initial} compounds using random sampling")
        return selected_ids

    elif strategy == 'diverse':
        try:
            from learnm8.acquisition.bitbirch import BitBIRCHAcquisition

            cache_dir = cache_dir or Path('.cache')

            features = extract_features(
                compounds_df['SMILES'].tolist(),
                featurizer_type,
                cache_dir
            )

            acq = BitBIRCHAcquisition(
                features=features,
                compound_ids=compounds_df['ID'].tolist(),
                featurizer_type=featurizer_type,
                random_state=random_state
            )

            temp_df = compounds_df.copy()
            temp_df['prediction'] = 0.0

            selected_df = acq.select(temp_df, n_initial)
            selected_ids = selected_df['ID'].tolist()

            logger.info(f"Selected {n_initial} compounds using diverse sampling (BitBIRCH)")
            return selected_ids

        except ImportError:
            logger.warning("BitBIRCH not available, falling back to random sampling")
            return select_initial_sample(compounds_df, n_initial, 'random', random_state=random_state)
        except Exception as e:
            logger.warning(f"Diverse sampling failed ({e}), falling back to random sampling")
            return select_initial_sample(compounds_df, n_initial, 'random', random_state=random_state)

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Must be 'random' or 'diverse'")

import logging

import polars as pl

logger = logging.getLogger(__name__)

STATUS_UNLABELED = 'unlabeled'
STATUS_LABELED = 'labeled'
STATUS_PRUNED = 'pruned'
VALID_STATUSES = [STATUS_UNLABELED, STATUS_LABELED, STATUS_PRUNED]


def initialize_master_dataframe(
    compound_pool: 'pl.DataFrame',
    initial_labeled_ids: list[str],
    initial_target_values: 'pl.Series',
    target_column: str
) -> 'pl.DataFrame':
    """Create initial master DataFrame from compound pool.

    .. deprecated:: 0.6.0
        This function is deprecated. Import from learnm8.core.initialization instead.
        This copy will be removed in a future version.

    .. note::
        This function uses pandas internally for backwards compatibility.
        Type hints indicate Polars but implementation is pandas-based.

    Uses vectorized operations for efficient initialization. Initial compounds are
    marked with labeled_cycle=-1 and selected_cycle=-1 to distinguish them from
    compounds labeled in cycle 0.

    Args:
        compound_pool: DataFrame with ID and SMILES columns
        initial_labeled_ids: List of compound IDs that are initially labeled
        initial_target_values: Series with target values indexed by compound ID
        target_column: Name of the target property column (e.g., 'Activity')

    Returns:
        Master DataFrame with all compounds and metadata columns

    Note:
        - Uses actual target column name (not generic 'target_value')
        - Initial compounds marked with labeled_cycle=-1 and selected_cycle=-1
        - Does not create last_prediction/last_uncertainty columns (redundant)
        - Vectorized initialization for O(n) performance

    Example:
        >>> pool = pd.DataFrame({'ID': ['C1', 'C2', 'C3'], 'SMILES': ['CCO', 'CCC', 'CCN']})
        >>> initial_ids = ['C1']
        >>> initial_values = pd.Series([0.5], index=['C1'])
        >>> master_df = initialize_master_dataframe(pool, initial_ids, initial_values, 'Activity')
        >>> print(master_df.loc[master_df['ID'] == 'C1', 'labeled_cycle'].values[0])
        -1
        >>> print(master_df.loc[master_df['ID'] == 'C1', 'selected_cycle'].values[0])
        -1
    """
    if 'ID' not in compound_pool.columns or 'SMILES' not in compound_pool.columns:
        missing = [c for c in ['ID', 'SMILES'] if c not in compound_pool.columns]
        raise ValueError(
            f"compound_pool is missing required columns: {missing}. "
            f"Available columns: {list(compound_pool.columns)}. "
            f"Ensure your compound pool has 'ID' and 'SMILES' columns."
        )

    master_df = compound_pool[['ID', 'SMILES']].copy()

    master_df['status'] = pd.Categorical(
        [STATUS_LABELED if cid in initial_labeled_ids else STATUS_UNLABELED
         for cid in master_df['ID']],
        categories=VALID_STATUSES
    )

    master_df['labeled_cycle'] = pd.Series(dtype='Int64')
    master_df['selected_cycle'] = pd.Series(dtype='Int64')
    master_df['pruned_cycle'] = pd.Series(dtype='Int64')
    master_df[target_column] = pd.Series(dtype='float64')

    mask = master_df['ID'].isin(initial_labeled_ids)
    id_to_value = initial_target_values.to_dict()
    master_df.loc[mask, target_column] = master_df.loc[mask, 'ID'].map(id_to_value)
    master_df.loc[mask, 'labeled_cycle'] = -1
    master_df.loc[mask, 'selected_cycle'] = -1

    logger.info(f"Initialized master DataFrame with {len(master_df)} compounds, "
                f"{len(initial_labeled_ids)} initially labeled")

    return master_df




def get_prediction_columns(
    master_df: 'pl.DataFrame'
) -> tuple[list[str], list[str]]:
    """Extract prediction and uncertainty column names.

    Args:
        master_df: Master DataFrame

    Returns:
        Tuple of (prediction_columns, uncertainty_columns) sorted by cycle number

    Example:
        >>> pred_cols, unc_cols = get_prediction_columns(master_df)
        >>> print(pred_cols)
        ['prediction_cycle_0', 'prediction_cycle_1', 'prediction_cycle_2']
    """
    pred_cols = [col for col in master_df.columns if col.startswith('prediction_cycle_')]
    unc_cols = [col for col in master_df.columns if col.startswith('uncertainty_cycle_')]

    pred_cols.sort(key=lambda x: int(x.split('_')[-1]))
    unc_cols.sort(key=lambda x: int(x.split('_')[-1]))

    return pred_cols, unc_cols


def validate_master_dataframe(
    master_df: 'pl.DataFrame'
) -> bool:
    """Validate master DataFrame schema.

    Validates core columns required for compound tracking. Does not validate:
    - Target column name (varies by experiment)
    - Prediction columns (added dynamically per cycle)
    - Deprecated columns (last_prediction, last_uncertainty removed in v0.5.0)

    Args:
        master_df: Master DataFrame to validate

    Returns:
        True if valid

    Raises:
        ValueError: If schema is invalid

    Note:
        Target column name varies by experiment (e.g., 'Activity', 'pIC50'),
        so it is not validated here. Prediction columns are added dynamically
        per cycle (e.g., 'prediction_cycle_0'), so they are not required at
        initialization.

    Example:
        >>> validate_master_dataframe(master_df)
        True
    """
    required_columns = [
        'ID', 'SMILES', 'status', 'labeled_cycle', 'selected_cycle', 'pruned_cycle'
    ]

    missing_cols = [col for col in required_columns if col not in master_df.columns]
    if missing_cols:
        raise ValueError(
            f"Master DataFrame missing required columns: {missing_cols}. "
            f"Available columns: {list(master_df.columns)}. "
            f"Required columns are: {required_columns}. "
            f"This may indicate the DataFrame was not properly initialized."
        )

    if master_df['ID'].is_duplicated().any():
        dupes = master_df.filter(pl.col('ID').is_duplicated())['ID'].unique().to_list()
        from learnm8.exceptions import _truncate_list
        raise ValueError(
            f"Duplicate compound IDs found in master DataFrame: {_truncate_list(dupes)}. "
            f"Each compound must have a unique ID. Check your input data for duplicate entries."
        )

    if master_df['status'].dtype == pl.Categorical:
        current_categories = master_df['status'].cat.get_categories().to_list()
        if set(current_categories) != set(VALID_STATUSES):
            logger.warning(f"Status column has incorrect categories {current_categories}, expected {VALID_STATUSES}")
    elif master_df['status'].dtype == pl.Enum:
        current_categories = master_df['status'].dtype.categories
        if set(current_categories) != set(VALID_STATUSES):
            logger.warning(f"Status column has incorrect enum categories {current_categories}, expected {VALID_STATUSES}")
    elif master_df['status'].dtype not in [pl.Categorical, pl.Utf8]:
        raise ValueError(
            f"Status column must be categorical, enum, or string type, "
            f"got {master_df['status'].dtype}. "
            f"This may indicate the DataFrame was not properly initialized."
        )

    invalid_statuses = set(master_df['status'].drop_nulls().unique().to_list()) - set(VALID_STATUSES)
    if invalid_statuses:
        raise ValueError(
            f"Invalid status values found: {invalid_statuses}. "
            f"Valid status values are: {VALID_STATUSES}. "
            f"Check that compound statuses are set correctly during initialization."
        )

    logger.debug("Master DataFrame validation passed")
    return True

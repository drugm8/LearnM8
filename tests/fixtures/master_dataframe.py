"""Master DataFrame fixtures for LearnM8 tests.

After the parquet migration, predictions live in per-cycle parquet files
rather than columns on the master DataFrame. The fixtures here build the
narrow master DataFrame and, where useful, write companion parquet files
into a tmp_path so tests can exercise the parquet read path.

CRITICAL: create_initialized_master_df() helper function is used by 50+ tests.
Preserve exact signature and behavior during refactoring.
"""

from typing import Optional, List
import pytest
import polars as pl
import numpy as np


def create_initialized_master_df(
    valid_compounds: pl.DataFrame,
    target_col: str,
    initial_labeled_ids: Optional[List[str]] = None,
    initial_measurements: Optional[dict] = None
) -> pl.DataFrame:
    """Helper to create master DataFrame with initial labeled compounds.

    This helper mimics the old initialize_master_dataframe behavior for test compatibility.
    It creates an empty master DataFrame and then manually labels initial compounds.

    CRITICAL: This function is used by 50+ tests. Do not modify signature or behavior.

    Args:
        valid_compounds: Polars DataFrame with ID and SMILES columns
        target_col: Name of target column
        initial_labeled_ids: Optional list of compound IDs to label initially
        initial_measurements: Optional dict mapping IDs to measurements

    Returns:
        Master DataFrame with initial compounds labeled
    """
    from learnm8.core.initialization import initialize_master_dataframe_empty
    from learnm8.core.dataframe_ops import update_status

    master_df = initialize_master_dataframe_empty(valid_compounds, target_col)

    if initial_labeled_ids and initial_measurements is not None and len(initial_labeled_ids) > 0:
        master_df = update_status(
            df=master_df,
            compound_ids=initial_labeled_ids,
            new_status='labeled',
            cycle=-1,
            target_col=target_col,
            target_values=initial_measurements
        )

    return master_df


@pytest.fixture
def sample_master_df(sample_compounds) -> pl.DataFrame:
    """Create master DataFrame with 2 labeled compounds (indices 0-1).

    Uses sample_compounds fixture as base (5 compounds).
    Depends on: sample_compounds from molecules.py
    """
    compounds = sample_compounds.clone()

    initial_compounds = compounds.head(2)
    initial_ids = initial_compounds.get_column('ID').to_list()
    initial_values = dict(zip(initial_ids, [0.3, 0.6]))

    return create_initialized_master_df(
        valid_compounds=compounds,
        target_col='Activity',
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values
    )


def _write_predictions_parquet(
    output_dir, cycle: int, compound_ids, predictions, uncertainties
):
    """Write a prediction parquet file using the production helper.

    Used by fixtures that previously added prediction columns to the master DF.
    """
    from learnm8.core.persistence import write_cycle_predictions

    cycle_predictions = pl.DataFrame({
        'ID': list(compound_ids),
        'prediction': predictions,
        'uncertainty': uncertainties,
    })
    return write_cycle_predictions(cycle_predictions, output_dir, cycle)


@pytest.fixture
def master_df_with_predictions(sample_master_df, tmp_path):
    """Master DataFrame plus a cycle-0 prediction parquet file.

    Returns a (master_df, output_dir) tuple. The master DF is the narrow
    post-migration shape; predictions live in ``output_dir/prediction_cycle_0.parquet``.

    Depends on: sample_master_df, tmp_path
    """
    master_df = sample_master_df.clone()

    unlabeled = master_df.filter(pl.col('status') == 'unlabeled')
    unlabeled_ids = unlabeled.get_column('ID').to_list()

    np.random.seed(42)
    predictions = np.random.uniform(0.2, 0.8, len(unlabeled_ids))
    uncertainties = np.random.uniform(0.1, 0.3, len(unlabeled_ids))

    _write_predictions_parquet(tmp_path, 0, unlabeled_ids, predictions, uncertainties)
    return master_df, tmp_path


@pytest.fixture
def master_df_multi_cycle(sample_master_df, tmp_path):
    """Master DataFrame plus prediction parquet files across two cycles.

    Returns a (master_df, output_dir) tuple. Each cycle's predictions are
    written to ``output_dir/prediction_cycle_{N}.parquet`` and the master DF
    captures the resulting status updates.

    Depends on: sample_master_df, tmp_path
    """
    from learnm8.core.dataframe_ops import update_status

    master_df = sample_master_df.clone()

    np.random.seed(42)

    unlabeled = master_df.filter(pl.col('status') == 'unlabeled')
    unlabeled_ids = unlabeled.get_column('ID').to_list()

    predictions_0 = np.random.uniform(0.2, 0.8, len(unlabeled_ids))
    uncertainties_0 = np.random.uniform(0.1, 0.3, len(unlabeled_ids))
    _write_predictions_parquet(tmp_path, 0, unlabeled_ids, predictions_0, uncertainties_0)

    selected_ids_0 = unlabeled_ids[:1]
    target_values_0 = dict(zip(selected_ids_0, [0.45]))
    master_df = update_status(
        df=master_df,
        compound_ids=selected_ids_0,
        new_status='labeled',
        cycle=0,
        target_col='Activity',
        target_values=target_values_0
    )

    unlabeled = master_df.filter(pl.col('status') == 'unlabeled')
    unlabeled_ids = unlabeled.get_column('ID').to_list()

    if len(unlabeled_ids) > 0:
        predictions_1 = np.random.uniform(0.3, 0.9, len(unlabeled_ids))
        uncertainties_1 = np.random.uniform(0.1, 0.25, len(unlabeled_ids))
        _write_predictions_parquet(tmp_path, 1, unlabeled_ids, predictions_1, uncertainties_1)

        selected_ids_1 = unlabeled_ids[:1]
        target_values_1 = dict(zip(selected_ids_1, [0.65]))
        master_df = update_status(
            df=master_df,
            compound_ids=selected_ids_1,
            new_status='labeled',
            cycle=1,
            target_col='Activity',
            target_values=target_values_1
        )

    return master_df, tmp_path

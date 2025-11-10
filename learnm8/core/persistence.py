"""
CSV-based persistence for active learning results.

Saves all experiment data to organized CSV files with metadata comments:
- compounds_final.csv: Master DataFrame with all compound data
- cycle_metrics.csv: Per-cycle performance metrics
- selection_history.csv: Detailed selection records
- validation_report.csv: Invalid compounds (if any)
- config.json: Experiment configuration

All CSV files include metadata comments for self-documentation.
No query classes - files are directly usable with pandas, Excel, etc.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import polars as pl
import numpy as np

from .validation import ValidationResult

logger = logging.getLogger(__name__)


def _add_csv_metadata(file_path: Path, metadata: Dict[str, Any]) -> None:
    """
    Add metadata as comment lines at the top of an existing CSV file.

    Modifies the file in-place by prepending comment lines (prefixed with #).
    Gracefully handles errors by logging warnings without raising exceptions.

    Parameters
    ----------
    file_path : Path
        Path to existing CSV file
    metadata : Dict[str, Any]
        Metadata key-value pairs. Empty string values create blank comment lines.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        comment_lines = []
        comment_lines.append('# LearnM8 Active Learning Results\n')
        for key, value in metadata.items():
            if value == '':
                comment_lines.append('#\n')
            else:
                comment_lines.append(f'# {key}: {value}\n')
        comment_lines.append('#\n')

        with open(file_path, 'w') as f:
            f.writelines(comment_lines)
            f.writelines(lines)

    except IOError as e:
        logger.warning(f"Failed to add metadata to {file_path}: {e}")


def _organize_columns(df: pl.DataFrame, column_groups: List[List[str]]) -> pl.DataFrame:
    """
    Reorder DataFrame columns into logical groups for readability.

    Handles missing columns gracefully (skips if not present).
    Preserves all columns by adding unlisted ones at the end, sorted alphabetically.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to reorder
    column_groups : List[List[str]]
        Nested list of column names in desired order.
        Example: [['ID', 'SMILES'], ['prediction_0', 'prediction_1']]

    Returns
    -------
    pl.DataFrame
        DataFrame with reordered columns
    """
    ordered_columns = []
    for group in column_groups:
        for col in group:
            if col in df.columns:
                ordered_columns.append(col)

    remaining = set(df.columns) - set(ordered_columns)
    remaining = sorted(remaining)
    ordered_columns.extend(remaining)

    return df.select(ordered_columns)


def save_results(
    compounds_df: pl.DataFrame,
    cycle_metrics: List[Dict[str, Any]],
    validation_result: ValidationResult,
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Path]:
    """
    Save all experiment results to organized CSV files with metadata.

    Creates 4-5 files in output_dir:
    1. compounds_final.csv: Master DataFrame with all compound data
       - Base columns (ID, SMILES, status, cycles)
       - Prediction columns (chronologically ordered)
       - Uncertainty columns (chronologically ordered)

    2. cycle_metrics.csv: Per-cycle performance metrics
       - Core metrics (cycle, strategy, batch size, counts)
       - Prediction statistics (mean, std, min, max, median)
       - Measured statistics (from oracle measurements)
       - Best compound tracking

    3. selection_history.csv: Detailed selection records
       - One row per compound per cycle
       - Includes prediction/uncertainty at selection time
       - Useful for analyzing acquisition strategy performance

    4. validation_report.csv: Invalid compounds (OPTIONAL)
       - Only created if invalid compounds exist
       - Lists all compounds that failed validation
       - Includes error messages

    5. config.json: Experiment configuration
       - Complete parameter record
       - JSON format for easy parsing

    All CSV files include metadata comments for self-documentation.

    Parameters
    ----------
    compounds_df : pl.DataFrame
        Master DataFrame with all compound data and predictions
    cycle_metrics : List[Dict[str, Any]]
        List of per-cycle metric dictionaries
    validation_result : ValidationResult
        Validation results with invalid compounds and errors
    config : Dict[str, Any]
        Experiment configuration dictionary
    output_dir : Path
        Directory to save all output files

    Returns
    -------
    Dict[str, Path]
        Mapping of file type to saved file path
        Keys: 'compounds_final', 'cycle_metrics', 'selection_history',
              'validation_report' (optional), 'config'

    Example
    -------
    >>> saved_files = save_results(
    ...     compounds_df=final_df,
    ...     cycle_metrics=metrics_list,
    ...     validation_result=validation,
    ...     config=experiment_config,
    ...     output_dir=Path('results/experiment_001')
    ... )
    >>> print(saved_files['compounds_final'])
    results/experiment_001/compounds_final.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    logger.info("═══════════════════════════════════════════════════════════════")
    logger.info("Phase 5: Saving Results")
    logger.info("═══════════════════════════════════════════════════════════════")

    target_col = config.get('target_col', 'target')
    featurizer_type = config.get('featurizer_type', 'unknown')
    n_cycles = config.get('n_cycles', len([c for c in compounds_df.columns if c.startswith('prediction_cycle_')]))
    mode = config.get('mode', 'run')
    score_direction = config.get('score_direction', 'higher')

    try:
        compounds_final = compounds_df.clone()
        base_cols = ['ID', 'SMILES', 'status', 'labeled_cycle', 'selected_cycle', 'pruned_cycle', target_col]
        pred_cols = sorted([c for c in compounds_df.columns if c.startswith('prediction_cycle_')],
                          key=lambda x: int(x.split('_')[-1]))
        unc_cols = sorted([c for c in compounds_df.columns if c.startswith('uncertainty_cycle_')],
                         key=lambda x: int(x.split('_')[-1]))
        compounds_final = _organize_columns(compounds_final, [base_cols, pred_cols, unc_cols])

        final_path = output_dir / 'compounds_final.csv'
        compounds_final.write_csv(final_path)

        n_labeled = compounds_df.filter(pl.col('status') == 'labeled').height
        n_unlabeled = compounds_df.filter(pl.col('status') == 'unlabeled').height
        n_pruned = compounds_df.filter(pl.col('status') == 'pruned').height

        metadata = {
            'Read Hint': 'Use pandas.read_csv(path, comment=\'#\') to ignore metadata comments',
            'Target': target_col,
            'Featurizer': featurizer_type,
            'Score Direction': score_direction,
            'Total Cycles': n_cycles,
            'Total Compounds': len(compounds_final),
            'Labeled': n_labeled,
            'Unlabeled': n_unlabeled,
            'Pruned': n_pruned,
            '': '',
            'Column Guide': '',
            'ID': 'Unique compound identifier',
            'SMILES': 'Molecular structure',
            'status': 'labeled/unlabeled/pruned',
            'labeled_cycle': 'Cycle when labeled (-1 for initial)',
            'selected_cycle': 'Cycle when selected for measurement',
            'pruned_cycle': 'Cycle when pruned (-1 if not pruned)',
            target_col: 'Measured target value',
            'prediction_cycle_N': 'Model prediction at cycle N',
            'uncertainty_cycle_N': 'Model uncertainty at cycle N (if available)'
        }
        _add_csv_metadata(final_path, metadata)

        saved_files['compounds_final'] = final_path
        logger.debug(f"Saved compounds_final.csv: {len(compounds_final)} compounds")
        logger.info(f"Saved compounds_final.csv ({len(compounds_final)} compounds)")

    except IOError as e:
        logger.error(f"Failed to save compounds_final.csv: {e}")
        raise RuntimeError(f"Failed to save compounds_final.csv: {e}")

    try:
        metrics_df = pl.DataFrame(cycle_metrics)
        list_cols = ['selected_ids', 'pruned_ids']
        cols_to_drop = [c for c in list_cols if c in metrics_df.columns]
        if cols_to_drop:
            metrics_df = metrics_df.drop(cols_to_drop)

        core_cols = ['cycle', 'strategy', 'batch_size', 'selected_count', 'remaining_unlabeled',
                    'cumulative_labeled', 'cumulative_pruned']
        pred_cols = ['prediction_mean', 'prediction_std', 'prediction_min', 'prediction_max', 'prediction_median']
        unc_cols = ['uncertainty_mean', 'uncertainty_std', 'uncertainty_min', 'uncertainty_max']
        measured_cols = ['measured_mean', 'measured_std', 'measured_min', 'measured_max',
                        'measured_best', 'best_so_far']
        metrics_df = _organize_columns(metrics_df, [core_cols, pred_cols, unc_cols, measured_cols])

        metrics_path = output_dir / 'cycle_metrics.csv'
        metrics_df.write_csv(metrics_path)

        metadata = {
            'Read Hint': 'Use pandas.read_csv(path, comment=\'#\') to ignore metadata comments',
            'Description': 'Per-cycle performance metrics',
            '': '',
            'Key Metrics': '',
            'cycle': 'Cycle number',
            'strategy': 'Acquisition strategy used',
            'batch_size': 'Compounds selected this cycle',
            'selected_count': 'Total compounds selected',
            'remaining_unlabeled': 'Unlabeled compounds remaining',
            'cumulative_labeled': 'Total labeled compounds',
            'cumulative_pruned': 'Total pruned compounds',
            'prediction_*': 'Statistics of model predictions',
            'uncertainty_*': 'Statistics of model uncertainties',
            'measured_*': 'Statistics of oracle measurements',
            'best_so_far': 'Best measured value found so far'
        }
        _add_csv_metadata(metrics_path, metadata)

        saved_files['cycle_metrics'] = metrics_path
        logger.debug(f"Saved cycle_metrics.csv: {len(metrics_df)} cycles")
        logger.info(f"Saved cycle_metrics.csv ({len(metrics_df)} cycles)")

    except IOError as e:
        logger.error(f"Failed to save cycle_metrics.csv: {e}")
        raise RuntimeError(f"Failed to save cycle_metrics.csv: {e}")

    try:
        selection_history = []
        for cycle_data in cycle_metrics:
            cycle = cycle_data['cycle']
            strategy = cycle_data['strategy']

            selected_compounds = compounds_df.filter(pl.col('selected_cycle') == cycle)

            for compound in selected_compounds.iter_rows(named=True):
                sel = int(compound['selected_cycle'])
                pred_col = f'prediction_cycle_{sel}'
                unc_col = f'uncertainty_cycle_{sel}'

                # Design invariant: In execute_cycle(), compounds are selected and labeled
                # within the same cycle, so selected_cycle equals the cycle whose predictions
                # were used for selection. This ensures we always look up the correct
                # prediction/uncertainty values that informed the selection decision.

                record = {
                    'cycle': cycle,
                    'strategy': strategy,
                    'ID': compound['ID'],
                    'SMILES': compound['SMILES'],
                    'measured_value': compound.get(target_col, None),
                    'prediction_at_selection': compound.get(pred_col, None),
                    'uncertainty_at_selection': compound.get(unc_col, None)
                }
                selection_history.append(record)

        if selection_history:
            selection_df = pl.DataFrame(selection_history)
        else:
            selection_df = pl.DataFrame(schema={
                'cycle': pl.Int64,
                'strategy': pl.Utf8,
                'ID': pl.Utf8,
                'SMILES': pl.Utf8,
                'measured_value': pl.Float64,
                'prediction_at_selection': pl.Float64,
                'uncertainty_at_selection': pl.Float64
            })

        selection_path = output_dir / 'selection_history.csv'
        selection_df.write_csv(selection_path)

        metadata = {
            'Read Hint': 'Use pandas.read_csv(path, comment=\'#\') to ignore metadata comments',
            'Description': 'Detailed selection records (one row per compound per cycle)',
            '': '',
            'Columns': '',
            'cycle': 'Cycle when selected',
            'strategy': 'Acquisition strategy used',
            'ID': 'Compound identifier',
            'SMILES': 'Molecular structure',
            'measured_value': 'Oracle measurement result',
            'prediction_at_selection': 'Model prediction when selected',
            'uncertainty_at_selection': 'Model uncertainty when selected'
        }
        _add_csv_metadata(selection_path, metadata)

        saved_files['selection_history'] = selection_path
        logger.debug(f"Saved selection_history.csv: {len(selection_df)} selections")
        logger.info(f"Saved selection_history.csv ({len(selection_df)} selections)")

    except IOError as e:
        logger.error(f"Failed to save selection_history.csv: {e}")
        raise RuntimeError(f"Failed to save selection_history.csv: {e}")

    if validation_result.invalid_compounds.height > 0:
        try:
            invalid_df = validation_result.invalid_compounds.clone()
            # Map errors using join
            from learnm8.utils.polars_utils import map_values_via_join
            invalid_df = map_values_via_join(invalid_df, validation_result.validation_errors, 'ID', 'error')

            validation_path = output_dir / 'validation_report.csv'
            invalid_df.write_csv(validation_path)

            success_rate = validation_result.success_rate
            metadata = {
                'Read Hint': 'Use pandas.read_csv(path, comment=\'#\') to ignore metadata comments',
                'Total Invalid': len(invalid_df),
                'Success Rate': f'{success_rate:.1%}',
                'Common Issues': 'Invalid SMILES, missing values, format errors',
                '': '',
                'Columns': '',
                'ID': 'Compound identifier',
                'SMILES': 'Molecular structure (invalid)',
                'error': 'Validation error message'
            }
            _add_csv_metadata(validation_path, metadata)

            saved_files['validation_report'] = validation_path
            logger.debug(f"Saved validation_report.csv: {len(invalid_df)} invalid compounds")
            logger.info(f"Saved validation_report.csv ({len(invalid_df)} invalid compounds)")

        except IOError as e:
            logger.warning(f"Failed to save validation_report.csv: {e}")

    try:
        config_path = output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        saved_files['config'] = config_path
        logger.debug(f"Saved config.json")
        logger.info(f"Saved config.json")

    except IOError as e:
        logger.error(f"Failed to save config.json: {e}")
        raise RuntimeError(f"Failed to save config.json: {e}")

    logger.info(f"All results saved to {output_dir}")
    return saved_files

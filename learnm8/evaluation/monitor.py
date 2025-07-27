"""Monitoring utilities for active learning experiments."""

import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
import numpy as np

from learnm8.evaluation.metrics import (
	calculate_rmse, calculate_mae, calculate_mse, calculate_r2, calculate_mape,
	calculate_spearman_correlation, calculate_average_score, 
	calculate_multiple_top_k_overlaps,
	calculate_multiple_enrichment_factors,
	calculate_molecular_similarity_metrics
)
from learnm8.utils.logging import get_logger, log_file_operation


def create_cycle_report(cycle: int, predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
					   newly_selected_df: pd.DataFrame, target_column: str, 
					   score_direction: str = 'higher',
					   cumulative_compounds_selected: int = 0, 
					   ground_truth_enrichment_factors: dict = None,
					   previously_selected_df: pd.DataFrame = None,
					   oracle_mode: str = 'benchmark') -> dict:
	"""
	Create a comprehensive report for a single cycle.
	
	Args:
		cycle: Cycle number
		predictions_df: DataFrame with 'ID' and 'prediction' columns
		ground_truth_df: DataFrame with ground truth values
		newly_selected_df: DataFrame of newly selected compounds this cycle
		target_column: Target column name
		score_direction: 'higher' or 'lower'
		cumulative_compounds_selected: Total compounds selected so far
		ground_truth_enrichment_factors: Pre-calculated ground truth enrichment factors (constant across cycles)
		previously_selected_df: DataFrame with compounds selected in previous cycles (for similarity metrics)
		oracle_mode: 'benchmark' or 'run' - determines which metrics are calculated
		
	Returns:
		Dictionary with cycle metrics (some may be None in run mode)
	"""
	# Calculate model performance metrics based on oracle mode
	if oracle_mode == 'benchmark':
		# Benchmark mode: calculate metrics on full prediction set
		merged = pd.merge(predictions_df, ground_truth_df[['ID', target_column]], on='ID', how='inner')
		y_true = merged[target_column].values
		y_pred = merged['prediction'].values
		
		# Calculate multiple top-k overlaps (only available in benchmark mode)
		top_k_overlaps = calculate_multiple_top_k_overlaps(
			predictions_df, ground_truth_df, target_column, score_direction
		)
		
	else:  # run mode
		# Run mode: calculate metrics only on acquired compounds this cycle
		if target_column in newly_selected_df.columns:
			# Get predictions for newly selected compounds
			acquired_with_predictions = pd.merge(
				newly_selected_df[['ID', target_column]], 
				predictions_df[['ID', 'prediction']], 
				on='ID', how='inner'
			)
			if len(acquired_with_predictions) >= 2:  # Need at least 2 points for meaningful metrics
				y_true = acquired_with_predictions[target_column].values
				y_pred = acquired_with_predictions['prediction'].values
			else:
				y_true, y_pred = np.array([]), np.array([])
		else:
			y_true, y_pred = np.array([]), np.array([])
		
		# Top-k overlaps not available in run mode (no ground truth for full dataset)
		top_k_overlaps = {
			'top_100_overlap': None,
			'top_1000_overlap': None,
			'top_0_1_percent_overlap': None,
			'top_1_percent_overlap': None
		}
	
	# Calculate model performance metrics (handle insufficient data case)
	if len(y_true) >= 2:
		rmse = calculate_rmse(y_true, y_pred)
		mae = calculate_mae(y_true, y_pred)
		mse = calculate_mse(y_true, y_pred)
		r2 = calculate_r2(y_true, y_pred)
		mape = calculate_mape(y_true, y_pred)
		spearman_corr = calculate_spearman_correlation(y_true, y_pred)
	else:
		# Insufficient data for meaningful metrics
		rmse = mae = mse = r2 = mape = spearman_corr = None
	
	# Calculate average score of newly selected compounds
	# newly_selected_df should already contain the target_column from oracle measurement
	if target_column in newly_selected_df.columns:
		avg_score_selected = calculate_average_score(newly_selected_df[target_column].values)
	else:
		# Fallback: merge with ground truth if target_column not in newly_selected_df
		newly_selected_with_truth = pd.merge(
			newly_selected_df, ground_truth_df[['ID', target_column]], on='ID', how='inner'
		)
		avg_score_selected = calculate_average_score(newly_selected_with_truth[target_column].values)
	
	# Calculate enrichment factors based on oracle mode and data availability
	enrichment_factors = {}
	if oracle_mode == 'benchmark' and 'Activity' in ground_truth_df.columns:
		# Benchmark mode: calculate EF using full prediction set vs Activity labels
		merged_activity = pd.merge(predictions_df, ground_truth_df[['ID', 'Activity']], on='ID', how='inner')
		if len(merged_activity) > 0:
			enrichment_factors = calculate_multiple_enrichment_factors(
				merged_activity['prediction'].values,
				merged_activity['Activity'].values,
				score_direction
			)
	else:
		# Run mode or no Activity column: enrichment factors not available
		enrichment_factors = {'ef_5': None, 'ef_1': None, 'ef_0_5': None, 'ef_0_1': None}
	
	# Ensure enrichment_factors has default structure if empty
	if not enrichment_factors:
		enrichment_factors = {'ef_5': None, 'ef_1': None, 'ef_0_5': None, 'ef_0_1': None}
	
	# Calculate molecular similarity metrics
	similarity_metrics = calculate_molecular_similarity_metrics(
		newly_selected_df, previously_selected_df
	)
	
	# Create base report dict with all metrics
	report = {
		'cycle': cycle,
		# Model performance metrics
		'rmse': rmse,
		'mae': mae,
		'mse': mse,
		'r2': r2,
		'mape': mape,
		'spearman_correlation': spearman_corr,
		# Selection quality metrics
		'avg_score_selected': avg_score_selected,
		'n_compounds_selected': len(newly_selected_df),
		'cumulative_compounds_selected': cumulative_compounds_selected,
		'n_total_predictions': len(predictions_df)
	}
	
	# Add top-k overlap metrics
	report.update(top_k_overlaps)
	
	# Add enrichment factors
	report.update(enrichment_factors)
	
	# Add molecular similarity metrics
	report.update(similarity_metrics)
	
	# Add ground truth enrichment factors (constant across cycles)
	if ground_truth_enrichment_factors:
		report.update(ground_truth_enrichment_factors)
	else:
		# Set default values if no ground truth enrichment factors provided
		report.update({
			'ground_truth_ef_5': None,
			'ground_truth_ef_1': None,
			'ground_truth_ef_0_5': None,
			'ground_truth_ef_0_1': None
		})
	
	return report


def save_monitoring_results(results: list[dict], output_path: Path) -> None:
	"""
	Save monitoring results to CSV file.
	
	Args:
		results: List of cycle reports
		output_path: Path to save CSV file
	"""
	df = pd.DataFrame(results)
	df.to_csv(output_path, index=False)
	logger = get_logger()
	log_file_operation(logger, "Saved monitoring results", str(output_path))


def save_consolidated_predictions(cycle_predictions: list[dict], ground_truth_df: pd.DataFrame, 
								  target_column: str, output_path: Path) -> None:
	"""
	Save consolidated predictions across all cycles to CSV file in compact wide format.
	
	Format: One row per compound with columns ID, SMILES, true_label, prediction_cycle_1, prediction_cycle_2, etc.
	This is more compact than the long format as it avoids repeating compound metadata.
	
	Args:
		cycle_predictions: List of dictionaries with cycle prediction data
		ground_truth_df: DataFrame with ground truth values
		target_column: Target column name for true labels
		output_path: Path to save predictions.csv file
	"""
	if not cycle_predictions:
		logger = get_logger()
		logger.warning("No cycle predictions to save")
		return
		
	# Convert to DataFrame (long format)
	predictions_df = pd.DataFrame(cycle_predictions)
	
	# Pivot to wide format: one row per compound, one column per cycle
	predictions_wide = predictions_df.pivot_table(
		index=['ID', 'SMILES'], 
		columns='cycle', 
		values='prediction', 
		aggfunc='first'
	).reset_index()
	
	# Rename prediction columns to prediction_cycle_X format
	prediction_columns = {}
	for col in predictions_wide.columns:
		if isinstance(col, int):  # These are the cycle numbers
			prediction_columns[col] = f'prediction_cycle_{col}'
	predictions_wide = predictions_wide.rename(columns=prediction_columns)
	
	# Merge with ground truth to add true labels
	predictions_with_truth = pd.merge(
		predictions_wide, 
		ground_truth_df[['ID', target_column]], 
		on='ID', 
		how='left'
	)
	
	# Rename target column to true_label for clarity
	predictions_with_truth = predictions_with_truth.rename(columns={target_column: 'true_label'})
	
	# Ensure proper column order: ID, SMILES, true_label, then prediction columns in cycle order
	base_columns = ['ID', 'SMILES', 'true_label']
	prediction_cols = [col for col in predictions_with_truth.columns if col.startswith('prediction_cycle_')]
	prediction_cols.sort(key=lambda x: int(x.split('_')[-1]))  # Sort by cycle number
	
	column_order = base_columns + prediction_cols
	predictions_with_truth = predictions_with_truth[column_order]
	
	# Save to CSV
	predictions_with_truth.to_csv(output_path, index=False)
	logger = get_logger()
	log_file_operation(logger, "Saved predictions", str(output_path))


def save_monitoring_csv(results: list[dict], output_path: Path) -> None:
	"""
	Save monitoring results to monitoring.csv file.
	
	Args:
		results: List of cycle reports
		output_path: Path to save monitoring.csv file
	"""
	df = pd.DataFrame(results)
	df.to_csv(output_path, index=False)
	logger = get_logger()
	log_file_operation(logger, "Saved monitoring", str(output_path))


def print_cycle_report(report: dict, score_direction: str = 'higher') -> None:
	"""
	Print a formatted cycle report to console.
	
	Args:
		report: Cycle report dictionary
		score_direction: Direction for score interpretation
	"""
	console = Console()
	
	# Helper function to format metric values
	def format_metric(value, decimals=4, suffix=""):
		if value is None:
			return "N/A"
		return f"{value:.{decimals}f}{suffix}"
	
	# Create cycle metrics table with multiple columns
	table = Table(title=f"[bold cyan]Cycle {report['cycle']} Report[/bold cyan]", show_header=True)
	table.add_column("Model Performance", style="blue", width=20)
	table.add_column("Value", style="white", justify="right", width=12)
	table.add_column("Selection Quality", style="green", width=20)
	table.add_column("Value", style="white", justify="right", width=12)
	table.add_column("Enrichment Factors", style="yellow", width=20)
	table.add_column("Value", style="white", justify="right", width=12)

	# Prepare data for each column
	model_metrics = [
		("RMSE", format_metric(report.get('rmse'))),
		("MAE", format_metric(report.get('mae'))),
		("MSE", format_metric(report.get('mse'))),
		("R²", format_metric(report.get('r2'))),
		("MAPE", format_metric(report.get('mape'), 2, "%")),
		("Spearman Corr", format_metric(report.get('spearman_correlation'))),
	]

	selection_metrics = [
		("Avg Score Selected", format_metric(report.get('avg_score_selected'))),
		("Top-100 Overlap", format_metric(report.get('top_100_overlap'), 2, "%") if report.get('top_100_overlap') else "N/A"),
		("Top-1000 Overlap", format_metric(report.get('top_1000_overlap'), 2, "%") if report.get('top_1000_overlap') else "N/A"),
		("Compounds Selected", str(report['n_compounds_selected'])),
		("Cumulative Selected", str(report['cumulative_compounds_selected'])),
		("Total Predictions", str(report['n_total_predictions'])),
	]

	enrichment_metrics = [
		("ML EF 5%", format_metric(report.get('ef_5_0'), 2) if report.get('ef_5_0') else "N/A"),
		("ML EF 1%", format_metric(report.get('ef_1_0'), 2) if report.get('ef_1_0') else "N/A"),
		("GT EF 5%", format_metric(report.get('ground_truth_ef_5_0'), 2) if report.get('ground_truth_ef_5_0') else "N/A"),
		("GT EF 1%", format_metric(report.get('ground_truth_ef_1_0'), 2) if report.get('ground_truth_ef_1_0') else "N/A"),
		("Intra-batch Div", format_metric(report.get('intra_batch_diversity'), 3)),
		("Inter-cycle Div", format_metric(report.get('inter_cycle_diversity'), 3)),
		("Batch Novelty", format_metric(report.get('batch_novelty_score'), 3)),
	]

	# Add rows (fill empty cells as needed)
	max_rows = max(len(model_metrics), len(selection_metrics), len(enrichment_metrics))
	for i in range(max_rows):
		model = model_metrics[i] if i < len(model_metrics) else ("", "")
		selection = selection_metrics[i] if i < len(selection_metrics) else ("", "")
		enrichment = enrichment_metrics[i] if i < len(enrichment_metrics) else ("", "")
		
		table.add_row(model[0], model[1], selection[0], selection[1], enrichment[0], enrichment[1])
	
	# Score trend
	if 'prev_avg_score' in report and report['prev_avg_score'] is not None:
		curr = report['avg_score_selected']
		prev = report['prev_avg_score']
		
		if score_direction == 'higher':
			trend_symbol = "↑" if curr > prev else "↓" if curr < prev else "→"
			trend_color = "green" if curr > prev else "red" if curr < prev else "white"
		else:
			trend_symbol = "↓" if curr < prev else "↑" if curr > prev else "→"
			trend_color = "green" if curr < prev else "red" if curr > prev else "white"
		
		table.add_row("Score Trend", f"[{trend_color}]{trend_symbol}[/{trend_color}] ({prev:.4f} → {curr:.4f})")
	
	console.print()
	console.print(table)
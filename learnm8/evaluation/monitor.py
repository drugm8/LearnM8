"""Monitoring utilities for active learning experiments."""

import pandas as pd
from pathlib import Path
from evaluation.metrics import (
	calculate_rmse, calculate_average_score, 
	calculate_enrichment_factor, calculate_top_k_overlap
)


def create_cycle_report(cycle: int, predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
					   newly_selected_df: pd.DataFrame, target_column: str, 
					   score_direction: str = 'higher', top_k: int = 100,
					   enrichment_percentile: float = 1.0) -> dict:
	"""
	Create a comprehensive report for a single cycle.
	
	Args:
		cycle: Cycle number
		predictions_df: DataFrame with 'ID' and 'prediction' columns
		ground_truth_df: DataFrame with ground truth values
		newly_selected_df: DataFrame of newly selected compounds this cycle
		target_column: Target column name
		score_direction: 'higher' or 'lower'
		top_k: K value for top-k overlap
		enrichment_percentile: Percentile for enrichment factor
		
	Returns:
		Dictionary with cycle metrics
	"""
	# Merge predictions with ground truth
	merged = pd.merge(predictions_df, ground_truth_df[['ID', target_column]], on='ID')
	
	print(merged.head())  # Debugging line to check merged DataFrame
	
	# Calculate RMSE on all predictions
	rmse = calculate_rmse(merged[target_column].values, merged['prediction'].values)
	
	# Calculate top-k overlap
	top_k_overlap = calculate_top_k_overlap(
		predictions_df, ground_truth_df, top_k, target_column, score_direction
	)
	
	# Calculate average score of newly selected compounds
	newly_selected_with_truth = pd.merge(
		newly_selected_df, ground_truth_df[['ID', target_column]], on='ID'
	)
	avg_score_selected = calculate_average_score(newly_selected_with_truth[target_column].values)
	
	# Calculate enrichment factor if Activity column exists
	enrichment_factor = None
	if 'Activity' in ground_truth_df.columns:
		merged_activity = pd.merge(predictions_df, ground_truth_df[['ID', 'Activity']], on='ID')
		if len(merged_activity) > 0:
			enrichment_factor = calculate_enrichment_factor(
				merged_activity['prediction'].values,
				merged_activity['Activity'].values,
				enrichment_percentile
			)
	
	return {
		'cycle': cycle,
		'rmse': rmse,
		'top_k_overlap': top_k_overlap,
		'avg_score_selected': avg_score_selected,
		'enrichment_factor': enrichment_factor,
		'n_compounds_selected': len(newly_selected_df),
		'n_total_predictions': len(predictions_df)
	}


def save_monitoring_results(results: list[dict], output_path: Path) -> None:
	"""
	Save monitoring results to CSV file.
	
	Args:
		results: List of cycle reports
		output_path: Path to save CSV file
	"""
	df = pd.DataFrame(results)
	df.to_csv(output_path, index=False)
	print(f"Monitoring results saved to {output_path}")


def print_cycle_report(report: dict, score_direction: str = 'higher') -> None:
	"""
	Print a formatted cycle report to console.
	
	Args:
		report: Cycle report dictionary
		score_direction: Direction for score interpretation
	"""
	print(f"\n{'='*60}")
	print(f"Cycle {report['cycle']} Report")
	print(f"{'='*60}")
	print(f"RMSE: {report['rmse']:.4f}")
	print(f"Top-{report.get('k', 100)} Overlap: {report['top_k_overlap']:.2f}%")
	print(f"Average Score of Selected: {report['avg_score_selected']:.4f}")
	
	if report.get('enrichment_factor') is not None:
		print(f"Enrichment Factor: {report['enrichment_factor']:.2f}")
	
	print(f"Compounds Selected: {report['n_compounds_selected']}")
	print(f"Total Predictions: {report['n_total_predictions']}")
	
	# Add trend indicator for average score
	if 'prev_avg_score' in report and report['prev_avg_score'] is not None:
		curr = report['avg_score_selected']
		prev = report['prev_avg_score']
		
		if score_direction == 'higher':
			trend = "↑" if curr > prev else "↓" if curr < prev else "→"
		else:
			trend = "↓" if curr < prev else "↑" if curr > prev else "→"
		
		print(f"Score Trend: {trend} (from {prev:.4f} to {curr:.4f})")
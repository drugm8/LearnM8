"""Core evaluation functions for LearnM8 active learning.

This module contains the main evaluation logic that intelligently adapts
metrics based on mode (benchmark vs run) and data availability.
"""

import logging
from typing import Dict, Any, Optional
import polars as pl
import numpy as np


# Import specialized metric functions from modular metrics package
from .metrics.performance import (
	calculate_spearman_correlation, calculate_average_score, calculate_mape
)
from .metrics.enrichment import (
	calculate_top_k_overlap, calculate_multiple_top_k_overlaps,
	calculate_enrichment_factor, calculate_multiple_enrichment_factors,
	calculate_ground_truth_enrichment_factors,
	# NEW IMPORTS
	calculate_multiple_top_k_discovery_rates,
	calculate_cumulative_enrichment_factor,
	calculate_batch_hit_rate,
	calculate_batch_enrichment_factor,
	calculate_average_score_ratio,
	calculate_batch_average_score_ratio,
	calculate_multiple_unlabeled_top_k_overlaps,
	calculate_multiple_unlabeled_enrichment_factors,
	calculate_unlabeled_ranking_correlation,
)
from .metrics.similarity import (
	calculate_molecular_similarity_metrics
)

logger = logging.getLogger(__name__)


def evaluate_cycle(
	cycle: int,
	predictions: np.ndarray,
	ground_truth: np.ndarray,
	labeled_data: pl.DataFrame,
	selected_compounds: pl.DataFrame,
	target_col: str,
	oracle_type: str = 'auto',
	ground_truth_data: Optional[pl.DataFrame] = None,
	pool_predictions: Optional[np.ndarray] = None,
	pool_ids: Optional[np.ndarray] = None,
	uncertainties: Optional[np.ndarray] = None,
	previously_selected: Optional[pl.DataFrame] = None,
	advanced_metrics: bool = False,
	disable_molecular_similarity: bool = False,
	score_direction: str = 'higher',
	cumulative_selected_ids: Optional[set] = None,
	cumulative_labeled_count: Optional[int] = None
) -> Dict[str, Any]:
	"""
	Comprehensive evaluation with adaptive metrics based on available data.

	Always calculates:
	- Selection quality metrics (avg_score_selected, batch_size, cumulative_labeled)
	- Molecular similarity metrics (when SMILES available and not disabled)
	- Progress tracking metrics

	Adaptively adds (benchmark mode only):
	- Discovery metrics (Top-K discovery rates, enrichment factors, score ratios)
	- Unlabeled ranking metrics (unlabeled Top-K overlap, unlabeled EF, unlabeled Spearman)
	- Uncertainty metrics (when available)
	
	Args:
		cycle: Current cycle number
		predictions: Model predictions on labeled data
		ground_truth: True values for labeled data
		labeled_data: Currently labeled compounds DataFrame
		selected_compounds: Newly selected compounds this cycle
		target_col: Name of target property column
		oracle_type: 'benchmark', 'run', or 'auto' for auto-detection
		ground_truth_data: Full ground truth data (benchmark mode)
		pool_predictions: Predictions on unlabeled pool
		pool_ids: IDs corresponding to pool predictions
		uncertainties: Uncertainty estimates (if available)
		previously_selected: Previously selected compounds for molecular analysis
		advanced_metrics: Include additional metrics (MAPE, etc.)
		disable_molecular_similarity: Skip expensive molecular calculations
		score_direction: Direction of score optimization ('higher' or 'lower' is better)
		cumulative_labeled_count: Total count of labeled compounds (if None, uses len(labeled_data))
		
	Returns:
		Dictionary containing all calculated metrics
	"""
	metrics = {}

	# Basic cycle info
	metrics['cycle'] = cycle
	metrics['batch_size'] = len(selected_compounds)

	if cumulative_labeled_count is not None:
		metrics['cumulative_labeled'] = cumulative_labeled_count
	else:
		metrics['cumulative_labeled'] = len(labeled_data)
	
	# Selection quality (scores of selected compounds this cycle)
	if target_col in selected_compounds.columns and len(selected_compounds) > 0:
		scores = selected_compounds.get_column(target_col).to_numpy()
		if len(scores) > 0 and not all(np.isnan(scores)):
			metrics['avg_score_selected'] = calculate_average_score(scores)
		else:
			metrics['avg_score_selected'] = None
	else:
		metrics['avg_score_selected'] = None

	# Ground truth average score (if available)
	if ground_truth_data is not None and target_col in ground_truth_data.columns:
		try:
			gt_scores = ground_truth_data.get_column(target_col).to_numpy()
			metrics['ground_truth_avg_score'] = calculate_average_score(gt_scores)
		except Exception as e:
			logger.warning(f"Error calculating ground truth average score: {e}")
			metrics['ground_truth_avg_score'] = None
	else:
		metrics['ground_truth_avg_score'] = None
	
	# Uncertainty metrics (when available)
	if uncertainties is not None and len(uncertainties) > 0:
		try:
			metrics['uncertainty_mean'] = float(np.mean(uncertainties))
			metrics['uncertainty_std'] = float(np.std(uncertainties))
		except Exception as e:
			logger.warning(f"Error calculating uncertainty metrics: {e}")
			metrics['uncertainty_mean'] = None
			metrics['uncertainty_std'] = None
	else:
		metrics['uncertainty_mean'] = None
		metrics['uncertainty_std'] = None
	
	# Molecular similarity metrics (when SMILES available and not disabled)
	if not disable_molecular_similarity and 'SMILES' in selected_compounds.columns:
		try:
			molecular_metrics = calculate_molecular_similarity_metrics(
				newly_selected_df=selected_compounds,
				previously_selected_df=previously_selected
			)
			metrics.update(molecular_metrics)
		except Exception as e:
			logger.warning(f"Error calculating molecular similarity metrics: {e}")
			metrics.update({
				'intra_batch_diversity': None,
				'inter_cycle_similarity': None,
				'batch_novelty_score': None
			})
	else:
		metrics.update({
			'intra_batch_diversity': None,
			'inter_cycle_similarity': None,
			'batch_novelty_score': None
		})

	# Benchmark mode specific metrics (when ground truth available)
	# Auto mode: enable benchmark mode when ground_truth_data and pool_predictions are available
	is_benchmark_mode = (oracle_type == 'benchmark' or
						(oracle_type == 'auto' and ground_truth_data is not None and pool_predictions is not None))

	# Discovery Metrics (Category A) - PRIMARY in benchmark mode
	if is_benchmark_mode and ground_truth_data is not None and cumulative_selected_ids is not None:
		try:
			# Top-K Discovery Rates (6 values)
			discovery_rates = calculate_multiple_top_k_discovery_rates(
				cumulative_selected_ids,
				ground_truth_data,
				target_col,
				score_direction
			)
			metrics.update(discovery_rates)

			# Cumulative Enrichment Factor (if Activity present)
			cumulative_ef = calculate_cumulative_enrichment_factor(
				cumulative_selected_ids,
				ground_truth_data,
				'Activity'
			)
			metrics['cumulative_ef'] = cumulative_ef

			# Batch Hit Rate (if Activity present)
			batch_hit_rate = calculate_batch_hit_rate(
				selected_compounds,
				'Activity'
			)
			metrics['batch_hit_rate'] = batch_hit_rate

			# Batch Enrichment Factor (if Activity present)
			batch_ef = calculate_batch_enrichment_factor(
				selected_compounds,
				ground_truth_data,
				'Activity'
			)
			metrics['batch_ef'] = batch_ef

			# Average Score Ratios (always available with continuous data)
			cumulative_score_ratio = calculate_average_score_ratio(
				cumulative_selected_ids,
				ground_truth_data,
				target_col,
				score_direction
			)
			metrics['cumulative_avg_score_ratio'] = cumulative_score_ratio

			batch_avg_score_ratio = calculate_batch_average_score_ratio(
				selected_compounds,
				ground_truth_data,
				target_col,
				score_direction
			)
			metrics['batch_avg_score_ratio'] = batch_avg_score_ratio

		except Exception as e:
			logger.warning(f"Error calculating discovery metrics: {e}")
			metrics.update({
				'top_10_discovery': None,
				'top_100_discovery': None,
				'top_1000_discovery': None,
				'top_0_1_pct_discovery': None,
				'top_1_pct_discovery': None,
				'top_10_pct_discovery': None,
				'cumulative_ef': None,
				'batch_hit_rate': None,
				'batch_ef': None,
				'cumulative_avg_score_ratio': None,
				'batch_avg_score_ratio': None
			})
	
	# Ranking Metrics (Category B) - UNLABELED ONLY
	if is_benchmark_mode:
		try:
			# CRITICAL: Filter predictions to exclude labeled compounds
			if pool_predictions is not None and pool_ids is not None and cumulative_selected_ids is not None:
				# Create unlabeled predictions DataFrame (EXCLUDE labeled)
				unlabeled_mask = ~np.isin(pool_ids, list(cumulative_selected_ids))
				unlabeled_predictions_df = pl.DataFrame({
					'ID': pool_ids[unlabeled_mask],
					'prediction': pool_predictions[unlabeled_mask]
				})

				if len(unlabeled_predictions_df) > 0:
					# Calculate unlabeled ranking overlaps
					unlabeled_top_k_metrics = calculate_multiple_unlabeled_top_k_overlaps(
						unlabeled_predictions_df=unlabeled_predictions_df,
						ground_truth_df=ground_truth_data,
						target_column=target_col,
						score_direction=score_direction
					)
					metrics.update(unlabeled_top_k_metrics)

					# Calculate unlabeled prospective EFs (if Activity present)
					unlabeled_ef_metrics = calculate_multiple_unlabeled_enrichment_factors(
						unlabeled_predictions_df=unlabeled_predictions_df,
						ground_truth_df=ground_truth_data,
						activity_column='Activity',
						score_direction=score_direction
					)
					metrics.update(unlabeled_ef_metrics)

					# Calculate unlabeled ranking correlation
					unlabeled_spearman = calculate_unlabeled_ranking_correlation(
						unlabeled_predictions_df=unlabeled_predictions_df,
						ground_truth_df=ground_truth_data,
						target_column=target_col
					)
					metrics['unlabeled_spearman_correlation'] = unlabeled_spearman
				else:
					# No unlabeled compounds left
					metrics.update({
						'unlabeled_top_100_overlap': None,
						'unlabeled_top_1000_overlap': None,
						'unlabeled_ef_1_0': None,
						'unlabeled_ef_5_0': None,
						'unlabeled_spearman_correlation': None
					})

		except Exception as e:
			logger.warning(f"Error calculating unlabeled ranking metrics: {e}")
			metrics.update({
				'unlabeled_top_100_overlap': None,
				'unlabeled_top_1000_overlap': None,
				'unlabeled_ef_1_0': None,
				'unlabeled_ef_5_0': None,
				'unlabeled_spearman_correlation': None
			})
	
	# Ground truth enrichment factors (when ground truth data available)
	if ground_truth_data is not None:
		try:
			gt_ef_metrics = calculate_ground_truth_enrichment_factors(
				ground_truth_data, target_col, score_direction
			)
			metrics.update(gt_ef_metrics)
		except Exception as e:
			logger.warning(f"Error calculating ground truth enrichment factors: {e}")
			metrics.update({
				'ground_truth_ef_5_0': None, 'ground_truth_ef_1_0': None,
				'ground_truth_ef_0_5': None, 'ground_truth_ef_0_1': None
			})
	
	# Round numeric values for cleaner output
	for key, value in metrics.items():
		if isinstance(value, float) and not np.isnan(value):
			metrics[key] = round(value, 4)
	
	return metrics


def format_progress_output(metrics: Dict[str, Any], oracle_type: str = 'auto', previous_metrics: Optional[Dict[str, Any]] = None) -> str:
	"""
	Format cycle metrics as a compact rich table for clean console output.
	Automatically adapts formatting for terminal vs Jupyter notebook environments.
	
	Args:
		metrics: Metrics dictionary from evaluate_cycle
		oracle_type: Oracle type for mode-specific formatting
		previous_metrics: Previous cycle metrics for change indicators
		
	Returns:
		Formatted string for console output
	"""
	try:
		from rich.table import Table
		from rich.console import Console
		from io import StringIO
		
		# Import environment utilities with fallback
		try:
			from ..utils.environment import get_console_config, detect_jupyter_environment, format_change_indicator
			console_config = get_console_config()
			in_jupyter = detect_jupyter_environment()
		except ImportError as e:
			logger.warning(f"Could not import environment utilities: {e}. Using default configuration.")
			console_config = {'width': 100, 'force_terminal': True}
			in_jupyter = False
			def format_change_indicator(diff, is_improvement):
				symbol = "↑" if diff > 0 else "↓"
				color = "green" if is_improvement else "red"
				return symbol, color
		
		# Create console that captures output
		string_io = StringIO()
		
		# For Jupyter environments, we need to force string output
		if console_config.get('force_jupyter', False):
			# Override Jupyter detection to get string output for our use case
			console_config_modified = console_config.copy()
			console_config_modified['force_jupyter'] = False
			console_config_modified['force_terminal'] = True
			console = Console(file=string_io, **console_config_modified)
		else:
			console = Console(file=string_io, **console_config)
		
		cycle = metrics.get('cycle', '?')
		batch_size = metrics.get('batch_size', '?')
		
		def format_value_with_change(key: str, current_val: Any, digits: int = 3) -> str:
			"""Format value with change indicator vs previous cycle."""
			if current_val is None:
				return "N/A"
			
			formatted = f"{current_val:.{digits}f}"
			
			# Check if we have previous metrics and can show change
			if (previous_metrics and 
				isinstance(previous_metrics, dict) and 
				key in previous_metrics and 
				previous_metrics[key] is not None):
				
				try:
					prev_val = float(previous_metrics[key])
					current_float = float(current_val)
					diff = current_float - prev_val
					
					# Show change if difference is significant
					threshold = 10**(-digits)
					if abs(diff) > threshold:
						# Determine if change is good or bad (higher is generally better except for RMSE, MAE)
						bad_metrics = {'rmse', 'mae', 'mse', 'inter_cycle_similarity'}
						is_improvement = diff > 0 if key not in bad_metrics else diff < 0
						
						# Get environment-appropriate change indicator
						symbol, color = format_change_indicator(diff, is_improvement)
						
						if in_jupyter and symbol in ['📈', '📉']:
							# Jupyter environment with emoji indicators
							return f"[{color}]{formatted} {symbol}[/{color}]"
						else:
							# Terminal environment with arrow indicators
							return f"[{color}]{formatted}{symbol}[/{color}]"
				except (ValueError, TypeError):
					pass
			
			return formatted
		
		# Create horizontally compact table with 3 columns
		table = Table(show_header=True, header_style="bold white", 
					 title=f"📊 Cycle {cycle} ({batch_size} selected)",
					 title_style="bold white", padding=(0, 1))
		
		# Add 3 columns with headers for logical grouping
		table.add_column("Selection Quality", style="white", width=22)
		table.add_column("Discovery Metrics", style="white", width=24)
		table.add_column("Ranking (Unlabeled)", style="white", width=22)
		
		# Helper to format metric with optional unit and label
		def format_metric_with_label(label: str, key: str, digits: int = 3, unit: str = "") -> str:
			if metrics.get(key) is not None:
				value = format_value_with_change(key, metrics[key], digits) + unit
				return f"{label}: {value}"
			return ""
		
		# Organize metrics into logical groups (now as single strings per column)
		# Column 1: Selection Quality (replaces model performance)
		selection_quality = []
		# Batch selection metrics
		if metrics.get('batch_size') is not None:
			selection_quality.append(f"Batch Size: {metrics['batch_size']}")
		if metrics.get('avg_score_selected') is not None:
			selection_quality.append(format_metric_with_label('Batch Avg', 'avg_score_selected', 3))
		if metrics.get('ground_truth_avg_score') is not None:
			selection_quality.append(format_metric_with_label('GT Avg', 'ground_truth_avg_score', 3))
		if metrics.get('cumulative_labeled') is not None:
			selection_quality.append(f"Total Labeled: {metrics['cumulative_labeled']}")

		# Uncertainty metrics (if available)
		if metrics.get('uncertainty_mean') is not None:
			selection_quality.append(format_metric_with_label('Uncert μ', 'uncertainty_mean', 3))
		if metrics.get('uncertainty_std') is not None:
			selection_quality.append(format_metric_with_label('Uncert σ', 'uncertainty_std', 3))

		# Molecular metrics
		if metrics.get('intra_batch_diversity') is not None:
			selection_quality.append(format_metric_with_label('Diversity', 'intra_batch_diversity', 3))
		if metrics.get('batch_novelty_score') is not None:
			selection_quality.append(format_metric_with_label('Novelty', 'batch_novelty_score', 3))

		# Column 2: Discovery Metrics
		discovery = []
		if oracle_type == 'benchmark':
			# Top-K Discovery Rates
			for key, name in [('top_10_discovery', 'Top-10'), ('top_100_discovery', 'Top-100'),
							  ('top_1000_discovery', 'Top-1K'), ('top_1_pct_discovery', 'Top-1%')]:
				if metrics.get(key) is not None:
					discovery.append(format_metric_with_label(name, key, 1, '%'))

			# Batch Quality
			if metrics.get('batch_hit_rate') is not None:
				discovery.append(format_metric_with_label('Batch HR', 'batch_hit_rate', 3))
			if metrics.get('batch_ef') is not None:
				discovery.append(format_metric_with_label('Batch EF', 'batch_ef', 2))
			if metrics.get('batch_avg_score_ratio') is not None:
				discovery.append(format_metric_with_label('Batch Ratio', 'batch_avg_score_ratio', 2))

			# Cumulative Quality
			if metrics.get('cumulative_ef') is not None:
				discovery.append(format_metric_with_label('Cumul EF', 'cumulative_ef', 2))
			if metrics.get('cumulative_avg_score_ratio') is not None:
				discovery.append(format_metric_with_label('Cumul Ratio', 'cumulative_avg_score_ratio', 2))

		# Column 3: Ranking (Unlabeled Only)
		ranking = []
		if oracle_type == 'benchmark':
			# Unlabeled ranking metrics
			for key, name in [('unlabeled_top_100_overlap', 'U-Top100'),
							  ('unlabeled_top_1000_overlap', 'U-Top1K')]:
				if metrics.get(key) is not None:
					ranking.append(format_metric_with_label(name, key, 1, '%'))

			for key, name in [('unlabeled_ef_1_0', 'U-EF@1%'),
							  ('unlabeled_ef_5_0', 'U-EF@5%')]:
				if metrics.get(key) is not None:
					ranking.append(format_metric_with_label(name, key, 2))

			if metrics.get('unlabeled_spearman_correlation') is not None:
				ranking.append(format_metric_with_label('U-Spear', 'unlabeled_spearman_correlation', 3))

			# Ground truth EFs (still valid)
			for key, name in [('ground_truth_ef_5_0', 'GT@5%'), ('ground_truth_ef_1_0', 'GT@1%')]:
				if metrics.get(key) is not None:
					ranking.append(format_metric_with_label(name, key, 2, 'x'))

		# Find max length for rows
		max_rows = max(len(selection_quality), len(discovery), len(ranking))
		
		# Pad lists to same length
		while len(selection_quality) < max_rows:
			selection_quality.append("")
		while len(discovery) < max_rows:
			discovery.append("")
		while len(ranking) < max_rows:
			ranking.append("")

		# Add rows
		for i in range(max_rows):
			table.add_row(
				selection_quality[i],
				discovery[i],
				ranking[i]
			)
		
		console.print(table)
		
		# Get the rendered output
		output = string_io.getvalue()
		string_io.close()
		
		return output
	except ImportError:
		# Show rich error if rich is not installed
		logger.error("Rich library not installed. Please install it to see formatted output.")
		return ""
	except Exception as e:
		# Fallback for any other errors
		logger.error(f"Error formatting progress output: {e}")
		# Provide basic text fallback
		cycle = metrics.get('cycle', '?')
		batch_size = metrics.get('batch_size', '?')
		avg_score = metrics.get('avg_score_selected')

		fallback = f"Cycle {cycle} ({batch_size} selected)"
		if avg_score is not None:
			fallback += f" | Avg Score: {avg_score:.3f}"

		return fallback


def export_metrics_csv(all_cycle_metrics: list, output_path: str, oracle_type: str = 'auto', score_direction: str = 'higher', target_col: str = 'Activity') -> None:
	"""
	Export enhanced cycle metrics to CSV file with metadata and mode-specific organization.

	Args:
		all_cycle_metrics: List of cycle metrics dictionaries
		output_path: Path to output CSV file
		oracle_type: Oracle type ('run', 'benchmark', 'auto')
		score_direction: Score direction ('higher' or 'lower')
		target_col: Target property column name
	"""
	if not all_cycle_metrics:
		logger.warning("No metrics to export")
		return

	try:
		metrics_df = pl.DataFrame(all_cycle_metrics)
		
		# Add metadata as comments at the top
		with open(output_path, 'w') as f:
			f.write("# LearnM8 Active Learning Cycle Metrics\n")
			f.write(f"# Oracle Type: {oracle_type}\n")
			f.write(f"# Target Column: {target_col}\n")
			f.write(f"# Score Direction: {score_direction} is better\n")
			f.write(f"# Total Cycles: {len(all_cycle_metrics)}\n")
			
			# Add uncertainty availability info
			has_uncertainty = any(m.get('uncertainty_mean') is not None for m in all_cycle_metrics)
			f.write(f"# Uncertainty Available: {has_uncertainty}\n")
			
			# Add mode-specific info
			if oracle_type == 'benchmark':
				has_topk = any('top_k_overlap_100' in m for m in all_cycle_metrics)
				has_ef = any('enrichment_factor_5_percent' in m for m in all_cycle_metrics)
				f.write(f"# Benchmark Metrics Available: Top-K={has_topk}, EF={has_ef}\n")
			f.write("#\n")
		
		# Organize columns by category for better readability
		core_cols = ['cycle', 'strategy', 'batch_fraction', 'selected_count', 'remaining_pool', 'cumulative_labeled', 'batch_size']
		selection_cols = [col for col in metrics_df.columns if col in ['avg_score_selected', 'ground_truth_avg_score', 'uncertainty_mean', 'uncertainty_std']]
		molecular_cols = [col for col in metrics_df.columns if 'diversity' in col or 'similarity' in col or 'novelty' in col]
		discovery_cols = [col for col in metrics_df.columns if 'discovery' in col or 'cumulative_ef' in col or 'batch_ef' in col or 'batch_hit_rate' in col or 'score_ratio' in col]
		unlabeled_ranking_cols = [col for col in metrics_df.columns if 'unlabeled_' in col]
		ground_truth_cols = [col for col in metrics_df.columns if 'ground_truth_ef' in col]
		other_cols = [col for col in metrics_df.columns if col not in core_cols + selection_cols + molecular_cols + discovery_cols + unlabeled_ranking_cols + ground_truth_cols]

		# Reorder columns for logical grouping
		ordered_cols = core_cols + selection_cols + molecular_cols + discovery_cols + unlabeled_ranking_cols + ground_truth_cols + other_cols
		metrics_df = metrics_df.select([col for col in ordered_cols if col in metrics_df.columns])

		# Append the DataFrame (without header since we added metadata)
		# Note: Polars write_csv doesn't support append mode, so we'll use manual approach
		csv_content = metrics_df.write_csv(include_header=True)
		with open(output_path, 'a') as f:
			f.write(csv_content)
		
		logger.info(f"Exported {len(all_cycle_metrics)} cycles of enhanced metrics to {output_path}")
	except Exception as e:
		logger.error(f"Error exporting enhanced metrics to CSV: {e}")
		# Fallback to basic export
		try:
			metrics_df = pl.DataFrame(all_cycle_metrics)
			metrics_df.write_csv(output_path)
			logger.info(f"Exported {len(all_cycle_metrics)} cycles of basic metrics to {output_path}")
		except Exception as fallback_e:
			logger.error(f"Fallback CSV export also failed: {fallback_e}")
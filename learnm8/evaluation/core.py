"""Core evaluation functions for LearnM8 active learning.

This module contains the main evaluation logic that intelligently adapts
metrics based on mode (benchmark vs run) and data availability.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Direct sklearn imports for basic metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import specialized metric functions from modular metrics package
from .metrics.performance import (
	calculate_spearman_correlation, calculate_average_score, calculate_mape
)
from .metrics.enrichment import (
	calculate_top_k_overlap, calculate_multiple_top_k_overlaps,
	calculate_enrichment_factor, calculate_multiple_enrichment_factors,
	calculate_ground_truth_enrichment_factors
)
from .metrics.similarity import (
	calculate_molecular_similarity_metrics
)

logger = logging.getLogger(__name__)


def evaluate_cycle(
	cycle: int,
	predictions: np.ndarray,
	ground_truth: np.ndarray,
	labeled_data: pd.DataFrame,
	selected_compounds: pd.DataFrame,
	target_column: str,
	oracle_type: str = 'auto',
	ground_truth_data: Optional[pd.DataFrame] = None,
	pool_predictions: Optional[np.ndarray] = None,
	pool_ids: Optional[np.ndarray] = None,
	uncertainties: Optional[np.ndarray] = None,
	previously_selected: Optional[pd.DataFrame] = None,
	advanced_metrics: bool = False,
	disable_molecular_similarity: bool = False,
	score_direction: str = 'higher'
) -> Dict[str, Any]:
	"""
	Comprehensive evaluation with adaptive metrics based on available data.
	
	Always calculates:
	- All model performance metrics (RMSE, MAE, MSE, R²)
	- Molecular similarity metrics (when SMILES available and not disabled)
	- Progress tracking metrics
	
	Adaptively adds:
	- Top-K overlaps (benchmark mode with ground truth)
	- Enrichment factors (Activity column present)
	- Uncertainty metrics (when available)
	
	Args:
		cycle: Current cycle number
		predictions: Model predictions on labeled data
		ground_truth: True values for labeled data
		labeled_data: Currently labeled compounds DataFrame
		selected_compounds: Newly selected compounds this cycle
		target_column: Name of target property column
		oracle_type: 'benchmark', 'run', or 'auto' for auto-detection
		ground_truth_data: Full ground truth data (benchmark mode)
		pool_predictions: Predictions on unlabeled pool
		pool_ids: IDs corresponding to pool predictions
		uncertainties: Uncertainty estimates (if available)
		previously_selected: Previously selected compounds for molecular analysis
		advanced_metrics: Include additional metrics (MAPE, etc.)
		disable_molecular_similarity: Skip expensive molecular calculations
		score_direction: Direction of score optimization ('higher' or 'lower' is better)
		
	Returns:
		Dictionary containing all calculated metrics
	"""
	metrics = {}
	
	# Basic cycle info
	metrics['cycle'] = cycle
	metrics['batch_size'] = len(selected_compounds)
	metrics['cumulative_labeled'] = len(labeled_data)
	
	# Core model performance metrics (always calculated)
	try:
		metrics['rmse'] = np.sqrt(mean_squared_error(ground_truth, predictions))
		metrics['mae'] = mean_absolute_error(ground_truth, predictions)
		metrics['mse'] = mean_squared_error(ground_truth, predictions)
		metrics['r2_score'] = r2_score(ground_truth, predictions)
		
		# Spearman correlation (good for ranking quality)
		metrics['spearman_correlation'] = calculate_spearman_correlation(ground_truth, predictions)
		
		# Selection quality
		if target_column in selected_compounds.columns and len(selected_compounds) > 0:
			scores = selected_compounds[target_column].values
			if len(scores) > 0 and not all(pd.isna(scores)):
				metrics['avg_score_selected'] = calculate_average_score(scores)
			else:
				metrics['avg_score_selected'] = None
		else:
			metrics['avg_score_selected'] = None
		
		# Ground truth average score (if available)
		if ground_truth_data is not None and target_column in ground_truth_data.columns:
			try:
				gt_scores = ground_truth_data[target_column].values
				metrics['ground_truth_avg_score'] = calculate_average_score(gt_scores)
			except Exception as e:
				logger.warning(f"Error calculating ground truth average score: {e}")
				metrics['ground_truth_avg_score'] = None
		else:
			metrics['ground_truth_avg_score'] = None
			
		if advanced_metrics:
			metrics['mape'] = calculate_mape(ground_truth, predictions)
			
	except Exception as e:
		logger.warning(f"Error calculating core metrics: {e}")
		# Set default values
		metrics.update({
			'rmse': None, 'mae': None, 'mse': None, 'r2_score': None,
			'spearman_correlation': None, 'avg_score_selected': None, 'ground_truth_avg_score': None
		})
	
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
	
	if is_benchmark_mode:
		try:
			# Create predictions DataFrame for top-k calculations
			if pool_predictions is not None and pool_ids is not None:
				predictions_df = pd.DataFrame({
					'ID': pool_ids,
					'prediction': pool_predictions
				})
				
				# Calculate multiple top-K overlaps
				top_k_metrics = calculate_multiple_top_k_overlaps(
					predictions_df=predictions_df,
					ground_truth_df=ground_truth_data,
					target_column=target_column,
					score_direction=score_direction
				)
				metrics.update(top_k_metrics)
			
		except Exception as e:
			logger.warning(f"Error calculating top-K overlap metrics: {e}")
			metrics.update({
				'top_100_overlap': None,
				'top_1000_overlap': None,
				'top_0_1_percent_overlap': None,
				'top_1_percent_overlap': None,
				'top_10_percent_overlap': None
			})
	
	# Enrichment factors (when Activity column is available)
	if ground_truth_data is not None and 'Activity' in ground_truth_data.columns:
		try:
			# For enrichment factors, use pool predictions vs Activity labels
			if pool_predictions is not None and pool_ids is not None:
				# Merge predictions with ground truth Activity labels
				pred_with_labels = pd.merge(
					pd.DataFrame({'ID': pool_ids, 'prediction': pool_predictions}),
					ground_truth_data[['ID', 'Activity']],
					on='ID'
				)
				
				if len(pred_with_labels) > 0:
					ef_metrics = calculate_multiple_enrichment_factors(
						scores=pred_with_labels['prediction'].values,
						labels=pred_with_labels['Activity'].values,
						score_direction=score_direction
					)
					metrics.update(ef_metrics)
		except Exception as e:
			logger.warning(f"Error calculating enrichment factors: {e}")
			metrics.update({
				'ef_5_0': None, 'ef_1_0': None, 'ef_0_5': None, 'ef_0_1': None
			})
	
	# Ground truth enrichment factors (when ground truth data available)
	if ground_truth_data is not None:
		try:
			gt_ef_metrics = calculate_ground_truth_enrichment_factors(
				ground_truth_data, target_column, score_direction
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
		table.add_column("Model Performance", style="white", width=18)
		table.add_column("Selection Quality", style="white", width=18)
		table.add_column("Benchmark Metrics", style="white", width=18)
		
		# Helper to format metric with optional unit and label
		def format_metric_with_label(label: str, key: str, digits: int = 3, unit: str = "") -> str:
			if metrics.get(key) is not None:
				value = format_value_with_change(key, metrics[key], digits) + unit
				return f"{label}: {value}"
			return ""
		
		# Organize metrics into logical groups (now as single strings per column)
		# Column 1: Model Performance
		model_perf = []
		if metrics.get('r2_score') is not None:
			model_perf.append(format_metric_with_label('R²', 'r2_score'))
		if metrics.get('rmse') is not None:
			model_perf.append(format_metric_with_label('RMSE', 'rmse'))
		if metrics.get('mae') is not None:
			model_perf.append(format_metric_with_label('MAE', 'mae'))
		if metrics.get('mse') is not None:
			model_perf.append(format_metric_with_label('MSE', 'mse'))
		if metrics.get('spearman_correlation') is not None:
			model_perf.append(format_metric_with_label('Spearman', 'spearman_correlation'))
		if metrics.get('mape') is not None:
			model_perf.append(format_metric_with_label('MAPE', 'mape', 1, '%'))
		if metrics.get('uncertainty_mean') is not None:
			model_perf.append(format_metric_with_label('Uncert μ', 'uncertainty_mean'))
		if metrics.get('uncertainty_std') is not None:
			model_perf.append(format_metric_with_label('Uncert σ', 'uncertainty_std'))
		
		# Column 2: Selection Quality  
		selection_qual = []
		if metrics.get('avg_score_selected') is not None:
			selection_qual.append(format_metric_with_label('Avg Sel', 'avg_score_selected'))
		if metrics.get('ground_truth_avg_score') is not None:
			selection_qual.append(format_metric_with_label('GT Avg', 'ground_truth_avg_score'))
		if metrics.get('cumulative_labeled') is not None:
			selection_qual.append(f"Labeled: {metrics['cumulative_labeled']}")
		if metrics.get('intra_batch_diversity') is not None:
			selection_qual.append(format_metric_with_label('Diversity', 'intra_batch_diversity'))
		if metrics.get('inter_cycle_similarity') is not None:
			selection_qual.append(format_metric_with_label('Similarity', 'inter_cycle_similarity'))
		if metrics.get('batch_novelty_score') is not None:
			selection_qual.append(format_metric_with_label('Novelty', 'batch_novelty_score'))
		
		# Column 3: Benchmarks (Top-K & EFs)
		benchmarks = []
		if oracle_type == 'benchmark':
			# All Top-K overlaps
			for key, name in [('top_100_overlap', 'Top-100'), ('top_1000_overlap', 'Top-1K'), 
							  ('top_0_1_percent_overlap', 'Top-0.1%'), ('top_1_percent_overlap', 'Top-1%'), 
							  ('top_10_percent_overlap', 'Top-10%')]:
				if metrics.get(key) is not None:
					benchmarks.append(format_metric_with_label(name, key, 1, '%'))
			
			# All current EFs
			for key, name in [('ef_5_0', 'EF@5%'), ('ef_1_0', 'EF@1%'), 
							  ('ef_0_5', 'EF@0.5%'), ('ef_0_1', 'EF@0.1%')]:
				if metrics.get(key) is not None:
					benchmarks.append(format_metric_with_label(name, key, 2, 'x'))
			
			# All ground truth EFs
			for key, name in [('ground_truth_ef_5_0', 'GT@5%'), ('ground_truth_ef_1_0', 'GT@1%'), 
							  ('ground_truth_ef_0_5', 'GT@0.5%'), ('ground_truth_ef_0_1', 'GT@0.1%')]:
				if metrics.get(key) is not None:
					benchmarks.append(format_metric_with_label(name, key, 2, 'x'))
		
		# Find max length for rows
		max_rows = max(len(model_perf), len(selection_qual), len(benchmarks))
		
		# Pad lists to same length
		while len(model_perf) < max_rows:
			model_perf.append("")
		while len(selection_qual) < max_rows:
			selection_qual.append("")
		while len(benchmarks) < max_rows:
			benchmarks.append("")
		
		# Add rows
		for i in range(max_rows):
			table.add_row(
				model_perf[i],
				selection_qual[i], 
				benchmarks[i]
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
		r2 = metrics.get('r2_score')
		rmse = metrics.get('rmse')
		
		fallback = f"Cycle {cycle} ({batch_size} selected)"
		if r2 is not None:
			fallback += f" | R²: {r2:.3f}"
		if rmse is not None:
			fallback += f" | RMSE: {rmse:.3f}"
		
		return fallback


def export_metrics_csv(all_cycle_metrics: list, output_path: str, oracle_type: str = 'auto', score_direction: str = 'higher', target_column: str = 'Activity') -> None:
	"""
	Export enhanced cycle metrics to CSV file with metadata and mode-specific organization.
	
	Args:
		all_cycle_metrics: List of cycle metrics dictionaries  
		output_path: Path to output CSV file
		oracle_type: Oracle type ('run', 'benchmark', 'auto')
		score_direction: Score direction ('higher' or 'lower')
		target_column: Target property column name
	"""
	if not all_cycle_metrics:
		logger.warning("No metrics to export")
		return
	
	try:
		metrics_df = pd.DataFrame(all_cycle_metrics)
		
		# Add metadata as comments at the top
		with open(output_path, 'w') as f:
			f.write("# LearnM8 Active Learning Cycle Metrics\n")
			f.write(f"# Oracle Type: {oracle_type}\n")
			f.write(f"# Target Column: {target_column}\n")
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
		core_cols = ['cycle', 'strategy', 'batch_fraction', 'selected_count', 'remaining_pool', 'cumulative_labeled']
		model_cols = [col for col in metrics_df.columns if col in ['r2_score', 'rmse', 'mae', 'mse', 'spearman_correlation', 'mape', 'uncertainty_mean', 'uncertainty_std']]
		selection_cols = [col for col in metrics_df.columns if col in ['prediction_mean', 'prediction_std', 'measured_mean', 'measured_std', 'measured_min', 'measured_max', 'avg_score_selected', 'ground_truth_avg_score']]
		molecular_cols = [col for col in metrics_df.columns if 'diversity' in col or 'similarity' in col or 'novelty' in col]
		benchmark_cols = [col for col in metrics_df.columns if 'top_k_overlap' in col or 'enrichment_factor' in col or 'ground_truth_ef' in col]
		other_cols = [col for col in metrics_df.columns if col not in core_cols + model_cols + selection_cols + molecular_cols + benchmark_cols]
		
		# Reorder columns for logical grouping
		ordered_cols = core_cols + model_cols + selection_cols + molecular_cols + benchmark_cols + other_cols
		metrics_df = metrics_df[[col for col in ordered_cols if col in metrics_df.columns]]
		
		# Append the DataFrame (without header since we added metadata)
		metrics_df.to_csv(output_path, mode='a', index=False)
		
		logger.info(f"Exported {len(all_cycle_metrics)} cycles of enhanced metrics to {output_path}")
	except Exception as e:
		logger.error(f"Error exporting enhanced metrics to CSV: {e}")
		# Fallback to basic export
		try:
			metrics_df = pd.DataFrame(all_cycle_metrics)
			metrics_df.to_csv(output_path, index=False)
			logger.info(f"Exported {len(all_cycle_metrics)} cycles of basic metrics to {output_path}")
		except Exception as fallback_e:
			logger.error(f"Fallback CSV export also failed: {fallback_e}")
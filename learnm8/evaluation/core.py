"""Core evaluation functions for LearnM8 active learning.

This module contains the main evaluation logic that intelligently adapts
metrics based on mode (benchmark vs run) and data availability.
"""

import logging
from typing import Any

import numpy as np
import polars as pl

from .metrics.enrichment import (
	calculate_average_score_ratio,
	calculate_batch_average_score_ratio,
	calculate_batch_enrichment_factor,
	calculate_batch_hit_rate,
	calculate_cumulative_enrichment_factor,
	calculate_ground_truth_enrichment_factors,
	# NEW IMPORTS
	calculate_multiple_top_k_discovery_rates,
	calculate_multiple_unlabeled_enrichment_factors,
	calculate_multiple_unlabeled_top_k_overlaps,
	calculate_unlabeled_ranking_correlation,
)

# Import specialized metric functions from modular metrics package
from .metrics.performance import calculate_average_score
from .metrics.similarity import calculate_molecular_similarity_metrics

logger = logging.getLogger(__name__)


def evaluate_cycle(
	cycle: int,
	predictions: np.ndarray,
	ground_truth: np.ndarray,
	labeled_data: pl.DataFrame,
	selected_compounds: pl.DataFrame,
	target_col: str,
	oracle_type: str = 'auto',
	ground_truth_data: pl.DataFrame | None = None,
	pool_predictions: np.ndarray | None = None,
	pool_ids: np.ndarray | None = None,
	uncertainties: np.ndarray | None = None,
	previously_selected: pl.DataFrame | None = None,
	advanced_metrics: bool = False,
	disable_molecular_similarity: bool = False,
	score_direction: str = 'higher',
	cumulative_selected_ids: set | None = None,
	cumulative_labeled_count: int | None = None
) -> dict[str, Any]:
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
		except (ValueError, TypeError, ZeroDivisionError) as e:
			logger.warning(
				f"Could not calculate ground_truth_avg_score: {e}. "
				f"Setting to None. Check input data quality."
			)
			metrics['ground_truth_avg_score'] = None
	else:
		metrics['ground_truth_avg_score'] = None

	# Uncertainty metrics (when available)
	if uncertainties is not None and len(uncertainties) > 0:
		try:
			metrics['uncertainty_mean'] = float(np.mean(uncertainties))
			metrics['uncertainty_std'] = float(np.std(uncertainties))
		except (ValueError, TypeError) as e:
			logger.warning(
				f"Could not calculate uncertainty metrics: {e}. "
				f"Setting to None. Check input data quality."
			)
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
		except (ValueError, TypeError, RuntimeError) as e:
			logger.warning(
				f"Could not calculate molecular similarity metrics: {e}. "
				f"Setting to None. Check input data quality."
			)
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

		except (ValueError, TypeError, ZeroDivisionError, KeyError) as e:
			logger.warning(
				f"Could not calculate discovery metrics: {e}. "
				f"Setting to None. Check input data quality."
			)
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

		except (ValueError, TypeError, ZeroDivisionError, KeyError) as e:
			logger.warning(
				f"Could not calculate unlabeled ranking metrics: {e}. "
				f"Setting to None. Check input data quality."
			)
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
		except (ValueError, TypeError, ZeroDivisionError) as e:
			logger.warning(
				f"Could not calculate ground_truth_enrichment_factors: {e}. "
				f"Setting to None. Check input data quality."
			)
			metrics.update({
				'ground_truth_ef_5_0': None, 'ground_truth_ef_1_0': None,
				'ground_truth_ef_0_5': None, 'ground_truth_ef_0_1': None
			})

	# Round numeric values for cleaner output
	for key, value in metrics.items():
		if isinstance(value, float) and not np.isnan(value):
			metrics[key] = round(value, 4)

	return metrics


def format_progress_output(metrics: dict[str, Any], oracle_type: str = 'auto', previous_metrics: dict[str, Any] | None = None) -> str:
	"""
	Format cycle metrics as compact plain-text output with change indicators.

	Format (benchmark mode):
	```
	Cycle 9 | 291 selected | 2910 labeled
	───────────────────────────────────────────────────────────
	Discovery │ Top10:80%↑  Top100:87%→  Top1K:92%↑
	          │ Top0.1%:45%↑  Top1%:67%→
	Ranking   │ Spearman:0.67↓  EF@1%:2.3→  EF@5%:1.8↑
	Selection │ Avg:-38.5↑  Div:0.45→  Nov:0.78↑
	```

	Args:
		metrics: Metrics dictionary from evaluate_cycle
		oracle_type: Oracle type for mode-specific formatting
		previous_metrics: Previous cycle metrics for change indicators

	Returns:
		Formatted string for console output (plain text, no ANSI codes)
	"""
	cycle = metrics.get('cycle', '?')
	batch_size = metrics.get('batch_size', '?')
	cumulative_labeled = metrics.get('cumulative_labeled', '?')
	is_benchmark = (oracle_type == 'benchmark')

	# Metrics where lower is better (error metrics, similarity to previous)
	# Note: avg_score_selected follows score_direction which defaults to 'higher is better'
	bad_metrics = {'rmse', 'mae', 'mse', 'inter_cycle_similarity'}

	def get_change_symbol(key: str, current_val: float, is_pct: bool = False) -> str:
		"""Return change indicator symbol: ↑ (improved), ↓ (worsened), → (stagnant), or empty."""
		if previous_metrics is None or key not in previous_metrics:
			return ""

		prev_val = previous_metrics.get(key)
		if prev_val is None or current_val is None:
			return ""

		try:
			diff = float(current_val) - float(prev_val)
		except (ValueError, TypeError):
			return ""

		# Determine stagnation threshold based on metric type
		# For percentage metrics (0-100), use 1.0 as threshold
		# For ratio/score metrics, use 0.01
		if is_pct or 'discovery' in key or 'overlap' in key:
			stagnation_threshold = 1.0
		else:
			stagnation_threshold = 0.01

		# Check for stagnation first
		if abs(diff) < stagnation_threshold:
			return "→"

		# Determine if higher is better for this metric
		is_higher_better = key not in bad_metrics
		is_improvement = (diff > 0) if is_higher_better else (diff < 0)

		return "↑" if is_improvement else "↓"

	def format_metric(key: str, label: str, digits: int = 1, is_pct: bool = False) -> str:
		"""Format a single metric with optional change indicator."""
		val = metrics.get(key)
		if val is None:
			return f"{label}:N/A"

		change = get_change_symbol(key, val, is_pct)
		if is_pct:
			return f"{label}:{val:.{digits}f}%{change}"
		else:
			return f"{label}:{val:.{digits}f}{change}"

	# Build output lines
	lines = []

	# Header line
	lines.append(f"Cycle {cycle} | {batch_size} selected | {cumulative_labeled} labeled")
	lines.append("─" * 60)

	if is_benchmark:
		# Discovery metrics - split into two rows for readability
		discovery_row1 = [
			format_metric('top_10_discovery', 'Top10', 1, True),
			format_metric('top_100_discovery', 'Top100', 1, True),
			format_metric('top_1000_discovery', 'Top1K', 1, True),
		]
		discovery_row2 = [
			format_metric('top_0_1_pct_discovery', 'Top0.1%', 1, True),
			format_metric('top_1_pct_discovery', 'Top1%', 1, True),
		]

		lines.append(f"Discovery │ {' '.join(discovery_row1)}")
		lines.append(f"          │ {' '.join(discovery_row2)}")

		# Ranking metrics
		ranking_parts = [
			format_metric('unlabeled_spearman_correlation', 'Spearman', 3, False),
			format_metric('unlabeled_ef_1_0', 'EF@1%', 2, False),
			format_metric('unlabeled_ef_5_0', 'EF@5%', 2, False),
		]
		lines.append(f"Ranking   │ {' '.join(ranking_parts)}")

	# Selection metrics (always shown)
	selection_parts = [format_metric('avg_score_selected', 'Avg', 2, False)]

	if metrics.get('uncertainty_mean') is not None:
		selection_parts.append(format_metric('uncertainty_mean', 'Unc', 3, False))
	if metrics.get('intra_batch_diversity') is not None:
		selection_parts.append(format_metric('intra_batch_diversity', 'Div', 2, False))
	if metrics.get('batch_novelty_score') is not None:
		selection_parts.append(format_metric('batch_novelty_score', 'Nov', 2, False))

	lines.append(f"Selection │ {' '.join(selection_parts)}")

	return "\n".join(lines)


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
	except (ValueError, TypeError, KeyError, OSError) as e:
		logger.error(f"Error exporting enhanced metrics to CSV: {e}")
		try:
			metrics_df = pl.DataFrame(all_cycle_metrics)
			metrics_df.write_csv(output_path)
			logger.info(f"Exported {len(all_cycle_metrics)} cycles of basic metrics to {output_path}")
		except (ValueError, TypeError, OSError) as fallback_e:
			logger.error(f"Fallback CSV export also failed: {fallback_e}")

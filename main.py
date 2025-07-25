#!/usr/bin/env python3
"""
LearnM8 Active Learning CLI with Integrated Monitoring

This script provides a command-line interface for running active learning experiments
with integrated monitoring using the top_x_of_x metric from scoring_metric.py.
"""

import argparse
import os
import sys
import gc
import traceback
import pandas as pd
from datetime import datetime
from typing import Dict, Any

from active_learning_function import learnM8
from helpers.scoring_metric import top_x_of_x_percentage
from helpers.query_functions import greedy_query_function, random_query_function, cluster_query_function
from learners.random_forest import rf_learner

def get_learner_from_string(learner_string: str):
	"""Convert learner string to learner class."""
	try:
		if learner_string == "rf_learner":
			return rf_learner
		else:
			raise ValueError(f"Unknown learner type: {learner_string}")
	except Exception as e:
		print(f"Error in get_learner_from_string: {e}")
		print(f"Traceback:\n{traceback.format_exc()}")
		raise


def get_query_function_from_string(query_function_string: str):
	"""Convert query function string to query function."""
	try:
		if query_function_string == "greedy":
			return greedy_query_function
		elif query_function_string == "random":
			return random_query_function
		elif query_function_string == "cluster":
			return cluster_query_function
		else:
			raise ValueError(f"Unknown query function: {query_function_string}")
	except Exception as e:
		print(f"Error in get_query_function_from_string: {e}")
		print(f"Traceback:\n{traceback.format_exc()}")
		raise


def validate_csv_file(file_path: str, required_columns: list) -> None:
	"""Validate that CSV file exists and has required columns."""
	try:
		if not os.path.exists(file_path):
			raise FileNotFoundError(f"CSV file not found: {file_path}")
		
		df = pd.read_csv(file_path)
		missing_columns = [col for col in required_columns if col not in df.columns]
		if missing_columns:
			raise ValueError(f"CSV file {file_path} missing required columns: {missing_columns}")
	except Exception as e:
		print(f"Error in validate_csv_file: {e}")
		print(f"Traceback:\n{traceback.format_exc()}")
		raise


def setup_output_directory(output_dir: str) -> str:
	"""Create output directory and return full path."""
	try:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		full_output_dir = os.path.join(output_dir, f"active_learning_{timestamp}")
		os.makedirs(full_output_dir, exist_ok=True)
		return full_output_dir
	except Exception as e:
		print(f"Error in setup_output_directory: {e}")
		print(f"Traceback:\n{traceback.format_exc()}")
		raise


def detect_score_direction(ground_truth_path: str, target_column: str, score_direction: str) -> str:
	"""
	Detect or validate the scoring direction for the target column.
	
	Args:
		ground_truth_path: Path to ground truth CSV
		target_column: Column name to analyze
		score_direction: User-specified direction ('higher', 'lower', 'auto')
		
	Returns:
		'higher' or 'lower' indicating the scoring direction
	"""
	try:
		if score_direction in ['higher', 'lower']:
			return score_direction
		
		# Auto-detect based on column name patterns
		column_lower = target_column.lower()
		
		# Common patterns for lower-is-better scores
		lower_patterns = ['dock', 'binding_energy', 'energy', 'rmsd', 'error', 'loss', 'distance']
		# Common patterns for higher-is-better scores
		higher_patterns = ['activity', 'affinity', 'score', 'rank', 'similarity', 'accuracy']
		
		for pattern in lower_patterns:
			if pattern in column_lower:
				print(f"Auto-detected scoring direction: LOWER is better (found '{pattern}' in column name)")
				return 'lower'
		
		for pattern in higher_patterns:
			if pattern in column_lower:
				print(f"Auto-detected scoring direction: HIGHER is better (found '{pattern}' in column name)")
				return 'higher'
		
		# If no pattern matches, try to analyze the data distribution
		try:
			df = pd.read_csv(ground_truth_path)
			if target_column in df.columns:
				values = df[target_column].dropna()
				if len(values) > 0:
					# Check if most values are negative (common for binding energies/docking scores)
					negative_fraction = (values < 0).mean()
					if negative_fraction > 0.7:
						print(f"Auto-detected scoring direction: LOWER is better (70%+ values are negative)")
						return 'lower'
					else:
						print(f"Auto-detected scoring direction: HIGHER is better (default assumption)")
						return 'higher'
		except Exception as e:
			print(f"Warning: Could not analyze data for direction detection: {e}")
			print(f"Traceback:\n{traceback.format_exc()}")
		
		# Default to higher-is-better if detection fails
		print("Auto-detected scoring direction: HIGHER is better (default assumption)")
		return 'higher'
	except Exception as e:
		print(f"Error in detect_score_direction: {e}")
		print(f"Traceback:\n{traceback.format_exc()}")
		raise


def run_monitored_active_learning(
	compound_pool_csv_path: str,
	ground_truth_path: str,
	target_column: str,
	learner_type: str = "rf_learner",
	batch_size_fraction: float = 0.1,
	cycles: int = 10,
	query_function_type: str = "greedy",
	seed: int = 42,
	top_n: int = 100,
	score_direction: str = "auto",
	output_dir: str = "./results"
) -> Dict[str, Any]:
	"""
	Run active learning with integrated monitoring using top_x_of_x_percentage metric.
	
	Args:
		compound_pool_csv_path: Path to compound pool CSV (must have ID, SMILES columns)
		ground_truth_path: Path to ground truth CSV (must have ID column and target_column)
		target_column: Column name to learn/predict
		learner_type: Type of learner to use ("rf_learner")
		batch_size_fraction: Fraction of compounds to query per cycle
		cycles: Number of active learning cycles
		query_function_type: Query strategy ("greedy", "random", "cluster")
		seed: Random seed for reproducibility
		top_n: Number of top compounds for monitoring metric
		output_dir: Directory to save results
		
	Returns:
		Dictionary containing experiment results and monitoring data
	"""
	try:
		print("Starting LearnM8 Active Learning with Monitoring")
		print(f"Compound Pool: {compound_pool_csv_path}")
		print(f"Ground Truth: {ground_truth_path}")
		print(f"Target Column: {target_column}")
		print(f"Learner: {learner_type}")
		print(f"Batch Size Fraction: {batch_size_fraction}")
		print(f"Cycles: {cycles}")
		print(f"Query Function: {query_function_type}")
		print(f"Seed: {seed}")
		print(f"Top-N for Monitoring: {top_n}")
		
		# Detect scoring direction
		detected_direction = detect_score_direction(ground_truth_path, target_column, score_direction)
		print(f"Score Direction: {detected_direction.upper()} values are better")
		print("-" * 60)
		
		# Setup output directory
		full_output_dir = setup_output_directory(output_dir)
		print(f"Results will be saved to: {full_output_dir}")
		
		# Initialize learner
		learner_class = get_learner_from_string(learner_type)
		learner = learner_class(max_out_system=True)
		learner.set_path(full_output_dir)
		learner.set_score_direction(detected_direction)
		
		# Get query functions
		first_query_function = random_query_function  # Always start with random
		query_function = get_query_function_from_string(query_function_type)
		
		# Initialize monitoring data storage (for future cycle-by-cycle monitoring)
		# monitoring_data = []  # Currently unused but kept for future enhancement
		
		# Set up parameters for learnM8
		learn_params = {
			'learner': learner,
			'compound_pool_csv_path': compound_pool_csv_path,
			'ground_truth_path': ground_truth_path,
			'target_column': target_column,
			'batch_size_fraction': batch_size_fraction,
			'cycles': cycles,
			'first_query_function': first_query_function,
			'query_function': query_function,
			'seed': seed
		}
		
		# Save experiment parameters
		try:
			params_file = os.path.join(full_output_dir, "experiment_parameters.txt")
			with open(params_file, 'w') as f:
				f.write("LearnM8 Active Learning Experiment Parameters\n")
				f.write("=" * 50 + "\n")
				for key, value in learn_params.items():
					if key == 'learner':
						f.write(f"{key}: {type(value).__name__}\n")
					elif callable(value):
						f.write(f"{key}: {value.__name__}\n")
					else:
						f.write(f"{key}: {value}\n")
				f.write(f"top_n_monitoring: {top_n}\n")
				f.write(f"output_directory: {full_output_dir}\n")
				f.write(f"timestamp: {datetime.now().isoformat()}\n")
			
			print(f"Experiment parameters saved to: {params_file}")
		except Exception as e:
			print(f"Warning: Could not save experiment parameters: {e}")
			print(f"Traceback:\n{traceback.format_exc()}")
		
		print("Starting active learning...")
		
		try:
			# Run the active learning
			success = learnM8(**learn_params)
			
			if not success:
				raise RuntimeError("Active learning failed")
			
			print("Active learning completed successfully!")
			
			# After active learning, calculate monitoring metrics for each cycle
			try:
				cache_file = os.path.join(full_output_dir, "cache.csv")
				if os.path.exists(cache_file):
					# Load predictions from cache
					cache_df = pd.read_csv(cache_file)
					
					# Get all cycle columns
					cycle_columns = [col for col in cache_df.columns if col.startswith('cycle_')]
					cycle_columns.sort(key=lambda x: int(x.split('_')[1]))  # Sort by cycle number
					
					if cycle_columns:
						print(f"\nCycle-by-Cycle Monitoring Results:")
						print("=" * 70)
						print(f"{'Cycle':<6} {'Top-' + str(top_n) + ' Overlap':<15} {'Avg Score':<12} {'Trend'}")
						print("-" * 70)
						
						cycle_metrics = []
						
						# Load ground truth for average score calculations
						ground_truth_df = pd.read_csv(ground_truth_path)
						
						# Calculate monitoring metrics for each cycle
						for i, cycle_col in enumerate(cycle_columns):
							cycle_num = int(cycle_col.split('_')[1])
							
							# Create predictions DataFrame for this cycle
							cycle_predictions = pd.DataFrame({
								'ID': cache_df['ID'],
								'estimation': cache_df[cycle_col]
							})
							
							# Calculate top-x overlap metric for this cycle
							try:
								cycle_metric = top_x_of_x_percentage(
									ground_truth_path, 
									cycle_predictions, 
									top_n, 
									target_column,
									detected_direction
								)
							except Exception as e:
								print(f"Cycle {cycle_num:2d}: Error calculating overlap - {e}")
								print(f"Traceback:\n{traceback.format_exc()}")
								cycle_metric = None
							
							# Calculate average score of newly queried compounds for this cycle
							avg_score = None
							trend_indicator = ""
							try:
								actual_batch_size = int(len(cache_df) * batch_size_fraction)
								
								if cycle_num == 0:
									# Cycle 0: Initial random batch - we can't determine this from predictions
									# Skip average score calculation for cycle 0
									avg_score = None
								else:
									# For cycles 1+: Find compounds that would be selected by the query function
									# at this specific cycle (excluding compounds already in training set)
									
									# Simulate which compounds would be available for querying at this cycle
									# (i.e., exclude compounds that were already selected in previous cycles)
									compounds_selected_so_far = (cycle_num - 1) * actual_batch_size
									
									# Sort predictions according to query function logic
									sorted_preds = cycle_predictions.sort_values('estimation', ascending=(detected_direction == 'lower'))
									
									# The compounds that would be selected in this cycle are:
									# positions [compounds_selected_so_far : compounds_selected_so_far + actual_batch_size]
									start_idx = compounds_selected_so_far
									end_idx = start_idx + actual_batch_size
									
									newly_queried_compounds = sorted_preds.iloc[start_idx:end_idx]
								
									# Get ground truth scores for the newly queried compounds
									if len(newly_queried_compounds) > 0:
										merged_data = pd.merge(newly_queried_compounds, ground_truth_df[['ID', target_column]], on='ID', how='inner')
										if len(merged_data) > 0:
											avg_score = merged_data[target_column].mean()
											
											# Calculate trend indicator
											if len(cycle_metrics) > 0 and cycle_metrics[-1]['avg_score'] is not None:
												prev_avg = cycle_metrics[-1]['avg_score']
												if detected_direction == 'lower':
													trend_indicator = "↓" if avg_score < prev_avg else "↑" if avg_score > prev_avg else "→"
												else:
													trend_indicator = "↑" if avg_score > prev_avg else "↓" if avg_score < prev_avg else "→"
										
							except Exception as e:
								print(f"Warning: Could not calculate average score for cycle {cycle_num}: {e}")
								print(f"Traceback:\n{traceback.format_exc()}")
							
							# Store results
							cycle_metrics.append({
								'cycle': cycle_num,
								'top_x_overlap': cycle_metric,
								'avg_score': avg_score
							})
							
							# Print results
							overlap_str = f"{cycle_metric:6.2f}%" if cycle_metric is not None else "Error"
							score_str = f"{avg_score:8.3f}" if avg_score is not None else "N/A"
							print(f"Cycle {cycle_num:<2d}: {overlap_str:<15} {score_str:<12} {trend_indicator}")
							
						print("-" * 70)
						
						# Save cycle-by-cycle monitoring data
						try:
							monitoring_df = pd.DataFrame(cycle_metrics)
							monitoring_file = os.path.join(full_output_dir, "cycle_monitoring.csv")
							monitoring_df.to_csv(monitoring_file, index=False)
							print(f"\nCycle-by-cycle monitoring saved to: {monitoring_file}")
						except Exception as e:
							print(f"Warning: Could not save monitoring data: {e}")
							print(f"Traceback:\n{traceback.format_exc()}")
						
						# Save final predictions (last cycle)
						try:
							final_cycle_col = cycle_columns[-1]
							final_predictions = pd.DataFrame({
								'ID': cache_df['ID'],
								'estimation': cache_df[final_cycle_col]
							})
							predictions_file = os.path.join(full_output_dir, "final_predictions.csv")
							final_predictions.to_csv(predictions_file, index=False)
							print(f"Final predictions saved to: {predictions_file}")
						except Exception as e:
							print(f"Warning: Could not save final predictions: {e}")
							print(f"Traceback:\n{traceback.format_exc()}")
						
						# Save comprehensive monitoring summary
						try:
							summary_file = os.path.join(full_output_dir, "monitoring_summary.txt")
							with open(summary_file, 'w') as f:
								f.write("LearnM8 Active Learning Monitoring Summary\n")
								f.write("=" * 55 + "\n")
								f.write(f"Target Column: {target_column}\n")
								f.write(f"Score Direction: {detected_direction.upper()} values are better\n")
								f.write(f"Top-N Compounds for Monitoring: {top_n}\n")
								f.write(f"Total Compounds Evaluated: {len(cache_df)}\n")
								f.write(f"Number of Cycles: {len(cycle_columns)}\n")
								f.write(f"Batch Size Fraction: {batch_size_fraction}\n")
								f.write(f"Experiment completed at: {datetime.now().isoformat()}\n\n")
								
								f.write("Cycle-by-Cycle Results:\n")
								f.write("-" * 55 + "\n")
								f.write(f"{'Cycle':<6} {'Top-' + str(top_n) + ' Overlap':<15} {'Avg Score':<12}\n")
								f.write("-" * 55 + "\n")
								
								for metric in cycle_metrics:
									overlap_str = f"{metric['top_x_overlap']:6.2f}%" if metric['top_x_overlap'] is not None else "Error"
									score_str = f"{metric['avg_score']:8.3f}" if metric['avg_score'] is not None else "N/A"
									f.write(f"Cycle {metric['cycle']:<2d}: {overlap_str:<15} {score_str:<12}\n")
								
								# Calculate improvements
								valid_overlap_metrics = [m for m in cycle_metrics if m['top_x_overlap'] is not None]
								valid_score_metrics = [m for m in cycle_metrics if m['avg_score'] is not None]
								
								f.write("\nOverall Performance:\n")
								f.write("-" * 25 + "\n")
								
								if len(valid_overlap_metrics) >= 2:
									first_overlap = valid_overlap_metrics[0]['top_x_overlap']
									last_overlap = valid_overlap_metrics[-1]['top_x_overlap']
									overlap_improvement = last_overlap - first_overlap
									f.write(f"Top-{top_n} Overlap Improvement: {overlap_improvement:+.2f}%\n")
								
								if len(valid_score_metrics) >= 2:
									first_score = valid_score_metrics[0]['avg_score']
									last_score = valid_score_metrics[-1]['avg_score']
									score_improvement = last_score - first_score
									direction_indicator = "better" if ((detected_direction == 'lower' and score_improvement < 0) or 
																	 (detected_direction == 'higher' and score_improvement > 0)) else "worse"
									f.write(f"Average Score Change: {score_improvement:+.3f} ({direction_indicator})\n")
							
							print(f"Monitoring summary saved to: {summary_file}")
						except Exception as e:
							print(f"Warning: Could not save monitoring summary: {e}")
							print(f"Traceback:\n{traceback.format_exc()}")
						
						# Return results with final metric
						final_metric = cycle_metrics[-1]['top_x_overlap'] if cycle_metrics else None
						return {
							'success': True,
							'final_metric': final_metric,
							'cycle_metrics': cycle_metrics,
							'predictions_count': len(cache_df),
							'output_directory': full_output_dir
						}
					else:
						print("Warning: No cycle columns found in cache file")
						return {
							'success': True,
							'final_metric': None,
							'cycle_metrics': [],
							'predictions_count': 0,
							'output_directory': full_output_dir
						}
				else:
					print("Warning: Cache file not found for monitoring")
					return {
						'success': True,
						'final_metric': None,
						'cycle_metrics': [],
						'predictions_count': 0,
						'output_directory': full_output_dir
					}
					
			except Exception as e:
				print(f"Warning: Could not calculate monitoring metrics: {e}")
				print(f"Traceback:\n{traceback.format_exc()}")
				return {
					'success': True,
					'final_metric': None,
					'cycle_metrics': [],
					'predictions_count': 0,
					'output_directory': full_output_dir
				}
				
		except Exception as e:
			print(f"Error during active learning: {e}")
			print(f"Traceback:\n{traceback.format_exc()}")
			return {
				'success': False,
				'error': str(e),
				'output_directory': full_output_dir
			}
	except Exception as e:
		print(f"Error in run_monitored_active_learning: {e}")
		print(f"Traceback:\n{traceback.format_exc()}")
		return {
			'success': False,
			'error': str(e),
			'output_directory': output_dir if 'full_output_dir' not in locals() else full_output_dir
		}
	finally:
		# Clean up memory
		gc.collect()


def main():
	"""Main entry point for the CLI."""
	parser = argparse.ArgumentParser(
		description="LearnM8 Active Learning with Integrated Monitoring",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python main.py --compound-pool data/compounds.csv --ground-truth data/ground_truth.csv --target-column Activity
  
  python main.py --compound-pool data/GBA_2v3e_scoring_and_consensus_maxAL.csv \\
				 --ground-truth data/GBA_2v3e_scoring_and_consensus_maxAL.csv \\
				 --target-column ConvexPLR --cycles 5 --batch-size-fraction 0.05
		"""
	)
	
	# Required arguments
	parser.add_argument(
		'--compound-pool', 
		type=str, 
		required=True,
		help='Path to CSV file containing compound pool (must have ID and SMILES columns)'
	)
	
	parser.add_argument(
		'--ground-truth', 
		type=str, 
		required=True,
		help='Path to CSV file containing ground truth data (must have ID column and target column)'
	)
	
	parser.add_argument(
		'--target-column', 
		type=str, 
		required=True,
		help='Name of the column to learn/predict (e.g., Activity, ConvexPLR, CHEMPLP)'
	)
	
	# Optional arguments
	parser.add_argument(
		'--learner', 
		type=str, 
		default='rf_learner',
		choices=['rf_learner'],
		help='Type of learner to use (default: rf_learner)'
	)
	
	parser.add_argument(
		'--batch-size-fraction', 
		type=float, 
		default=0.1,
		help='Fraction of total compounds to query per cycle (default: 0.1)'
	)
	
	parser.add_argument(
		'--cycles', 
		type=int, 
		default=10,
		help='Number of active learning cycles (default: 10, use -1 for single large batch)'
	)
	
	parser.add_argument(
		'--query-function', 
		type=str, 
		default='greedy',
		choices=['greedy', 'random', 'cluster'],
		help='Query strategy for compound selection (default: greedy)'
	)
	
	parser.add_argument(
		'--seed', 
		type=int, 
		default=42,
		help='Random seed for reproducibility (default: 42)'
	)
	
	parser.add_argument(
		'--top-n', 
		type=int, 
		default=100,
		help='Number of top compounds for monitoring metric (default: 100)'
	)
	
	parser.add_argument(
		'--score-direction', 
		type=str, 
		default='auto',
		choices=['higher', 'lower', 'auto'],
		help='Scoring direction: "higher" for higher-is-better, "lower" for lower-is-better, "auto" to detect (default: auto)'
	)
	
	parser.add_argument(
		'--output-dir', 
		type=str, 
		default='./results',
		help='Output directory for results (default: ./results)'
	)
	
	args = parser.parse_args()
	
	# Validate input files
	try:
		validate_csv_file(args.compound_pool, ['ID', 'SMILES'])
		validate_csv_file(args.ground_truth, ['ID', args.target_column])
	except (FileNotFoundError, ValueError) as e:
		print(f"Error: {e}", file=sys.stderr)
		print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr)
		sys.exit(1)
	
	# Run the active learning with monitoring
	try:
		results = run_monitored_active_learning(
			compound_pool_csv_path=args.compound_pool,
			ground_truth_path=args.ground_truth,
			target_column=args.target_column,
			learner_type=args.learner,
			batch_size_fraction=args.batch_size_fraction,
			cycles=args.cycles,
			query_function_type=args.query_function,
			seed=args.seed,
			top_n=args.top_n,
			score_direction=args.score_direction,
			output_dir=args.output_dir
		)
		
		# Print final results
		print("\n" + "=" * 60)
		print("EXPERIMENT COMPLETED")
		print("=" * 60)
		
		if results['success']:
			print("✓ Active learning completed successfully")
			print(f"✓ Results saved to: {results['output_directory']}")
			
			if results['final_metric'] is not None:
				print(f"✓ Final Top-{args.top_n} Overlap: {results['final_metric']:.2f}%")
				print(f"✓ Predictions generated for: {results['predictions_count']} compounds")
				
				# Show cycle progression if available
				if 'cycle_metrics' in results and results['cycle_metrics']:
					valid_metrics = [m for m in results['cycle_metrics'] if m['top_x_overlap'] is not None]
					if len(valid_metrics) >= 2:
						first_metric = valid_metrics[0]['top_x_overlap']
						last_metric = valid_metrics[-1]['top_x_overlap']
						improvement = last_metric - first_metric
						print(f"✓ Overall improvement: {improvement:+.2f}% over {len(valid_metrics)} cycles")
			else:
				print("⚠ Monitoring metric could not be calculated")
			
			sys.exit(0)
		else:
			print(f"✗ Active learning failed: {results['error']}")
			print(f"✗ Partial results may be in: {results['output_directory']}")
			sys.exit(1)
	except Exception as e:
		print(f"Fatal error in main: {e}", file=sys.stderr)
		print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr)
		sys.exit(1)


if __name__ == "__main__":
	main()
"""Example usage of the LearnM8 active learning framework."""

from pathlib import Path
from experiments.runner import create_experiment_config, run_experiment

# Example 1: Basic usage with default parameters
def example_basic():
	"""Run a basic active learning experiment."""
	config = create_experiment_config(
		compound_pool_path="/home/tony/LearnM8/ESSENCE_benchmark_input/ADA.csv",
		ground_truth_path="/home/tony/LearnM8/ESSENCE_benchmark_input/ADA.csv",
		target_column="Activity",
		n_cycles=10,
		batch_size_fraction=0.1
	)
	
	output_dir = Path("results/basic_experiment")
	final_predictions, monitoring_results = run_experiment(config, output_dir)
	
	print(f"Experiment completed. Results saved to {output_dir}")


# Example 2: Docking score optimization (lower is better)
def example_docking():
	"""Optimize for docking scores where lower values are better."""
	config = create_experiment_config(
		compound_pool_path="/home/tony/LearnM8/ESSENCE_benchmark_input/ADA.csv",
		ground_truth_path="/home/tony/LearnM8/ESSENCE_benchmark_input/ADA.csv",
		target_column="CHEMPLP",
		n_cycles=5,
		batch_size_fraction=0.05,
		selection_strategy="greedy",
		score_direction="lower",  # Lower docking scores are better
		top_k=100,
		enrichment_percentile=1.0
	)
	
	output_dir = Path("results/docking_optimization")
	final_predictions, monitoring_results = run_experiment(config, output_dir)


# Example 3: Diversity-based exploration
def example_diversity():
	"""Use diversity-based selection for exploration."""
	config = create_experiment_config(
		compound_pool_path="/home/tony/LearnM8/ESSENCE_benchmark_input/ADA.csv",
		ground_truth_path="/home/tony/LearnM8/ESSENCE_benchmark_input/ADA.csv",
		target_column="Activity",
		n_cycles=10,
		batch_size_fraction=0.1,
		selection_strategy="diverse",
		initial_strategy="diverse",  # Also use diversity for initial selection
		random_state=123
	)
	
	output_dir = Path("results/diversity_experiment")
	final_predictions, monitoring_results = run_experiment(config, output_dir)


# Example 4: Programmatic usage with custom monitoring
def example_custom():
	"""Direct usage of components for custom workflows."""
	from core.active_learning import run_active_learning
	from oracles.csv_oracle import CSVOracle
	from learners.random_forest import RandomForestLearner
	from strategies.greedy import select_greedy
	from strategies.random import select_random
	import pandas as pd
	
	# Load data
	compound_pool = pd.read_csv("/home/tony/LearnM8/ESSENCE_benchmark_input/ADA.csv")
	
	# Create components
	oracle = CSVOracle("/home/tony/LearnM8/ESSENCE_benchmark_input/ADA.csv")
	learner = RandomForestLearner(random_state=42)
	
	# Custom monitoring configuration
	monitoring_config = {
		'top_k': 50,  # Monitor top-50 compounds
		'enrichment_percentile': 5.0  # Calculate EF at 5%
	}
	
	# Run active learning
	final_predictions, monitoring_results = run_active_learning(
		compound_pool=compound_pool,
		oracle=oracle,
		learner=learner,
		target_column="Activity",
		n_cycles=10,
		batch_size=100,  # Fixed batch size instead of fraction
		selection_strategy=select_greedy,
		initial_selection_strategy=select_random,
		score_direction="higher",
		random_state=42,
		output_dir=Path("results/custom_experiment"),
		monitoring_config=monitoring_config
	)
	
	# Analyze results
	for result in monitoring_results:
		print(f"Cycle {result['cycle']}: "
			  f"Top-50 overlap = {result['top_k_overlap']:.1f}%, "
			  f"EF@5% = {result.get('enrichment_factor', 'N/A')}")


# Example 5: Command-line usage
def example_cli():
	"""Show equivalent command-line usage."""
	print("Command-line examples:")
	print()
	
	# Basic usage
	print("# Basic usage:")
	print("python -m cli.main data/compounds.csv data/ground_truth.csv Activity")
	print()
	
	# With custom parameters
	print("# Custom parameters:")
	print("python -m cli.main data/compounds.csv data/ground_truth.csv CHEMPLP \\")
	print("    --cycles 5 \\")
	print("    --batch-size-fraction 0.05 \\")
	print("    --strategy greedy \\")
	print("    --direction lower \\")
	print("    --top-k 100 \\")
	print("    --output results/cli_experiment")


if __name__ == "__main__":
	# Run examples
	print("LearnM8 Example Usage")
	print("=" * 50)
	
	# Uncomment to run examples:
	example_basic()
	# example_docking()
	# example_diversity()
	# example_custom()
	
	example_cli()
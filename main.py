from active_learning_function import active_learning_function
import itertools
import gc

learner="learner"
SMIDS="smids"
GTP="ground_truth_path"

param_combinations = {
    'learner': ['learner1', 'learner2'],
    'hyperparameter_tuning': [True, False],
    'batch_size_percentage': [0.05, 0.1, 0.5, 1],
    'smids_input_path': [SMIDS], #todo 
    'ground_truth_path': [GTP], #todo 
    'cycles': [-1, 10],
    'column_to_learn': ['docking1', 'consensus1'],
    'do_scoring_function_list_prediction': [True, False]
}

# Generate all combinations
keys, values = zip(*param_combinations.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Call the function with each combination
for combo in combinations:
    active_learning_function(**combo)
    gc.collect()
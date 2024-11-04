from active_learning_function import active_learning_function
import itertools
import gc

from helpers.helpers import hash_params
from helpers.query_functions import greedy_query_function, random_query_function
from learners.pipe_cp_learner import chemprop_gpu_learner as cp_learner
from learners.rf_learner import rf_learner as rf_learner
from learners.gp_learner import gp_learner as gp_learner

learner="learner"
SMIDS="smids"
GTP="ground_truth_path"

path_tuples = [("sdf","gttt")]

def get_learner_from_string(learner_string):
    if learner_string == "cp_learner":
        return cp_learner
    elif learner_string == "rf_learner":
        return rf_learner
    elif learner_string == "gp_learner":
        return gp_learner
    else:
        raise ValueError("Invalid learner string")

def get_query_function_from_string(query_function_string):
    if query_function_string == "greedy_query_function":
        return greedy_query_function
    elif query_function_string == "random_query_function":
        return random_query_function

param_combinations = {
    "path_id": [0,1], #this is for each dataset tuple
    'learner': ["cp_learner","rf_learner"],
    'hyperparameter_tuning': [True, False],
    'batch_size_percentage': [0.05, 0.1, 0.5, 1],
    'smids_input_path': None, #!stay
    'ground_truth_path': None, #!stay 
    'cycles': [-1, 10], #-1 is flag for one batch
    'column_to_learn': ['docking1', 'consensus1'],
    'do_scoring_function_list_prediction': [True, False],
    'first_query_function': ["random_query_function", "cl"], #todo cluster functio
    'query_function': ["greedy_query_function", "random_query_function"], 
}

# Generate all combinations
keys, values = zip(*param_combinations.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Call the function with each combination
for combo in combinations:
    path_id=combo.pop("path_id")#paths are only a valid combo if the are to the same two parts of a dataset
    combo['smids_input_path'] = path_tuples[path_id][0]
    combo['ground_truth_path'] = path_tuples[path_id][1]
    #Assumption: combo purely consists of json serializable objects
    experiment_hash = hash_params(combo)
    combo['experiment_hash'] = experiment_hash

    combo['learner'] = get_learner_from_string(combo['learner'])(max_out_system=True)#instantiate learner
    combo['first_query_function'] = get_query_function_from_string(combo.pop("first_query_function"))
    combo['query_function'] = get_query_function_from_string(combo.pop("query_function"))
    

    active_learning_function(**combo)
    gc.collect()

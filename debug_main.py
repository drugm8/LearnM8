from active_learning_function import active_learning_function
import itertools
import gc
import json
import time
import os
import pandas as pd

from helpers.helpers import hash_params
from helpers.query_functions import greedy_query_function, random_query_function, cluster_query_function
from learners.pipe_cp_learner import pipe_cp_learner as ler
from learners.rf_learner import rf_learner as rf_learner
from learners.gp_learner import gp_learner as gp_learner

learner="learner"
SMIDS='./data/final_input.csv'
GTP="./data/data_raw.csv"



path_tuples = [[SMIDS,GTP]]

def get_learner_from_string(learner_string):
    if learner_string == "cp_learner":
        return ler
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
    elif query_function_string == "cluster_query_function":
        return cluster_query_function
    #mcdm py 

param_combinations = {
    "path_id": [0], #index in list of dataset paths
    'learner': ["cp_learner", "rf_learner"],
    'hyperparameter_tuning': [True, False],
    'batch_size_percentage': [1, 0.5, 0.1, 0.01],
    'smids_input_path': [None], #!stay
    'ground_truth_path': [None], #!stay 
    'cycles': [10, -1], #-1 is flag for one batch
    'column_to_learn': ['Zscore_best_scaled', 'ECR_avg_scaled'],
    'do_scoring_function_list_prediction': [False, True],
    'first_query_function': ["random_query_function"], 
    'query_function': ["greedy_query_function", "random_query_function"], 
    'statistical' : [0,1,2,3,]
}

#todo ECFP <=> butina clustering (based on scaffold) <=> [murcko scaffold]

# Generate all combinations
keys, values = zip(*param_combinations.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]


# Call the function with each combination
for combo in combinations:
    statistical = combo.pop("statistical")
    path_id=combo.pop("path_id")#paths are only a valid combo if the are to the same two parts of a dataset
    combo['smids_input_path'] = path_tuples[path_id][0]
    combo['ground_truth_path'] = path_tuples[path_id][1]

    ##DONT TOUCH, NEEDS TO BE LIKE THIS FOR WEBINTERFACE
    copy = combo.copy()
    for key, value in copy.items():
        copy[key] = str(value)
    experiment_hash = str(hash_params(copy))
    ##END

    #Assumption: combo purely consists of json serializable objects
    if not os.path.exists("./results/"+experiment_hash):
        os.mkdir("./results/"+experiment_hash)
        with open("./results/"+experiment_hash+"/combination.txt", 'w') as convert_file: 
            convert_file.write(json.dumps(combo))

    combo['learner'] = get_learner_from_string(combo['learner'])(max_out_system=True)#instantiate learner
    combo['first_query_function'] = get_query_function_from_string(combo.pop("first_query_function"))
    combo['query_function'] = get_query_function_from_string(combo.pop("query_function"))
    

    

    path ="./results/"+str(experiment_hash)+"/"+str(statistical)+"/"
    if not os.path.exists(path):
        os.mkdir(path)
    else: 
        print("skipping"+experiment_hash+" because it already exists")
        continue

    active_learning_function(**combo)

    os.rename("./internal_al_chache/", path+experiment_hash+"_internal_al_chache.csv")

    os.rename("./hpopt/", path+experiment_hash+"_hpopt.csv")
    gc.collect()

from active_learning_function import active_learning_function
import itertools
import gc
import json
import time
import os
import random
import pandas as pd
import numpy as np
import argparse
from helpers.helpers import hash_params
from helpers.query_functions import greedy_query_function, random_query_function, cluster_query_function
from learners.pipe_cp_learner import pipe_cp_learner as ler
from learners.rf_learner import rf_learner as rf_learner
from learners.gp_learner import gp_learner as gp_learner
import random

parser = argparse.ArgumentParser(description='Description of your program')
    
# Add argument with both short and long form
parser.add_argument('-m', '--mode', 
                        help='Description of the argument',
                        required=False,  # Make it optional
                        type=str)  # Specify type (str, int, etc.)

args = parser.parse_args()

mode = args.mode
learner="learner"
SMIDS='./data/final_input.csv'
GTP="./data/data_raw.csv"
scoring_files = [
    ['./data/ALDH1_4x4l_scoring_and_consensus_maxAL.csv', './data/ALDH1_4x4l_scoring_and_consensus_maxAL.csv'],
    ['./data/FEN1_5fv7_scoring_and_consensus_maxAL.csv', './data/FEN1_5fv7_scoring_and_consensus_maxAL.csv'],
    ['./data/GBA_2v3e_scoring_and_consensus_maxAL.csv', './data/GBA_2v3e_scoring_and_consensus_maxAL.csv'],
    ['./data/KAT2A_5h84_scoring_and_consensus_maxAL.csv', './data/KAT2A_5h84_scoring_and_consensus_maxAL.csv'],
    ['./data/MAPK1_2ojg_scoring_and_consensus_maxAL.csv', './data/MAPK1_2ojg_scoring_and_consensus_maxAL.csv'],
    ['./data/PKM2_3gr4_scoring_and_consensus_maxAL.csv', './data/PKM2_3gr4_scoring_and_consensus_maxAL.csv'],
    ['./data/VDR_3a2j_scoring_and_consensus_maxAL.csv', './data/VDR_3a2j_scoring_and_consensus_maxAL.csv']
]


path_tuples = [[SMIDS,GTP]]
path_tuples = scoring_files

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

if mode == "cpu":
    param_combinations = {
        "path_id": [0,1,2,3,4,5,6], #index in list of dataset paths
        'learner': ["rf_learner"],
        'hyperparameter_tuning': [False],
        'batch_size_percentage': [1, 0.5, 0.1, 0.01],
        'smids_input_path': [None], #!stay
        'ground_truth_path': [None], #!stay 
        'cycles': [10, -1], #-1 is flag for one batch
        'column_to_learn': [""],
        'do_scoring_function_list_prediction': [False], #todo or take care!!
        'first_query_function': ["random_query_function"], 
        'query_function': ["greedy_query_function","random_query_function"],
        'statistical' : [7,8,9]
    }

elif mode == "gpu":
    param_combinations = {
        "path_id": [2], #index in list of dataset paths
        'learner': ["cp_learner"],
        'hyperparameter_tuning': [False],
        'batch_size_percentage': [1,0.5,0.1, 0.01],
        'smids_input_path': [None], #!stay
        'ground_truth_path': [None], #!stay 
        'cycles': [10,-1], #-1 is flag for one batch
        'column_to_learn': [""],
        'do_scoring_function_list_prediction': [False,True], 
        'first_query_function': ["random_query_function"], 
        'query_function': ["greedy_query_function","random_query_function"],
        'statistical' : [7,8,9]
    }
else:
    param_combinations = {
        "path_id": [0], #index in list of dataset paths
        'learner': ["cp_learner","rf_learner"],
        'hyperparameter_tuning': [False],
        'batch_size_percentage': [1, 0.5, 0.1, 0.01],
        'smids_input_path': [None], #!stay
        'ground_truth_path': [None], #!stay 
        'cycles': [10, -1], #-1 is flag for one batch
        'column_to_learn': ['Zscore_best_scaled', 'ECR_avg_scaled'],
        'do_scoring_function_list_prediction': [False], #todo!!
        'first_query_function': ["random_query_function"], 
        'query_function': ["greedy_query_function", "random_query_function"], 
        'statistical' : [0,1,2,3,]
    }

#todo ECFP <=> butina clustering (based on scaffold) <=> [murcko scaffold]

# Generate all combinations
keys, values = zip(*param_combinations.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

random.shuffle(combinations)

# Call the function with each combination
for combo in combinations:
    print("combo", combo)
    if combo["do_scoring_function_list_prediction"]:
        if combo['learner'] != "cp_learner":
            continue


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
    try:
        os.mkdir(path)
    except FileExistsError:
        print("skipping"+experiment_hash+"since it exists")
        continue
    combo['learner'].set_path(path)

    seed=42
    if statistical ==0 or statistical ==1 or statistical ==2:
        seed=42
    else: seed=statistical

    random.seed(seed)
    np.random.seed(seed)

    active_learning_function(**combo, seed=seed)


    #try:
    #    os.rename("./hpopt/", path+experiment_hash+"_hpopt.csv")
    #except:
    #    pass
    gc.collect()

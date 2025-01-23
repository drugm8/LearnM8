import pandas as pd
import numpy as np
import gc
import math

from helpers.dock import dock
from helpers.helpers import remove_right_df_from_left_df
from helpers.query_functions import random_query_function
from scripts.consensus.consensus_wrapper import final_consensus_wrapper as consensus
from helpers.normalization import normalize_wrapper
from helpers.normalization import RESCORING_FUNCTIONS
from consensus.consensus import _METHODS as CONSENSUS_METHODS


def active_learning_function(learner, hyperparameter_tuning= False,
                             batch_size_percentage=0.1, smids_input_path=None,
                               ground_truth_path=None, cycles=10, column_to_learn=None,
                                 do_scoring_function_list_prediction=False, 
                                  first_query_function=None,
                                  query_function=None,
                                  ):
    learner.set_query_function(query_function)
    scoring_functions = []
    
    ##print(learner.name)



    smids_pool = pd.read_csv(smids_input_path)


    for col in smids_pool.columns:

        if col in RESCORING_FUNCTIONS.keys():
            ##print("added ", col, "to scoring functions")
            scoring_functions.append(col)
        elif col in CONSENSUS_METHODS.keys():
            ##print("added ", col, "to consensus methods")
            if column_to_learn =="":
                column_to_learn = col
            
        else:
           print("column ", col, " is not a scoring function or a consensus method")

    smids_pool = smids_pool.loc[:,["ID","SMILES"]]
    learner.set_column_to_learn(column_to_learn)
    learner.set_do_scoring_function_list_prediction(do_scoring_function_list_prediction)

    percentage = batch_size_percentage/100
    actual_batch_size = math.floor(smids_pool.shape[0]*percentage) 
    
    initial_sample = first_query_function(smids_pool, actual_batch_size)

    if cycles == -1: #-1 is used as a flag to just do one batch with same size as it would be otherwise in one batch
        actual_batch_size *= 10
        cycles = 1#!0

    learner.set_int_batch_size(batch_size = actual_batch_size)
    learner.set_scoring_functions(scoring_functions)



    print("asdf",initial_sample)



    

    smids_pool = remove_right_df_from_left_df(smids_pool, initial_sample)
    docked_inital_sample = dock(ground_truth_path, initial_sample, scoring_functions)
    #!happy


    print("docked_inital sample:", docked_inital_sample)
    #normalized_scores = normalize_wrapper(docked_inital_sample)
    #print("normalized scores:", normalized_scores)


    

    learner.teach(docked_inital_sample)

    for i in range(cycles+1):
        if hyperparameter_tuning and i == 0 and cycles == 1:
            #fall hyp and -1
            learner.optimize_hyperparameters() 
        if hyperparameter_tuning and i == 1 and cycles != 1:
           learner.optimize_hyperparameters()

        smids_queried = learner.query(smids_pool, smids_input_path,do_scoring_function_list_prediction, scoring_functions, column_to_learn)#
        
        if i == cycles:
           continue
      

        smids_pool = remove_right_df_from_left_df(smids_pool, smids_queried)

        docked_smids_queried = dock(ground_truth_path, smids_queried, scoring_functions)

        #normalized_scores = normalize_wrapper(docked_smids_queried)

        learner.teach(docked_smids_queried)

        gc.collect()

    return True

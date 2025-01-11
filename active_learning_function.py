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
    
    #print(learner.name)



    smids_pool = pd.read_csv(smids_input_path)

    for col in smids_pool.columns:

        if col in RESCORING_FUNCTIONS.keys():
            #print("added ", col, "to scoring functions")
            scoring_functions.append(col)
        elif col in CONSENSUS_METHODS.keys():
            #print("added ", col, "to consensus methods")
            column_to_learn = col

    smids_pool = smids_pool.loc[:,["ID","SMILES"]]

    if cycles == -1: #-1 is used as a flag to just do one batch with same size as it would be otherwise in one batch
        percentage = batch_size_percentage/100
        actual_batch_size = math.floor(smids_pool.shape[0]*percentage*10) # 10 hardcoded here to adjust to cycles:
        learner.set_int_batch_size(batch_size = actual_batch_size)
        cycles = 1
    else:
        percentage = batch_size_percentage/100
        actual_batch_size = math.floor(smids_pool.shape[0]*percentage)
        learner.set_int_batch_size(batch_size = actual_batch_size)

    initial_sample = first_query_function(smids_pool, None, actual_batch_size)#todo none here is what?

    smids_pool = remove_right_df_from_left_df(smids_pool, initial_sample)
    docked_inital_sample = dock(ground_truth_path, initial_sample, scoring_functions)


    #print("docked_inital sample", docked_inital_sample)
    normalized_scores = normalize_wrapper(docked_inital_sample)
    #print("nromaasdlfalsjd", normalized_scores)
    consensus_res = consensus(normalized_scores, column_to_learn, scoring_functions )#get a dataframe with scoring functions and consensus
    consensus_res= consensus_res.merge(docked_inital_sample, on=["ID"], how="inner")
    #print("joined consesns", consensus_res)

    # if column_to_learn in valid_consensus_methods:
    #   #todo normalize
    #   #print(consensus_res)
    #   consensus_res = consensus_res.merge(ini)
    # elif column_to_learn in valid_scoring_functions:
    #   consensus_res = docked_inital_sample#dunno
    # else:
    #     raise ValueError("Invalid column to learn")

    if do_scoring_function_list_prediction:
       learner.teach(consensus_res.loc[:,"SMILES"].values, consensus_res.loc[:,scoring_functions])
    else:
      #print("\n\n\n\n\n\n\n")
      #print(consensus_res)
      learner.teach(consensus_res.loc[:,"SMILES"].values, consensus_res.loc[:,column_to_learn].values)



    #print("know")

    for i in range(cycles):
        print("loopey1----",i)
        if hyperparameter_tuning and i ==1:
           learner.optimize_hyperparameters() #!time penalty
        print("loopey2")
        smids_queried = learner.query(smids_pool, smids_input_path,do_scoring_function_list_prediction, scoring_functions)#
        print("loopey3")
        smids_pool = remove_right_df_from_left_df(smids_pool, smids_queried)
        print("loopey4")
        docked_smids_queried = dock(ground_truth_path, smids_queried, scoring_functions)
        print("loopey5")
        normalized_scores = normalize_wrapper(docked_smids_queried)
        print("loopey6")

        consens_queried = consensus(normalized_scores, column_to_learn, scoring_functions)
        print("loopey7")
        consens_queried = consens_queried.merge(docked_smids_queried, on=["ID"], how="inner")
        print("loopey8")
        print(consens_queried)
        if do_scoring_function_list_prediction:
           #print("sf list predictionse")
           learner.teach(consens_queried.loc[:,"SMILES"].values, consens_queried.loc[:,scoring_functions])
        else:
          #print("consensus predictionse")
          learner.teach(consens_queried.loc[:,"SMILES"].values, consens_queried.loc[:,column_to_learn].values)
        gc.collect()

    return True

import pandas as pd
import numpy as np
import gc
import math

from helpers.dock import dock
from helpers.helpers import remove_right_df_from_left_df
from helpers.query_functions import random_query_function
from scripts.consensus.consensus_wrapper import new_consensus_wrapper as consensus
from helpers.normalization import normalize_scores


valid_scoring_functions = [
    "CNN-Score",
    "GenScore-scoring",
    "ConvexPLR",
    "KORP-PL"
]
valid_consensus_methods =[
    "ECR_avg_scaled",
    "ECR_best_scaled",
    "RbR_avg_scaled",
    "RbR_best_scaled",
    "RbV_avg_scaled",
    "RbV_best_scaled",
    "Zscore_avg_scaled",
    "Zscore_best_scaled",
    "Pareto_rank_avg_scaled",
    "Pareto_rank_best_scaled",
    "TOPSIS_avg_scaled",
    "TOPSIS_best_scaled",
    "WeightedSumModel_avg_scaled",
    "WeightedSumModel_best_scaled",
]
def active_learning_function(learner, hyperparameter_tuning= False,
                             batch_size_percentage=0.1, smids_input_path=None,
                               ground_truth_path=None, cycles=10, column_to_learn=None,
                                 do_scoring_function_list_prediction=False, 
                                  first_query_function=None,
                                  query_function=None,
                                  ):
    learner.set_query_function(query_function)
    


    smids_pool = pd.read_csv(smids_input_path)

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
    docked_inital_sample = dock(ground_truth_path, initial_sample)

    if column_to_learn in valid_consensus_methods:
      consensus_res = consensus(docked_inital_sample, column_to_learn)#get a dataframe with scoring functions and consensus
    elif column_to_learn in valid_scoring_functions:
      consensus_res = docked_inital_sample#dunno
    else:
        raise ValueError("Invalid column to learn")

    if do_scoring_function_list_prediction:
       consensus_res= consensus_res.merge(docked_inital_sample, on=["ID","SMILES"], how="inner")
       
       all_scoring_functions = ["CNN-Score","GenScore-scoring","ConvexPLR","KORP-PL"]
       print(consensus_res)
       print(consensus_res.columns.tolist())
       learner.teach(consensus_res.loc[:,"SMILES"].values, consensus_res.loc[:,all_scoring_functions])
    else:
      learner.teach(consensus_res.loc[:,"SMILES"].values, consensus_res.loc[:,column_to_learn].values)



    print("know")

    for i in range(cycles):
        if hyperparameter_tuning and i ==1:
           learner.optimize_hyperparameters() #!time penalty
        smids_queried = learner.query(smids_pool, smids_input_path)#
        print(smids_queried.columns)
        print(smids_pool)
        smids_pool = remove_right_df_from_left_df(smids_pool, smids_queried)
        docked_smids_queried = dock(ground_truth_path, smids_queried)
        consens_queried = consensus(docked_smids_queried, column_to_learn)
        if do_scoring_function_list_prediction:
           all_scoring_functions = [["CNN-Score","GenScore-scoring","ConvexPLR","KORP-PL"]]
           learner.teach(consens_queried.loc[:,"SMILES"].values, consens_queried.loc[:,all_scoring_functions])
        else:
          learner.teach(consens_queried.loc[:,"SMILES"].values, consens_queried.loc[:,column_to_learn].values)
        gc.collect()

    return True

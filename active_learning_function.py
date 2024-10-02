import pandas as pd
import numpy as np
import gc
import math

from helpers.helpers import remove_right_df_from_left_df
from scripts.consensus.consensus_wrapper import new_consensus_wrapper as consensus
from learners import pipe_cp_learner 
from learners import learner_abc as learner_type

valid_scoring_functions = ['CNN?']
valid_consensus_methods = ['ECR_avg_scaled', 'ECR_best_scaled', 'RbR_avg_scaled', 'RbR_best_scaled', 'RbV_avg_scaled', 'RbV_best_scaled', 'Zscore_avg_scaled', 'Zscore_best_scaled', 'Pareto_rank_avg_scaled', 'Pareto_rank_best_scaled', 'TOPSIS_avg_scaled', 'TOPSIS_best_scaled', 'WeightedSumModel_avg_scaled', 'WeightedSumModel_best_scaled']
learner = pipe_cp_learner.PipeCPLearner()

def active_learning_function(learnerr, hyperparameter_tuning= False,
                             batch_size_percentage=0.1, smids_input_path=None,
                               ground_truth_path=None, cycles=10, column_to_learn=None,
                                 do_scoring_function_list_prediction=False):
    learner.figure_out_system_and_set_parameters_accordingly()
    smids_pool = pd.read_csv(smids_input_path)

    actual_batch_size = math.floor(smids_pool.shape[0]*batch_size_percentage)
    learner.set_int_batch_size(batch_size = actual_batch_size)

    initial_sample = learner.first_query(smids_pool)
    smids_pool = remove_right_df_from_left_df(smids_pool, initial_sample)
    docked_inital_sample = dock(ground_truth_path, initial_sample)

    if column_to_learn in valid_consensus_methods:
      consensus = consensus(docked_inital_sample, column_to_learn)#get a dataframe with scoring functions and consensus
    elif column_to_learn in valid_scoring_functions:
      consensus = docked_inital_sample#dunno
    else:
        raise ValueError("Invalid column to learn")

    if do_scoring_function_list_prediction:
       return

    learner.teach(consensus.loc[:,"SMILES"].values, consensus.loc[:,column_to_learn].values)

    print("collected")
    for i in range(cycles):
        smids_queried = learner.query(smids_pool, smids_input_path)#
        smids_pool = remove_right_df_from_left_df(smids_pool, smids_queried)
        docked_smids_queried = dock(ground_truth_df_path, smids_queried)
        consens_queried = consensus(docked_smids_queried, column_to_learn)
        learner.teach(consens_queried.loc[:,"SMILES"].values, consens_queried.loc[:,column_to_learn].values)
        gc.collect()
from learners.chemprop_learner import chemprop_learner
import pandas as pd
import numpy as np
from helpers.query_functions import greedy_query_function, random_query_function
from helpers.scoring_metric import top_x_of_x_percentage
from learners.rf_learner import rf_learner
from learners.gp_learner import gp_learner
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from learners.sklearn_learner import sklearn_learner
from scripts.consensus.consensus_wrapper import consensus_wrapper as consensus
from helpers.dock import dock
from helpers.helpers import convert_list_of_smiles_to_morgan_fingerprints, log_and_save, log_list, remove_right_df_from_left_df
import sys
from sklearn.model_selection import GridSearchCV
import time
import os
from helpers.helpers import initialize_logging
##############ADJUST THESE########################
BATCH_SIZE = 1000
AL_CYCLES = 10
TOPX = 5000

rf_param_grid = {"n_estimators": [100, 200, 300, 400, 500], 
                "max_depth": [None, 10, 15, 20, 25],
                "min_samples_split": [2, 5, 10, 15, 20],
                "min_samples_leaf": [1, 2, 3, 4, 5],
                "n_jobs": [-1]}

#################################################
docking_scores = ["CNN-Score","GenScore-scoring","ConvexPLR"]
#cs_methods = ['ECR_avg_scaled', 'ECR_best_scaled', 'RbR_avg_scaled', 'RbR_best_scaled', 'RbV_avg_scaled', 'RbV_best_scaled', 'Zscore_avg_scaled', 'Zscore_best_scaled', 'Pareto_rank_avg_scaled', 'Pareto_rank_best_scaled', 'TOPSIS_avg_scaled', 'TOPSIS_best_scaled', 'WeightedSumModel_avg_scaled', 'WeightedSumModel_best_scaled']
# Open log file for writing

log_file = initialize_logging(__file__)

def evaluate_learner(learner):
    prediction_df = full_smids_final_input
    prediction_df["estimation"] = learner.estimate(full_smids_final_input.loc[:,"SMILES"])
    top_x_score = top_x_of_x_percentage(ground_truth_df_path, prediction_df, TOPX, metric=do)
    return top_x_score


ground_truth_df_path = "./data/data_raw.csv"


for do in docking_scores:
    smids_final_input = pd.read_csv('./data/final_input.csv')
    full_smids_final_input = smids_final_input.copy()

    #Initializing data
    inital_random_sample = random_query_function(smids_final_input, None, BATCH_SIZE)
    smids_final_input = remove_right_df_from_left_df(smids_final_input, inital_random_sample)
    docked_inital_random_sample  = dock(ground_truth_df_path, inital_random_sample)

    #gettigngood parameters from first batch based on grid search
    rf = RandomForestRegressor()
    gs = GridSearchCV(rf, rf_param_grid, cv=5, scoring='neg_mean_absolute_error')
    gs.fit(convert_list_of_smiles_to_morgan_fingerprints(docked_inital_random_sample.loc[:,"SMILES"].values),  docked_inital_random_sample.loc[:,do].values)
    best_model = gs.best_estimator_
    log_and_save(f"Best model:{gs.best_parmameters_}",log_file)

    learner = sklearn_learner(greedy_query_function, docked_inital_random_sample.loc[:,"SMILES"].values, docked_inital_random_sample.loc[:,do].values, batch_size=BATCH_SIZE)

    log_and_save(f"Batch size: {BATCH_SIZE}; active learning cycles: {AL_CYCLES}; top X of X percentage score: {TOPX}; Machine learning architecture:{learner.getName()}; DOCKING SCORE {do};",log_file)
    topxlist = []
    topxlist.append(evaluate_learner(learner))

    #execute AL
    for i in range(AL_CYCLES):
        smids_queried = learner.query(smids_final_input)
        smids_final_input = remove_right_df_from_left_df(smids_final_input, smids_queried)
        docked_smids_queried = dock(ground_truth_df_path, smids_queried)

        learner.teach(smids_queried.loc[:,"SMILES"].values, docked_smids_queried.loc[:,do].values)
        topxlist.append(evaluate_learner(learner))


    log_list(topxlist, log_file)
log_and_save("\n\n\n",log_file)



for do in docking_scores:
    smids_final_input = pd.read_csv('./data/final_input.csv')
    full_smids_final_input = smids_final_input.copy()

    #Initializing data
    inital_random_sample = random_query_function(smids_final_input, None, BATCH_SIZE)
    smids_final_input = remove_right_df_from_left_df(smids_final_input, inital_random_sample)
    docked_inital_random_sample  = dock(ground_truth_df_path, inital_random_sample)


    learner = chemprop_learner(greedy_query_function, docked_inital_random_sample.loc[:,"SMILES"].values, docked_inital_random_sample.loc[:,do].values, batch_size=BATCH_SIZE)

    log_and_save(f"Batch size: {BATCH_SIZE}; active learning cycles: {AL_CYCLES}; top X of X percentage score: {TOPX}; Machine learning architecture:{learner.getName()}; DOCKING SCORE {do};",log_file)
    topxlist = []
    topxlist.append(evaluate_learner(learner))

    #execute AL
    for i in range(AL_CYCLES):
        smids_queried = learner.query(smids_final_input)
        smids_final_input = remove_right_df_from_left_df(smids_final_input, smids_queried)
        docked_smids_queried = dock(ground_truth_df_path, smids_queried)

        learner.teach(smids_queried.loc[:,"SMILES"].values, docked_smids_queried.loc[:,do].values)
        topxlist.append(evaluate_learner(learner))


    log_list(topxlist, log_file)
log_and_save("\n\n\n",log_file)

for do in docking_scores:
    smids_final_input = pd.read_csv('./data/final_input.csv')
    full_smids_final_input = smids_final_input.copy()

    #Initializing data
    inital_random_sample = random_query_function(smids_final_input, None, BATCH_SIZE)
    smids_final_input = remove_right_df_from_left_df(smids_final_input, inital_random_sample)
    docked_inital_random_sample  = dock(ground_truth_df_path, inital_random_sample)


    learner = gp_learner(greedy_query_function, docked_inital_random_sample.loc[:,"SMILES"].values, docked_inital_random_sample.loc[:,do].values, batch_size=BATCH_SIZE)

    log_and_save(f"Batch size: {BATCH_SIZE}; active learning cycles: {AL_CYCLES}; top X of X percentage score: {TOPX}; Machine learning architecture:{learner.getName()}; DOCKING SCORE {do};",log_file)
    topxlist = []
    topxlist.append(evaluate_learner(learner))

    #execute AL
    for i in range(AL_CYCLES):
        smids_queried = learner.query(smids_final_input)
        smids_final_input = remove_right_df_from_left_df(smids_final_input, smids_queried)
        docked_smids_queried = dock(ground_truth_df_path, smids_queried)

        learner.teach(smids_queried.loc[:,"SMILES"].values, docked_smids_queried.loc[:,do].values)
        topxlist.append(evaluate_learner(learner))


    log_list(topxlist, log_file)
log_and_save("\n\n\n",log_file)


log_file.close()
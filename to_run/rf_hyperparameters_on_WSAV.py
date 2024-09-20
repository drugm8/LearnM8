
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
from helpers.helpers import log_and_save, log_list, remove_right_df_from_left_df
import sys
from sklearn.model_selection import GridSearchCV
import time
import os
##############ADJUST THESE########################
BATCH_SIZE = 1000
AL_CYCLES = 10
TOPX = 5000
CONSENSUS_METHOD = "WeightedSumModel_avg_scaled"
parameter_grid ={
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'bootstrap': [True, False],
    "warm_start": [True]
}
MODEL = RandomForestRegressor()
#################################################
cs_methods = ['ECR_avg_scaled', 'ECR_best_scaled', 'RbR_avg_scaled', 'RbR_best_scaled', 'RbV_avg_scaled', 'RbV_best_scaled', 'Zscore_avg_scaled', 'Zscore_best_scaled', 'Pareto_rank_avg_scaled', 'Pareto_rank_best_scaled', 'TOPSIS_avg_scaled', 'TOPSIS_best_scaled', 'WeightedSumModel_avg_scaled', 'WeightedSumModel_best_scaled']
# Open log file for writing
filename = os.path.basename(__file__).split(".")[0]
if not os.path.exists("./runs/"+filename):
    os.makedirs("./runs/"+filename)
log_file_path = "./runs/"+filename+"/"+"log_"+str(time.strftime("%Y-%m-%d %H:%M:%S"))+".txt"
log_file = open(log_file_path, "w")

def evaluate_learner(learner):
    prediction_df = full_smids_final_input
    prediction_df["estimation"] = learner.estimate(full_smids_final_input.loc[:,"SMILES"])
    top_x_score = top_x_of_x_percentage(ground_truth_df_path, prediction_df, TOPX, metric=CONSENSUS_METHOD)
    return top_x_score
ground_truth_df_path = "./data/data_raw.csv"
smids_final_input = pd.read_csv('./data/final_input.csv')
full_smids_final_input = smids_final_input.copy()
#Initializing data
inital_random_sample = random_query_function(smids_final_input, None, BATCH_SIZE)
smids_final_input = remove_right_df_from_left_df(smids_final_input, inital_random_sample)
docked_inital_random_sample  = dock(ground_truth_df_path, inital_random_sample)
consens = consensus(docked_inital_random_sample, CONSENSUS_METHOD)
gridsearch= GridSearchCV(MODEL, parameter_grid, cv=5, scoring='neg_mean_absolute_error')
gridsearch.fit(docked_inital_random_sample.loc[:,"SMILES"].values, docked_inital_random_sample.loc[:,"consensus"].values)
best_model = gridsearch.best_estimator_
log_and_save(f"Best model:{gridsearch.best_parmameters_}",log_file)
learner = sklearn_learner(greedy_query_function, consens.loc[:,"SMILES"].values, consens.loc[:,"consensus"].values, model=best_model, batch_size=BATCH_SIZE)
log_and_save(f"Batch size: {BATCH_SIZE}; active learning cycles: {AL_CYCLES}; top X of X percentage score: {TOPX}; Machine learning architecture:{learner.getName()}; Consensus Method:{CONSENSUS_METHOD};",log_file)
topxlist = []
topxlist.append(evaluate_learner(learner))
#execute AL
for i in range(AL_CYCLES):
    smids_queried = learner.query(smids_final_input)
    smids_final_input = remove_right_df_from_left_df(smids_final_input, smids_queried)
    docked_smids_queried = dock(ground_truth_df_path, smids_queried)
    consens_queried = consensus(docked_smids_queried, CONSENSUS_METHOD)
    learner.teach(smids_queried.loc[:,"SMILES"].values, consens_queried.loc[:,"consensus"].values)
    topxlist.append(evaluate_learner(learner))
log_list(topxlist, log_file)
log_file.close()
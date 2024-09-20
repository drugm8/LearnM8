from learners.chemprop_learner import chemprop_learner
import pandas as pd
import numpy as np
from helpers.query_functions import greedy_query_function, random_query_function
from helpers.scoring_metric import top_x_of_x_percentage
from learners.rf_learner import rf_learner
from learners.gp_learner import gp_learner
from sklearn.ensemble import RandomForestRegressor

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from learners.sklearn_learner import sklearn_learner

#import gridsearch
from sklearn.model_selection import GridSearchCV

from scripts.consensus.consensus_wrapper import consensus_wrapper as consensus
from helpers.dock import dock
from helpers.helpers import convert_list_of_smiles_to_morgan_fingerprints, remove_right_df_from_left_df
import sys
import time

BATCH_SIZE = 10
AL_CYCLES = 2
TOPX = 5000
METRIC="Zscore_avg_scaled"
metrics = ['ECR_avg_scaled', 'ECR_best_scaled', 'RbR_avg_scaled', 'RbR_best_scaled', 'RbV_avg_scaled', 'RbV_best_scaled', 'Zscore_avg_scaled', 'Zscore_best_scaled', 'Pareto_rank_avg_scaled', 'Pareto_rank_best_scaled', 'TOPSIS_avg_scaled', 'TOPSIS_best_scaled', 'WeightedSumModel_avg_scaled', 'WeightedSumModel_best_scaled']

# Open log file for writing
log_file_path = "./log_file.txt"
log_file = open(log_file_path, "w")

def log_and_save(message):
    log_file.write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + message + "\n")
    #log_file.write(message + "\n")
    log_file.flush()  # Ensure it's written to the file immediately



ground_truth_df_path = "./data/data_raw.csv"
smids_final_input = pd.read_csv('./data/final_input.csv')
full_smids_final_input = smids_final_input.copy()
RESULT_DF = None
inital_random_sample = random_query_function(smids_final_input, None, 10000)
smids_final_input = remove_right_df_from_left_df(smids_final_input, inital_random_sample)
docked_inital_random_sample  = dock(ground_truth_df_path, inital_random_sample)
consens = consensus(docked_inital_random_sample, METRIC)

grid_search_params = dict(n_estimators=[50, 100, 200, 300, 400, 500, 1000])
grid_search = GridSearchCV(RandomForestRegressor(), grid_search_params, cv=5, verbose=10, n_jobs=-1)
fp =convert_list_of_smiles_to_morgan_fingerprints(consens.loc[:,"SMILES"].values)
grid_search.fit(fp, consens.loc[:,"consensus"].values)
print(grid_search.best_params_)



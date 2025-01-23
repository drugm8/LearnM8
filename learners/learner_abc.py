from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np
from scripts.consensus.consensus_wrapper import final_consensus_wrapper as consensus
from helpers.normalization import normalize_wrapper


from helpers.normalization import RESCORING_FUNCTIONS

from sklearn.metrics import mean_squared_error

class learner(ABC):
    dataset = None
    do_scoring_function_list_prediction = None
    column_to_learn = None
    scoring_functions = None


    @abstractmethod
    def __init__(self, query_function, initial_x, initial_y):
        pass

    @abstractmethod
    def teach(self, dataset_x, dataset_y):
        #appends data to the train set
        pass
    
    @abstractmethod
    def query(self, dataset, howmany):
        pass

    @abstractmethod
    def estimate(self, dataset):
        pass
    
    @abstractmethod
    def optimize_hyperparameters(self):
        pass

    def getName(self):
        return self.name
    
    def set_query_function(self, func):
        self.query_function = func

    def set_int_batch_size(self, batch_size):
        self.batch_size = batch_size
    def set_path(self, path):
        self.path=path



    def write_estimations(self, full_input_smids):
        if not os.path.exists(self.path+"cache.csv"):
            full_input_smids.rename(columns={"estimation": "cycle_0"}, inplace=True)
            full_input_smids.to_csv(self.path+"cache.csv", index=False)
        else:
            cachefile = pd.read_csv(self.path+"cache.csv")
            cachefile_columns_count = str(len(cachefile.columns)-1)
            full_input_smids.rename(columns={"estimation": "cycle_"+cachefile_columns_count}, inplace=True)
            merged = pd.merge(cachefile, full_input_smids, on='ID', how='outer')#unschüon aber sollte passe n
            merged.to_csv(self.path+"cache.csv", index=False)


    def preprocess_learnable_data(self):
        if self.column_to_learn in RESCORING_FUNCTIONS.keys(): #preprocess learnable data in case of single scoring function
                self.dataset_x = self.dataset.loc[:,"SMILES"].values
                self.dataset_y = self.dataset.loc[:,self.column_to_learn]
                return        
        normalized = normalize_wrapper(self.dataset)
        consensed = self.dataset.merge(consensus(normalized, self.column_to_learn, self.scoring_functions), on="ID", how="inner")
        self.dataset_x = consensed.loc[:,"SMILES"].values
        if self.do_scoring_function_list_prediction:
            self.dataset_y = self.dataset.loc[:,self.scoring_functions]
        else:
            self.dataset_y = consensed.loc[:,self.column_to_learn]
        return 
    
    def append_data(self, addition):
        if self.dataset is not None: #append data to existing dataset
            self.dataset = pd.concat([self.dataset, addition], axis=0, ignore_index=True)
        else: #create write full data to learner
            self.dataset = addition

        self.preprocess_learnable_data()

    def also_save_scoring_function_estimations(self, df):
        counter = 0
        while counter < 100:
            filename = f"{self.path}/cycle{counter}.csv"
            try:
                df.to_csv(filename, index=True)
                return
            except FileExistsError:
                counter += 1


    def query(self, smids_x_input, path, do_consensing=False, scoring_functions=None, column_to_learn=None):
        #print("querying...")
        #uses the intrinisc query function to run the inference first and then query the dataset
        full_input_smids = pd.read_csv(path)


        if do_consensing:
            scoring_function_estimations = self.estimate(full_input_smids.loc[:,"SMILES"])
            print(scoring_function_estimations)
            scoring_function_estimations_df = pd.DataFrame(scoring_function_estimations, columns=scoring_functions)
            scoring_function_estimations_df["ID"] = full_input_smids.loc[:,"ID"]

            self.also_save_scoring_function_estimations(scoring_function_estimations_df)

            normalized_predictions = normalize_wrapper(scoring_function_estimations_df)

            consensus_res = consensus(normalized_predictions, column_to_learn, scoring_functions)
            consensus_estimations = consensus_res.loc[:,column_to_learn]
        else:
            consensus_estimations = self.estimate(full_input_smids.loc[:,"SMILES"])
            #print("query estimations:", consensus_estimations)

        full_input_smids["estimation"] = consensus_estimations
        mse = mean_squared_error(full_input_smids.loc[:,"estimation"], full_input_smids.loc[:,column_to_learn])

        print(mse)

        self.write_estimations(full_input_smids.loc[:,["ID", "estimation"]].copy())

        merged_and_reduced_smids =  pd.merge(smids_x_input, full_input_smids, on=["ID", "SMILES"], how='inner')

        queried = self.query_function(merged_and_reduced_smids, batch_size=self.batch_size)
        return queried

    def set_column_to_learn(self, column_to_learn):
        self.column_to_learn = column_to_learn
    def set_scoring_functions(self, scoring_functions):
        self.scoring_functions = scoring_functions
    def set_do_scoring_function_list_prediction(self, do_scoring_function_list_prediction):
        self.do_scoring_function_list_prediction = do_scoring_function_list_prediction

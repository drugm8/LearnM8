from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np
from scripts.consensus.consensus_wrapper import final_consensus_wrapper as consensus


from helpers.normalization import normalize_wrapper

class learner(ABC):

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
            merged = pd.merge(cachefile, full_input_smids, on='ID', how='outer')
            merged.to_csv(self.path+"cache.csv", index=False)


    def append_data(self, addition_of_dataset_x, addition_of_dataset_y):
        if (self.dataset_x is not None) and (self.dataset_y is not None):
            #print("appending ", addition_of_dataset_x)
            #print("to", self.dataset_x)
            #print("appending ", addition_of_dataset_y)
            #print("to", self.dataset_y)
            if isinstance(addition_of_dataset_y, pd.DataFrame):
                self.dataset_y=pd.concat([self.dataset_y, addition_of_dataset_y], axis=0, ignore_index=True)
            else:
                self.dataset_y=np.append(self.dataset_y, addition_of_dataset_y)
            if isinstance(addition_of_dataset_x, pd.DataFrame):
                self.dataset_x=pd.concat([self.dataset_x, addition_of_dataset_x], axis=0, ignore_index=True)
            else:
                self.dataset_x=np.append(self.dataset_x, addition_of_dataset_x)
        else:
            self.dataset_y = addition_of_dataset_y
            self.dataset_x = addition_of_dataset_x
        #print("done appendign x",self.dataset_x)
        #print("done appendign y", self.dataset_y)

    def query(self, smids_x_input, path, do_consensing=False, scoring_functions=None, column_to_learn=None):
        #print("querying...")
        #uses the intrinisc query function to run the inference first and then query the dataset
        full_input_smids = pd.read_csv(path)
        if do_consensing:
            scoring_function_estimations = self.estimate(full_input_smids.loc[:,"SMILES"])

            scoring_function_estimations_df = pd.DataFrame(scoring_function_estimations, columns=scoring_functions)
            scoring_function_estimations_df["ID"] = full_input_smids.loc[:,"ID"]
            normalized_predictions = normalize_wrapper(scoring_function_estimations_df)

            consensus_res = consensus(normalized_predictions, column_to_learn, scoring_functions)
            consensus_estimations = consensus_res.loc[:,column_to_learn]
        else:
            consensus_estimations = self.estimate(full_input_smids.loc[:,"SMILES"])
            #print("query estimations:", consensus_estimations)

        full_input_smids["estimation"] = consensus_estimations

        self.write_estimations(full_input_smids.loc[:,["ID", "estimation"]].copy())
        merged_and_reduced_smids =  pd.merge(smids_x_input, full_input_smids, on=["ID", "SMILES"], how='inner')

        queried = self.query_function(merged_and_reduced_smids, merged_and_reduced_smids.loc[:,"estimation"], batch_size=self.batch_size)
        return queried

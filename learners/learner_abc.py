from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np
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
    



    def write_estimations(self, full_input_smids):
        if not os.path.exists("./internal_al_chache/"):
            os.makedirs("./internal_al_chache/")
            full_input_smids.rename(columns={"estimation": "cycle_0"}, inplace=True)
            full_input_smids.to_csv("./internal_al_chache/cache.csv", index=False)
        else:
            cachefile = pd.read_csv("./internal_al_chache/cache.csv")
            cachefile_columns_count = str(len(cachefile.columns)-1)
            full_input_smids.rename(columns={"estimation": "cycle_"+cachefile_columns_count}, inplace=True)
            merged = pd.merge(cachefile, full_input_smids, on='ID', how='outer')
            merged.to_csv("./internal_al_chache/cache.csv", index=False)


    def append_data(self, addition_of_dataset_x, addition_of_dataset_y):
        if (self.dataset_x is not None) and (self.dataset_y is not None):
            self.dataset_y=np.append(addition_of_dataset_y, self.dataset_y)
            self.dataset_x=np.append(addition_of_dataset_x, self.dataset_x)
        else:
            self.dataset_y = addition_of_dataset_y
            self.dataset_x = addition_of_dataset_x

    def query(self, smids_x_input, path):
        #uses the intrinisc query function to run the inference first and then query the dataset
        full_input_smids = pd.read_csv(path)

        full_input_smids["estimation"] = self.estimate(full_input_smids.loc[:,"SMILES"])
        self.write_estimations(full_input_smids.loc[:,["ID", "estimation"]].copy())
        merged_and_reduced_smids =  pd.merge(smids_x_input, full_input_smids, on=["ID", "SMILES"], how='inner')

        queried = self.query_function(merged_and_reduced_smids, merged_and_reduced_smids.loc[:,"estimation"], batch_size=self.batch_size)
        return queried
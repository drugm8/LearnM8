from abc import ABC, abstractmethod
from learners.learner_abc import learner
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from helpers.helpers import convert_list_of_smiles_to_morgan_fingerprints

    
class sklearn_learner(learner):
    #SMILES INPUT, they are featurized here
    #my implementation of a learner has the most up to date training set always stored internally

    def __init__(self, query_function, dataset_x, dataset_y,model, batch_size=32):
        #!NO smids here, just plain values, need to check if reihenfolge is kept
        self.query_function = query_function
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.batch_size = batch_size
        self.model = model
        self.model = self.train_new_model(dataset_x= self.dataset_x, dataset_y=self.dataset_y)
        self.name = model.__class__.__name__



    def teach(self, addition_of_dataset_x, addition_of_dataset_y):
        self.dataset_y=np.append(addition_of_dataset_y, self.dataset_y)
        self.dataset_x=np.append(addition_of_dataset_x, self.dataset_x)
        self.model = self.train_new_model(dataset_x= self.dataset_x, dataset_y=self.dataset_y )


   

    
    def query(self, smids_input):
        #uses the intrinisc query function to run the inference first and then query the dataset
        estimation = self.estimate(smids_input.loc[:,"SMILES"])
        queried = self.query_function(smids_input, estimation, batch_size=self.batch_size)
        return queried


    def estimate(self, x_input):
        fingerprints_array = convert_list_of_smiles_to_morgan_fingerprints(x_input)#!optimize here

        return self.model.predict(fingerprints_array)

    def train_new_model(self, dataset_x, dataset_y):
        model = self.model
        fingerprints_array = convert_list_of_smiles_to_morgan_fingerprints(dataset_x)
        model.fit(fingerprints_array, dataset_y)
        return model

        

    def print_inner_data(self):
        print("data_x", self.dataset_x)

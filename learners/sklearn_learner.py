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

    def __init__(self,model):
        #!NO smids here, just plain values, need to check if reihenfolge is kept
        self.model = model
        self.name = model.__class__.__name__



    def teach(self, addition):
        self.append_data(addition)
        self.model = self.train_new_model()


   

    




    def estimate(self, x_input):
        fingerprints_array = convert_list_of_smiles_to_morgan_fingerprints(x_input)#!optimize here

        return self.model.predict(fingerprints_array)

    def train_new_model(self):
        model = self.model
        
        fingerprints_array = convert_list_of_smiles_to_morgan_fingerprints(self.dataset_x)
        print(fingerprints_array)
        print(self.dataset_y)
        model.fit(fingerprints_array, self.dataset_y)
        return model

        

    def print_inner_data(self):
        print("data_x", self.dataset_x)


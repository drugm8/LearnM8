from abc import ABC, abstractmethod
from learners.learner_abc import learner
import pandas as pd
from pathlib import Path
import numpy as np

from lightning import pytorch as pl

from chemprop import data, featurizers, models, nn

from _chemprop.do_chemprop import do_chempop



class chemprop_learner(learner):
    #SMILES INPUT, they are featurized here
    #my implementation of a learner has the most up to date training set always stored internally

    def __init__(self, query_function, dataset_x, dataset_y, batch_size=32):

        self.query_function = query_function
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.batch_size = batch_size
        self.trainer, self.mpnn = do_chempop(smiles= self.dataset_x, ys=self.dataset_y ) #!ys can be multiple i think
        self.name = "chemprop"


    def teach(self, addition_of_dataset_x, addition_of_dataset_y):
        self.dataset_y=np.append(addition_of_dataset_y, self.dataset_y)
        self.dataset_x=np.append(addition_of_dataset_x, self.dataset_x)
        self.trainer, self.mpnn = do_chempop(smiles= self.dataset_x, ys=self.dataset_y )

    
    def query(self, smids_x_input):
        #uses the intrinisc query function to run the inference first and then query the dataset

        estimation = self.estimate(smids_x_input.loc[:,"SMILES"])

        queried = self.query_function(smids_x_input, estimation, batch_size=self.batch_size)
        return queried


    def estimate(self, x_input):
        #takes a dataframe of smiles and returns a dataframe with the estimation
        if self.trainer is None:
            raise Exception("trainer not trained")
        test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in x_input]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        test_dset = data.MoleculeDataset(test_data, featurizer)
        test_loader = data.build_dataloader(test_dset, num_workers=8, shuffle=False)
        predictions = self.trainer.predict(self.mpnn, test_loader)
        flat_estimations = [item.item() for sublist in predictions for item in sublist]
        return flat_estimations
    

    def print_inner_data(self):
        print("datax", self.dataset_x)
        print("datay", self.dataset_y)
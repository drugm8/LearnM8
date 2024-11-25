from abc import ABC, abstractmethod
import os

from sklearn.model_selection import RandomizedSearchCV
from learners.learner_abc import learner
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from learners.sklearn_learner import sklearn_learner




class rf_learner(sklearn_learner):
    def __init__(self,  max_out_system = True):
        n_jobs = -1 if max_out_system else 1
        if n_jobs == -1:
            n_jobs = os.cpu_count()
            if n_jobs > 32:
                n_jobs = 32

        super().__init__( RandomForestRegressor(n_estimators=100, n_jobs=n_jobs))
        self.name = "Random Forest Regressor"

    def optimize_hyperparameters(self):
        #gridsearch
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [5, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False],
            'criterion': ['mse', 'mae'],
            'max_leaf_nodes': [None, 30, 50, 70, 90],
            'min_impurity_decrease': [0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'min_impurity_split': [None, 0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
        gridsearch = RandomizedSearchCV(self.model, param_distributions=param_grid, n_iter=10, cv=5, verbose=1, random_state=42)
        self.model = gridsearch.fit(self.dataset_x, self.dataset_y).best_estimator_
        return
        
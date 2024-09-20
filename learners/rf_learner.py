from abc import ABC, abstractmethod
from learners.learner_abc import learner
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from learners.sklearn_learner import sklearn_learner




class rf_learner(sklearn_learner):
    def __init__(self, query_function, dataset_x, dataset_y, batch_size):
        super().__init__(query_function, dataset_x, dataset_y, RandomForestRegressor(n_estimators=100, n_jobs=-1), batch_size)
        self.name = "Random Forest Regressor"
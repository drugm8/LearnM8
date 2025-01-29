import pandas as pd
import numpy as np
from helpers.normalization import normalize_wrapper
import os

DATA_DIR="../data"

for data in os.listdir(DATA_DIR):
    if data.endswith("AL.csv"):
        df = pd.read_csv(DATA_DIR+"/"+data)
        normalized = normalize_wrapper(df)
        name = data.replace("AL.csv", "AL_normalized.csv")
        df.to_csv(DATA_DIR+"/"+name)
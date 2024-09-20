import pandas as pd
import numpy as np

dataset_tony_path = "../data/data_raw.csv"
output_path = "../data/final_input.csv"

dataset_tony = pd.read_csv(dataset_tony_path)

final_input = dataset_tony.loc[:, ["ID","SMILES"]]
final_input.to_csv(output_path, index=False)
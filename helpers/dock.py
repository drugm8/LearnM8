import pandas as pd
import numpy as np



def dock(ground_truth_df_path, dataset):
    #! takes some IDS and gives df with SMILES and the 4 SF values
    ground_trouth_df = pd.read_csv(ground_truth_df_path)
    merged_df = pd.merge(dataset, ground_trouth_df,left_on=["ID","SMILES"], right_on = ["ID", "SMILES"])
    #print (merged_df)
    return(merged_df.loc[:, ["ID", "SMILES","Pose ID", "KORP-PL","RFScoreVS","Vinardo","CHEMPLP","CNN-Affinity"]])
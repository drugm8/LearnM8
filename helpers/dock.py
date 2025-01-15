import pandas as pd
import numpy as np



def dock(ground_truth_df_path, dataset,scoring_functions):
    #! takes some IDS and gives df with SMILES and the 4 SF values
    ground_trouth_df = pd.read_csv(ground_truth_df_path)
    #print (ground_trouth_df)
    #print (dataset)
    merged_df = pd.merge(dataset.loc[:,["ID"]], ground_trouth_df,left_on=["ID"], right_on = ["ID"], how="inner")
    #print (merged_df)
    col_list = ["ID", "SMILES"]+scoring_functions

    #print(col_list)
    return(merged_df.loc[:,col_list])
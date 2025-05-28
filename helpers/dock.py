import pandas as pd

def dock(ground_truth_df_path, dataset,scoring_functions):
    #takes some IDS and gives df with SMILES and the 4 SF values
    ground_trouth_df = pd.read_csv(ground_truth_df_path)
    merged_df = pd.merge(dataset.loc[:,["ID"]], ground_trouth_df,left_on=["ID"], right_on = ["ID"], how="inner")
    col_list = ["ID", "SMILES"]+scoring_functions
    return(merged_df.loc[:,col_list])
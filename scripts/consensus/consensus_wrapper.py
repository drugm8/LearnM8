from scripts.consensus.consensus import apply_consensus_methods
from consensus.consensus import apply_consensus_scoring
import pandas as pd

# def consensus_wrapper(dataset, metric):
#     print("consensing...")
#     method= metric.rpartition('_')[0]
#     result = apply_consensus_methods(dataset, method, "scaled",False)
#     df =result[0].rename(columns={method: 'consensus'})
#     print("done consensing...")
#     return(df)

_used_scoring_functions = [
  "KORP-PL", "RFScoreVS", "Vinardo", "CHEMPLP", "CNN-Affinity"
]


def final_consensus_wrapper(dataset, col_to_learn, scoring_functions=_used_scoring_functions):
    if col_to_learn == "Consensus_SoftRbV":
        col_to_learn = "soft_rbv"
        result = apply_consensus_scoring(dataset, col_to_learn, scoring_functions)
        result = result.rename(columns={"SoftRbV": "Consensus_SoftRbV"})
    return(result)
    

def new_consensus_wrapper(dataset, col_to_learn): #currently used
    print("consensing...")
    method= col_to_learn.rpartition('_')[0]
    result = apply_consensus_methods(dataset, method, "scaled",False)
    df =result[0].rename(columns={method: col_to_learn})
    print("done consensing...")
    return(df)


# def merge_consensus(dataset, metric):
#     print("consensing, mergin...")
#     method= metric.rpartition('_')[0]
#     result = apply_consensus_methods(dataset, method, "scaled",False)
#     df =result[0].rename(columns={method: 'consensus'})
#     merged = pd.merge(dataset, df, left_on=["ID","SMILES"], right_on = ["ID", "SMILES"])
#     print(merged.columns)
#     print("done consensin and merging...")
#     return(merged)


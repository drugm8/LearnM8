from scripts.consensus.consensus import apply_consensus_methods

def consensus_wrapper(dataset, metric):
    method= metric.rpartition('_')[0]
    result = apply_consensus_methods(dataset, method, "scaled",False)
    df =result[0].rename(columns={method: 'consensus'})
    return(df)
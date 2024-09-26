import pandas as pd
def greedy_query_function(x_i, estimation, batch_size):
    x_input = x_i.copy()
    x_input["estimation"] = estimation
    x_input["estimation"] = [11.1, 10.2, 20.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sorted_df = x_input.sort_values(by='estimation', ascending=False)
    queried_df = sorted_df.head(batch_size)
    return queried_df

est = [1.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

x = pd.DataFrame({'SMILES': ['CCC', 'CCCC', 'CCCCC', 'CCCCCC', 'CCCCCCC', 'CCCCCCCC', 'CCCCCCCCC', 'CCCCCCCCCC', 'CCCCCCCCCCC', 'CCCCCCCCCCCC']})

print(greedy_query_function(x, est, 3))

def random_query_function(x_input,estimation, batch_size):
    dataset = x_input.sample(frac=1, random_state=42)
    return dataset.head(batch_size)

print(random_query_function(x, est, 3))
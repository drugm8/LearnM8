import pandas as pd
import numpy as np

def greedy_query_function(x_i, estimation, batch_size):
    x_input = x_i.copy()
    x_input["estimation"] = estimation
    sorted_df = x_input.sort_values(by='estimation', ascending=False)
    queried_df = sorted_df.head(batch_size)
    return queried_df


def random_query_function(x_input,estimation, batch_size):
    dataset = x_input.sample(frac=1, random_state=42)
    return dataset.head(batch_size)
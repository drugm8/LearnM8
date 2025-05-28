import pandas as pd


def top_x_of_x_percentage(ground_truth_df_path, estimation_df, x, metric):
    """
    Calculates the percentage of IDs in the top x of the estimation_df that are also in the top x of the ground_truth_df.
    Args:
        ground_truth_df_path (str): Path to the ground truth DataFrame CSV file.
        estimation_df (pd.DataFrame): DataFrame containing the estimated values with an 'ID' column.
        x (int): The number of top entries to consider.
        metric (str): The column name in the ground truth DataFrame to compare against.
    Returns:
        float: The percentage of IDs in the top x of the estimation_df that are also in the top x of the ground_truth_df.
    """

    ground_truth_df = pd.read_csv(ground_truth_df_path)
    inner_joined_df = pd.merge(ground_truth_df, estimation_df, left_on=["ID"], right_on = ["ID"])
    print(inner_joined_df)
    est = inner_joined_df.loc[:, ["ID","estimation"]]
    ground = inner_joined_df.loc[:, ["ID",metric]]

    est = est.sort_values(by=["estimation"], ascending=False)
    ground = ground.sort_values(by=[metric], ascending=False)

    est_head = est.head(x)
    ground_head = ground.head(x)
    count = 0
    for i in ground_head["ID"].values:
        if i in est_head["ID"].values:
            count += 1
    return (count/x)*100
    





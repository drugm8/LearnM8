import pandas as pd


def top_x_of_x_percentage(ground_truth_df_path, estimation_df, x, metric):

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
    















    # #!assertion, datatype in colum 0 and values/estimation in column 1


    # ground_truth_df = ground_truth_df.sort_values(by=ground_truth_df.columns[metric], ascending=False)





    # x_input = ground_truth_df.loc[:, ground_truth_df.columns[0]]



    # result_df = pd.DataFrame(x_input, columns=['smiles'])
    # result_df['estimation'] = estimation_df
    # sorted_df = result_df.sort_values(by='estimation', ascending=False)

    # ground_truth_df_head = ground_truth_df.head(x)
    # sorted_df_head = estimation_df.head(x)
    
    # count = 0
    # for i in ground_truth_df_head.index:
    #     if i in sorted_df_head.index:
    #         count += 1
    # return (count/x)*100
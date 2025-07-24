import pandas as pd


def top_x_of_x_percentage(ground_truth_csv_path, model_predictions_df, top_n_compounds, ground_truth_column, score_direction='higher'):
    """
    Evaluates active learning model performance by measuring overlap in top-ranked compounds.
    
    Calculates the percentage of compounds in the model's top-N predictions that are also 
    in the top-N compounds according to ground truth data. This is a key evaluation metric
    for active learning in drug discovery - measuring how well the model identifies the same
    high-value compounds that experimental data would reveal.
    
    Args:
        ground_truth_csv_path (str): Path to the ground truth DataFrame CSV file containing
                                   experimental or reference data with compound IDs and values
        model_predictions_df (pd.DataFrame): DataFrame containing model predictions with 
                                           'ID' column and 'estimation' column of predicted values
        top_n_compounds (int): Number of top-ranked compounds to compare between predictions
                             and ground truth (e.g., top 100, top 1000)
        ground_truth_column (str): Column name in the ground truth DataFrame to use as the
                                 reference metric for ranking compounds
        score_direction (str): 'higher' for higher-is-better scores, 'lower' for lower-is-better scores
    
    Returns:
        float: Percentage (0-100) of compounds that appear in both the model's top-N predictions
               and the ground truth's top-N compounds. Higher values indicate better model
               performance at identifying truly valuable compounds.
               
    Example:
        >>> predictions = pd.DataFrame({
        ...     'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
        ...     'estimation': [0.9, 0.7, 0.8]
        ... })
        >>> overlap = top_x_of_x_percentage('ground_truth.csv', predictions, 100, 'Activity', 'higher')
        >>> print(f"Model identifies {overlap}% of true top compounds")
    """
    # Load ground truth data and merge with model predictions
    ground_truth_dataframe = pd.read_csv(ground_truth_csv_path)
    merged_data = pd.merge(ground_truth_dataframe, model_predictions_df, left_on=["ID"], right_on=["ID"])
    print(merged_data)
    
    # Extract model predictions and ground truth values for comparison
    model_predictions_data = merged_data.loc[:, ["ID", "estimation"]]
    ground_truth_data = merged_data.loc[:, ["ID", ground_truth_column]]

    # Sort both datasets by their respective values according to score direction
    # For 'lower' direction (e.g., docking scores), ascending=True selects lowest (best) values first
    # For 'higher' direction (e.g., activity scores), ascending=False selects highest (best) values first
    ascending_order = (score_direction == 'lower')
    model_predictions_sorted = model_predictions_data.sort_values(by=["estimation"], ascending=ascending_order)
    ground_truth_sorted = ground_truth_data.sort_values(by=[ground_truth_column], ascending=ascending_order)

    # Get the top N compounds from each ranking
    top_predicted_compounds = model_predictions_sorted.head(top_n_compounds)
    top_actual_compounds = ground_truth_sorted.head(top_n_compounds)
    
    # Count how many compounds appear in both top-N lists
    matching_compounds_count = 0
    for compound_id in top_actual_compounds["ID"].values:
        if compound_id in top_predicted_compounds["ID"].values:
            matching_compounds_count += 1
            
    # Return as percentage
    overlap_percentage = (matching_compounds_count / top_n_compounds) * 100
    return overlap_percentage
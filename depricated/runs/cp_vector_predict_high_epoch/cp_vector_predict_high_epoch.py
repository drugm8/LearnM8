import pandas as pd
import numpy as np
from helpers.query_functions import greedy_query_function, random_query_function
from helpers.scoring_metric import top_x_of_x_percentage



from learners.learner_abc import learner
import pandas as pd

import numpy as np

from lightning import pytorch as pl

from chemprop import data, featurizers, models, nn


from scripts.consensus.consensus_wrapper import consensus_wrapper as consensus
from scripts.consensus.consensus_wrapper import merge_consensus as merge_consensus
from helpers.dock import dock
from helpers.helpers import initialize_logging, log_and_save, log_list, remove_right_df_from_left_df


import pandas as pd
from pathlib import Path

from lightning import pytorch as pl


from chemprop import data, featurizers, models, nn

import torch

#!IM SO SRRY FOR EVERYTHING


def do_chempop_gpu(smiles, ys):

   

    # Increase num_workers based on your CPU cores
    num_workers = 8  # Adjust based on your system

    if (torch.cuda.is_available()):
        print("GPU available")
    else:
        print("GPU NOT available")
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data points
    ys_array = ys.values #! only important line to add for vector input

    # Create data points
    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smiles, ys_array)]
    train_data = all_data
    print(train_data)

    # Use a more advanced featurizer if available
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # Create dataset and normalize targets
    train_dset = data.MoleculeDataset(train_data, featurizer)
    scaler = train_dset.normalize_targets()

    # Use a larger batch size for GPU
    batch_size = 256  # TODO for training set sizes of 1000 - 10000 something between 64 and 512 should be ok
    train_loader = data.build_dataloader(train_dset, batch_size=batch_size, num_workers=num_workers)

    # Define model components
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform)

    # Use batch normalization
    batch_norm = True

    # Define metrics
    metric_list = [nn.metrics.RMSEMetric(), nn.metrics.MAEMetric()] #

    # Create MPNN model
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    # Create TensorBoard logger
    #tb_logger = TensorBoardLogger("tb_logs", name=(__file__.split("/")[-1]).split(".")[0])#TODO __File__ to better distinguish in tensorboard

    # Configure trainer for GPU with TensorBoard
    trainer = pl.Trainer(
        enable_checkpointing=True,
        enable_progress_bar=False,
        accelerator="gpu",
        devices=1,
        max_epochs=150,  # Increase epochs for better training
        precision="16-mixed",  # Use mixed precision for faster training
        gradient_clip_val=1.0,  # Add gradient clipping to prevent exploding gradients
        accumulate_grad_batches=2,  # Accumulate gradients for larger effective batch size
    )

    # Train the model
    trainer.fit(mpnn, train_loader)



    return mpnn

class chemprop_gpu_learner(learner):
    #SMILES INPUT, they are featurized here
    #my implementation of a learner has the most up to date training set always stored internally

    def __init__(self, query_function, dataset_x, dataset_y, batch_size=32):

        self.query_function = query_function
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.batch_size = batch_size
        print(dataset_y)

        self.mpnn = do_chempop_gpu(smiles= self.dataset_x, ys=self.dataset_y )
        self.name = "chemprop_gpu_high epoch"


    def teach(self, addition_of_dataset_x, addition_of_dataset_y):
        print("teaching...")
        self.dataset_x=np.append(addition_of_dataset_x, self.dataset_x)
        self.dataset_y=pd.concat([self.dataset_y, addition_of_dataset_y], ignore_index=True) #!changed for vector
        self.mpnn = do_chempop_gpu(smiles= self.dataset_x, ys=self.dataset_y ) 
        print("done teaching...")

    
    def query(self, smids_x_input):
        #uses the intrinisc query function to run the inference first and then query the dataset

        estimation = self.estimate(smids_x_input.loc[:,"SMILES"])

        queried = self.query_function(smids_x_input, estimation, batch_size=self.batch_size)
        return queried


    def query_greedy_vector(self, smids_x_input, metric):
        estimation = self.estimate(smids_x_input.loc[:,"SMILES"])
        new_columns = ["CNN-Score","GenScore-scoring","ConvexPLR","KORP-PL"]  # replace with your desired column names
        new_df = pd.DataFrame(estimation, columns=new_columns)
        new_df.index = smids_final_input.index
        res = new_df.join(smids_final_input)

        res["Pose ID"] = 'some'
        print(res.columns)
        for index, row in res.iterrows():
            res.at[index, "Pose ID"] = row["ID"] + "_GNINA_1"

        consensiert = consensus(res, met)
        consensiert = consensiert.rename(columns={'consensus': 'estimation'})
        cons=consensiert.sort_values(by="estimation", ascending=False)
        queried = cons.head(self.batch_size)
        return queried

    def estimate(self, x_input):
        print("estimating...")
        #takes a dataframe of smiles and returns a dataframe with the estimation

        test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in x_input]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        test_dset = data.MoleculeDataset(test_data, featurizer)
        test_loader = data.build_dataloader(test_dset, num_workers=8, batch_size=256, shuffle=False) #!!!! ohmygod shuffle
        #tb_logger = TensorBoardLogger("tb_logs", name=(__file__.split("/")[-1]).split(".")[0]+" inference mode")

        with torch.inference_mode():
            trainerr = pl.Trainer(

                enable_progress_bar=False,
                accelerator="gpu",
                devices=1
            )

        predictions = trainerr.predict(self.mpnn, test_loader)
        ret = np.concatenate(predictions, axis=0)
        #flat_estimations = [item.item() for sublist in predictions for item in sublist]
        print("done estimating...")
        return ret
    
    def estimate_whole(self, x_input):
        print("estimating...")
        #TODO here i could recylce the dataloader for whole dataset

    def print_inner_data(self):
        print("datax", self.dataset_x)
        print("datay", self.dataset_y)

    def cleanup(self):

        self.mpnn = None
        self.dataset_x = None
        self.dataset_y = None
        print("cleaned up")
        

BATCH_SIZE = 1000
AL_CYCLES = 10
TOPX = 5000

metrics_full = ['ECR_avg_scaled', 'ECR_best_scaled', 'RbR_avg_scaled', 'RbR_best_scaled', 'RbV_avg_scaled', 'RbV_best_scaled', 'Zscore_avg_scaled', 'Zscore_best_scaled', 'Pareto_rank_avg_scaled', 'Pareto_rank_best_scaled', 'TOPSIS_avg_scaled', 'TOPSIS_best_scaled', 'WeightedSumModel_avg_scaled', 'WeightedSumModel_best_scaled']
metrics = ['WeightedSumModel_best_scaled']
# 'Zscore_avg_scaled' removed bc of memory crash
log_file = initialize_logging(__file__)


def evaluate_learner(learner):
    prediction_df = full_smids_final_input
    prediction_df = learner.estimate(full_smids_final_input.loc[:,"SMILES"])#?unsure if .values here would be right
    print(prediction_df)

    #!CHANGED FOR VECTOR INPUT

    new_columns = ["CNN-Score","GenScore-scoring","ConvexPLR","KORP-PL"]  # replace with your desired column names
    new_df = pd.DataFrame(prediction_df, columns=new_columns)
    new_df.index = full_smids_final_input.index
    res = new_df.join(full_smids_final_input)
    print (res)
    res["Pose ID"] = 'some'
    print(res.columns)
    for index, row in res.iterrows():
        res.at[index, "Pose ID"] = row["ID"] + "_GNINA_1"
    print (res)
    rescpy = res.copy()
    consensiert = consensus(res, met)
    consensiert = consensiert.rename(columns={'consensus': 'estimation'})
    consensiert=consensiert.loc[:,["ID","estimation"]]
    res = pd.merge(rescpy, consensiert, on="ID")
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(consens)

    top_x_score = top_x_of_x_percentage(ground_truth_df_path, consensiert, TOPX, metric=met)
    return top_x_score


ground_truth_df_path = "./data/data_raw.csv"
for met in metrics:
    smids_final_input = pd.read_csv('./data/final_input.csv')
    #smids_final_input = smids_final_input.head(120)
    full_smids_final_input = smids_final_input.copy()
    inital_random_sample = random_query_function(smids_final_input, None, BATCH_SIZE)
    smids_final_input = remove_right_df_from_left_df(smids_final_input, inital_random_sample)
    docked_inital_random_sample  = dock(ground_truth_df_path, inital_random_sample)
    rescpy = docked_inital_random_sample.copy()
    consens = merge_consensus(docked_inital_random_sample, met)

    consensiert = consens
    consensiert = consensiert.rename(columns={'consensus': 'estimation'})
    consensiert=consensiert.loc[:,["ID","estimation"]]
    res = pd.merge(rescpy, consensiert, on="ID")

    #!continue with res
    learner = chemprop_gpu_learner(greedy_query_function, res.loc[:,"SMILES"].values, res.loc[:,["CNN-Score","GenScore-scoring","ConvexPLR","KORP-PL"]], batch_size=BATCH_SIZE)
    log_and_save(f"Batch size: {BATCH_SIZE}; active learning cycles: {AL_CYCLES}; top X of X percentage score: {TOPX}; Machine learning architecture:{learner.getName()}; Consensus Method:{met};", log_file)
    topxlist = []
    topxlist.append(evaluate_learner(learner))
    for i in range(AL_CYCLES):
        print("cycle", i)
        smids_queried = learner.query_greedy_vector(smids_final_input, met)
        smids_final_input = remove_right_df_from_left_df(smids_final_input, smids_queried)
        docked_smids_queried = dock(ground_truth_df_path, smids_queried)
        consens_queried = merge_consensus(docked_smids_queried, met)

        consensiert = consens_queried
        consensiert = consensiert.rename(columns={'consensus': 'estimation'})
        consensiert=consensiert.loc[:,["ID","estimation"]]
        res = pd.merge(rescpy, consensiert, on="ID")

        learner.teach(res.loc[:,"SMILES"].values, res.loc[:,["CNN-Score","GenScore-scoring","ConvexPLR","KORP-PL"]])
        topxlist.append(evaluate_learner(learner))
        print(topxlist)
    log_list(topxlist, log_file)
    learner.cleanup()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


log_file.close()

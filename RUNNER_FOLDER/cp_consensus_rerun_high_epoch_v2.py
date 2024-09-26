
import pandas as pd
import numpy as np
from helpers.query_functions import greedy_query_function, random_query_function
from helpers.scoring_metric import top_x_of_x_percentage


from abc import ABC, abstractmethod
from learners.learner_abc import learner
import pandas as pd
from pathlib import Path
import numpy as np

from lightning import pytorch as pl

from chemprop import data, featurizers, models, nn


from scripts.consensus.consensus_wrapper import consensus_wrapper as consensus
from helpers.dock import dock
from helpers.helpers import initialize_logging, log_and_save, log_list, remove_right_df_from_left_df


import pandas as pd
from pathlib import Path

from lightning import pytorch as pl
from pytorch_lightning.loggers import TensorBoardLogger

from chemprop import data, featurizers, models, nn

import torch

def do_chempop_gpu(smiles, ys):
    # Reshape ys
    yss = ys.reshape(-1, 1)

    # Increase num_workers based on your CPU cores
    num_workers = 8  # Adjust based on your system

    if (torch.cuda.is_available()):
        print("GPU available")
    else:
        print("GPU NOT available")
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data points
    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smiles, yss)]
    train_data = all_data

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
    tb_logger = TensorBoardLogger("tb_logs", name=(__file__.split("/")[-1]).split(".")[0])#TODO __File__ to better distinguish in tensorboard

    # Configure trainer for GPU with TensorBoard
    trainer = pl.Trainer(
        logger=tb_logger,
        enable_checkpointing=False,
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



    return trainer, mpnn

class chemprop_gpu_learner(learner):
    #SMILES INPUT, they are featurized here
    #my implementation of a learner has the most up to date training set always stored internally

    def __init__(self, query_function, dataset_x, dataset_y, batch_size=32):

        self.query_function = query_function
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.batch_size = batch_size
        self.trainer, self.mpnn = do_chempop_gpu(smiles= self.dataset_x, ys=self.dataset_y ) #!ys can be multiple i think
        self.name = "chemprop_gpu_high epoch"


    def teach(self, addition_of_dataset_x, addition_of_dataset_y):
        print("teaching...")
        self.dataset_y=np.append(addition_of_dataset_y, self.dataset_y)
        self.dataset_x=np.append(addition_of_dataset_x, self.dataset_x)
        self.trainer, self.mpnn = do_chempop_gpu(smiles= self.dataset_x, ys=self.dataset_y )
        print("done teaching...")

    
    def query(self, smids_x_input):
        #uses the intrinisc query function to run the inference first and then query the dataset

        estimation = self.estimate(smids_x_input.loc[:,"SMILES"])

        queried = self.query_function(smids_x_input, estimation, batch_size=self.batch_size)
        return queried


    def estimate(self, x_input):
        print("estimating...")
        #takes a dataframe of smiles and returns a dataframe with the estimation
        if self.trainer is None:
            raise Exception("trainer not trained")
        test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in x_input]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        test_dset = data.MoleculeDataset(test_data, featurizer)
        test_loader = data.build_dataloader(test_dset, num_workers=8, batch_size=256, shuffle=False) #!!!! ohmygod shuffle
        tb_logger = TensorBoardLogger("tb_logs", name=(__file__.split("/")[-1]).split(".")[0]+" inference mode")
        with torch.inference_mode():
            trainerr = pl.Trainer(
                logger=tb_logger,
                enable_progress_bar=False,
                accelerator="gpu",
                devices=1
            )

        predictions = trainerr.predict(self.mpnn, test_loader)
        ret = np.concatenate(predictions, axis=0)
        #flat_estimations = [item.item() for sublist in predictions for item in sublist]
        print("done estimating...")
        return ret
    

    def print_inner_data(self):
        print("datax", self.dataset_x)
        print("datay", self.dataset_y)

BATCH_SIZE = 1000
AL_CYCLES = 10
TOPX = 5000

metrics = ['ECR_avg_scaled', 'ECR_best_scaled', 'RbR_avg_scaled', 'RbR_best_scaled', 'RbV_avg_scaled', 'RbV_best_scaled', 'Zscore_avg_scaled', 'Zscore_best_scaled', 'Pareto_rank_avg_scaled', 'Pareto_rank_best_scaled', 'TOPSIS_avg_scaled', 'TOPSIS_best_scaled', 'WeightedSumModel_avg_scaled', 'WeightedSumModel_best_scaled']
# Open log file for writing
log_file = initialize_logging(__file__)


def evaluate_learner(learner):
    prediction_df = full_smids_final_input
    prediction_df["estimation"] = learner.estimate(full_smids_final_input.loc[:,"SMILES"])#?unsure if .values here would be right
    top_x_score = top_x_of_x_percentage(ground_truth_df_path, prediction_df, TOPX, metric=met)
    return top_x_score


ground_truth_df_path = "./data/data_raw.csv"
for met in metrics:
    smids_final_input = pd.read_csv('./data/final_input.csv')
    full_smids_final_input = smids_final_input.copy()
    inital_random_sample = random_query_function(smids_final_input, None, BATCH_SIZE)
    smids_final_input = remove_right_df_from_left_df(smids_final_input, inital_random_sample)
    docked_inital_random_sample  = dock(ground_truth_df_path, inital_random_sample)
    consens = consensus(docked_inital_random_sample, met)
    learner = chemprop_gpu_learner(greedy_query_function, consens.loc[:,"SMILES"].values, consens.loc[:,"consensus"].values, batch_size=BATCH_SIZE)
    log_and_save(f"Batch size: {BATCH_SIZE}; active learning cycles: {AL_CYCLES}; top X of X percentage score: {TOPX}; Machine learning architecture:{learner.getName()}; Consensus Method:{met};", log_file)
    topxlist = []
    topxlist.append(evaluate_learner(learner))
    for i in range(AL_CYCLES):
        print("cycle", i)
        smids_queried = learner.query(smids_final_input)
        smids_final_input = remove_right_df_from_left_df(smids_final_input, smids_queried)
        docked_smids_queried = dock(ground_truth_df_path, smids_queried)
        consens_queried = consensus(docked_smids_queried, met)
        learner.teach(consens_queried.loc[:,"SMILES"].values, consens_queried.loc[:,"consensus"].values)
        topxlist.append(evaluate_learner(learner))
        print(topxlist)
    log_list(topxlist, log_file)



log_file.close()

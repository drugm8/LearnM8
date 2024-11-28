from learners.learner_abc import learner
import chemprop
import pandas as pd
import numpy as np
import torch
import os
import gc

from lightning import pytorch as pl

from chemprop import data, featurizers, models, nn

from pathlib import Path

import pandas as pd
from lightning import pytorch as pl
import ray
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (RayDDPStrategy, RayLightningEnvironment,
                                 RayTrainReportCallback, prepare_trainer)
from ray.train.torch import TorchTrainer
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import FIFOScheduler

from chemprop import data, featurizers, models, nn

class pipe_cp_learner(learner):

    def __init__(self, max_out_system = True):
        self.name = "chemprop learner"
        self.query_function = None
        self.batch_size = None
        self.mpnn = None
        self.dataset_x = None
        self.dataset_y = None

   

        self.accelerator = None
        self.config = {'depth': 2, #todo update to default
            'ffn_hidden_dim': 2200,
            'ffn_num_layers': 2,
            'message_hidden_dim': 400}
        
        


        if max_out_system:
            if os.cpu_count() > 32:
                self.cpu_cores = 32
            else:
                self.cpu_cores = os.cpu_count()

            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                self.accelerator = "gpu"
            else:
                self.accelerator = "cpu"
        else: 
            self.cpu_cores = 1
            self.cuda_available = False
            self.accelerator = "cpu"

        self.trainer = pl.Trainer(
        enable_checkpointing=True,
        enable_progress_bar=False,
        accelerator=self.accelerator,
        devices=1,
        max_epochs=100,  # Increase epochs for better training
        gradient_clip_val=0.5, #dunno 
        accumulate_grad_batches=4, #about 
        precision=16 #these values
        )
        print("trainer initialized")
        print("accelerator used:"+self.accelerator)
        print("cpu cores used:"+str(self.cpu_cores))


    def teach(self, addition_of_dataset_x, addition_of_dataset_y):
        self.append_data(addition_of_dataset_x, addition_of_dataset_y)

        self.train_mpnn_on_internal()


    



    def estimate(self, x_input):
        test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in x_input]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        test_dset = data.MoleculeDataset(test_data, featurizer)
        test_loader = data.build_dataloader(test_dset, num_workers=8, batch_size=256, shuffle=False) #!!!! ohmygod shuffle
        with torch.inference_mode():
            inference_trainer = pl.Trainer(
                enable_checkpointing=True,
                enable_progress_bar=False,
                accelerator="auto",
                devices=1
            )
        predictions = inference_trainer.predict(self.mpnn, test_loader)

        gc.collect()
        return np.concatenate(predictions, axis=0)
    
    def train_mpnn_on_internal(self):
        print("training...")
        yss = self.dataset_y.reshape(-1, 1)
        print("traing dimensions:", yss.shape)
        all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(self.dataset_x, yss)]

        # Use a more advanced featurizer if available
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        # Create dataset and normalize targets
        train_dset = data.MoleculeDataset(all_data, featurizer)
        scaler = train_dset.normalize_targets()

        # Use a larger batch size for GPU
        batch_size = 256  #for training set sizes of 1000 - 10000 something between 64 and 512 should be ok
        train_loader = data.build_dataloader(train_dset, batch_size=batch_size, num_workers=self.cpu_cores)

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

        # Train the model
        print("fitting...")
        self.trainer.fit(mpnn, train_loader)


        self.mpnn = mpnn
        print("done training...")
        gc.collect()

    def get_and_remove_internal_predictions(self):
        if not os.path.exists("./internal_al_cache/"):
            raise Exception("No internal cache found")
        return_val = pd.read_csv("./internal_al_cache/cache.csv")
        #os.remove("./internal_al_cache/cache.csv")
        #os.rmdir("./internal_al_cache/")
        return return_val


    
    def optimize_hyperparameters(self):
        
        hpopt_save_dir = Path.cwd() / "hpopt" # directory to save hyperopt results
        hpopt_save_dir.mkdir(exist_ok=True)
        all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(self.dataset_x, self.dataset_y.reshape(-1, 1))]
        mols = [d.mol for d in all_data]  # RDkit Mol objects are use for structure based splits

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        train_dset = data.MoleculeDataset(all_data, featurizer)
        scaler = train_dset.normalize_targets()

        def train_model(config, train_dset, num_workers, scaler):
            
            # config is a dictionary containing hyperparameters used for the trial
            depth = int(config["depth"])
            ffn_hidden_dim = int(config["ffn_hidden_dim"])
            ffn_num_layers = int(config["ffn_num_layers"])
            message_hidden_dim = int(config["message_hidden_dim"])

            train_loader = data.build_dataloader(train_dset, num_workers=num_workers, shuffle=True)

            mp = nn.BondMessagePassing(d_h=message_hidden_dim, depth=depth)
            agg = nn.MeanAggregation()
            output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
            ffn = nn.RegressionFFN(output_transform=output_transform, input_dim=message_hidden_dim, hidden_dim=ffn_hidden_dim, n_layers=ffn_num_layers)
            batch_norm = True
            metric_list = [nn.metrics.RMSEMetric(), nn.metrics.MAEMetric()]
            model = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

            trainer = pl.Trainer(
                accelerator="auto",
                devices=1,
                max_epochs=20, # number of epochs to train for
                # below are needed for Ray and Lightning integration
                strategy=RayDDPStrategy(),
                callbacks=[RayTrainReportCallback()],
                plugins=[RayLightningEnvironment()],
            )

            trainer = prepare_trainer(trainer)
            trainer.fit(model, train_loader)


        search_space = {
            "depth": tune.qrandint(lower=2, upper=6, q=1),
            "ffn_hidden_dim": tune.qrandint(lower=300, upper=2400, q=100),
            "ffn_num_layers": tune.qrandint(lower=1, upper=3, q=1),
            "message_hidden_dim": tune.qrandint(lower=300, upper=2400, q=100),
        }
        ray.init()

        scheduler = FIFOScheduler()

        # Scaling config controls the resources used by Ray
        if self.cuda_available:
            scaling_config = ScalingConfig(
                num_workers=1,#Todo 
                use_gpu=True, # change to True if you want to use GPU
            )
        else:
            scaling_config = ScalingConfig(
                num_workers=self.cpu_cores,
                use_gpu=False, # change to True if you want to use GPU
            )

        # Checkpoint config controls the checkpointing behavior of Ray
        checkpoint_config = CheckpointConfig(
            num_to_keep=1, # number of checkpoints to keep
            checkpoint_score_attribute="train_loss", # Save the checkpoint based on this metric
            checkpoint_score_order="min", # Save the checkpoint with the lowest metric value
        )

        run_config = RunConfig(
            checkpoint_config=checkpoint_config,
            storage_path=hpopt_save_dir / "ray_results", # directory to save the results
        )

        ray_trainer = TorchTrainer(
            lambda config: train_model(
                config, train_dset, self.cpu_cores, scaler
            ),
            scaling_config=scaling_config,
            run_config=run_config,
        )

        search_alg = HyperOptSearch(
            n_initial_points=1, # number of random evaluations before tree parzen estimators
            random_state_seed=42,
        )

        # OptunaSearch is another search algorithm that can be used
        # search_alg = OptunaSearch() 

        tune_config = tune.TuneConfig(
            metric="train_loss",
            mode="min",
            num_samples=2, # number of trials to run
            scheduler=scheduler,
            search_alg=search_alg,
            trial_dirname_creator=lambda trial: str(trial.trial_id), # shorten filepaths

        )

        tuner = tune.Tuner(
            ray_trainer,
            param_space={
                "train_loop_config": search_space,
            },
            tune_config=tune_config,
        )

        # Start the hyperparameter search
        results = tuner.fit()
        best_result = results.get_best_result()
        best_config = best_result.config
        self.config=best_config['train_loop_config']
        best_checkpoint_path = Path(best_result.checkpoint.path) / "checkpoint.ckpt"
        mpnn_from_checkpoint = torch.load(best_checkpoint_path)#like this way up here
        ray.shutdown()
        gc.collect()
        return mpnn_from_checkpoint #todo check if this is the way
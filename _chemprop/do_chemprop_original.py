import pandas as pd
from pathlib import Path

from lightning import pytorch as pl

from chemprop import data, featurizers, models, nn

def do_chempop(smiles, ys ):
    #!careful target columns was a list of strings
    yss=ys.reshape(-1, 1)

    num_workers = 8 # number of workers for dataloader. 0 means using main process for data loading
    
    


    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smiles, yss)]
    #!some fucky object floating around that is my new datapoint
    #TODO literature research what the f im inputting

    list(data.SplitType.keys())

    train_data = all_data

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dset = data.MoleculeDataset(train_data, featurizer)
    scaler = train_dset.normalize_targets()

    # val_dset = data.MoleculeDataset(val_data, featurizer)
    # val_dset.normalize_targets(scaler)


    train_loader = data.build_dataloader(train_dset, num_workers=num_workers)

    mp = nn.BondMessagePassing()


    print(nn.agg.AggregationRegistry)


    agg = nn.MeanAggregation()
    print(nn.PredictorRegistry)

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

    ffn = nn.RegressionFFN(output_transform=output_transform)

    batch_norm = True


    print(nn.metrics.MetricRegistry)

    metric_list = [nn.metrics.RMSEMetric(), nn.metrics.MAEMetric()] # Only the first metric is used for training and early stopping

    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        max_epochs=20, # number of epochs to train for
    )

    trainer.fit(mpnn, train_loader)

    return trainer, mpnn


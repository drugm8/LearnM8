import pandas as pd
from pathlib import Path
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn
from sklearn.model_selection import train_test_split


#IMPROVING CHEMPROP TRIAL
def do_chempop(smiles, ys):
    yss = ys.reshape(-1, 1)
    num_workers = 8

    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smiles, yss)]

    # Split data into train and validation sets
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)

    # Use a more advanced featurizer
    featurizer = featurizers.MolGraphFeaturizer(include_charge=True, include_chirality=True)

    train_dset = data.MoleculeDataset(train_data, featurizer)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_data, featurizer)
    val_dset.normalize_targets(scaler)

    train_loader = data.build_dataloader(train_dset, num_workers=num_workers, batch_size=32)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, batch_size=32)

    # Use more complex message passing and aggregation
    mp = nn.BondMessagePassing(hidden_size=300, depth=5)
    agg = nn.AttentionAggregation()

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

    # Use a more complex FFN
    ffn = nn.RegressionFFN(hidden_size=300, num_layers=3, dropout=0.1, output_transform=output_transform)

    batch_norm = True

    metric_list = [nn.metrics.RMSEMetric(), nn.metrics.MAEMetric(), nn.metrics.R2Metric()]

    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    # Learning rate scheduler
    lr_scheduler = pl.optimizers.lr_scheduler.ReduceLROnPlateau(
        mode='min', factor=0.1, patience=5, min_lr=1e-6
    )

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        max_epochs=100,  # Increased number of epochs
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=10),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        ],
    )

    trainer.fit(mpnn, train_loader, val_loader)

    return trainer, mpnn
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
    batch_size = 256  # Adjust based on your GPU memory
    train_loader = data.build_dataloader(train_dset, batch_size=batch_size, num_workers=num_workers)

    # Define model components
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform)

    # Use batch normalization
    batch_norm = True

    # Define metrics
    metric_list = [nn.metrics.RMSEMetric(), nn.metrics.MAEMetric()]

    # Create MPNN model
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    # Create TensorBoard logger
    tb_logger = TensorBoardLogger("tb_logs", name="chemprop_model")

    # Configure trainer for GPU with TensorBoard
    trainer = pl.Trainer(
        logger=tb_logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="gpu",
        devices=1,
        max_epochs=50,  # Increase epochs for better training
        precision="16-mixed",  # Use mixed precision for faster training
        gradient_clip_val=1.0,  # Add gradient clipping to prevent exploding gradients
        accumulate_grad_batches=2,  # Accumulate gradients for larger effective batch size
    )

    # Train the model
    trainer.fit(mpnn, train_loader)



    return trainer, mpnn
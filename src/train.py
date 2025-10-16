import os
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from model import UNet
from datamodule import FluidFlowDataModule
import torch
from utils import log_predictions


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("------------------------------------------------")
    print("Configuration:\n", cfg)
    print("------------------------------------------------")
    pl.seed_everything(42)

    # Model and data
    model = UNet(**cfg.model)
    datamodule = FluidFlowDataModule(**cfg.dataset)

    # MLflow logger
    mlf_logger = MLFlowLogger(experiment_name=cfg.experiment_name)

    # Trainer
    trainer = pl.Trainer(logger=mlf_logger, **cfg.trainer)

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Post-training
    print("\n---------------------------------------------------------")
    print("Starting post-training phase...")

    # Save the normalizer
    normalizer = datamodule.normalizer
    if normalizer is not None:
        os.makedirs("artifacts", exist_ok=True)
        normalizer.save("artifacts/normalizer.npz")
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, "artifacts/normalizer.npz")
        os.remove("artifacts/normalizer.npz")

    # Visualization of predictions on validation set
    val_loader = datamodule.val_dataloader()
    batch = next(iter(val_loader))
    x, y = batch
    max_samples = min(x.shape[0], 8)
    x, y = x[:max_samples], y[:max_samples]
    model.eval()
    with torch.no_grad():
        output = model(x)

    log_predictions(
        normalizer.inverse_transform(x),
        normalizer.inverse_transform(y),
        normalizer.inverse_transform(output),
        mlf_logger)


if __name__ == "__main__":
    main()
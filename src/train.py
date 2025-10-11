import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from model import UNet
from datamodule import FluidFlowDataModule


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


if __name__ == "__main__":
    main()
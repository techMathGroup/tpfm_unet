import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from src.model import UNet
from src.datamodule import FluidFlowDataset


@hydra.main(config_path="../configs", config_name="config", version_base="1.0")
def main(cfg: DictConfig):
    # Model and data
    model = UNet(**cfg.model)
    datamodule = FluidFlowDataset(**cfg.data)

    # MLflow logger
    mlf_logger = MLFlowLogger(experiment_name=cfg.experiment_name)

    # Trainer
    trainer = pl.Trainer(logger=mlf_logger, **cfg.trainer)

    # Train
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
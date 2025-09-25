import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


class FluidFlowDataset(pl.LightningDataModule):
    def __init__(self,
                 training_path,
                 validation_path,
                 batch_size=16,
                 num_workers=4,
                 max_samples=None
                 ):
        super().__init__()
        self.training_path = training_path
        self.validation_path = validation_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.max_samples = max_samples

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # Load training data
        # TODO: Replace with actual data loading logic
        x_train = torch.randn(100, 1, 64, 64)
        y_train = torch.randn(100, 4, 64, 64)
        x_val = torch.randn(20, 1, 64, 64)
        y_val = torch.randn(20, 4, 64, 64)

        # Limit to max_samples if specified (0 means no limit)
        if self.max_samples is not None and self.max_samples != 0 and self.max_samples < len(x_train):
            x_train = x_train[:self.max_samples]
            y_train = y_train[:self.max_samples]

        self.train_dataset = TensorDataset(x_train, y_train)
        self.val_dataset = TensorDataset(x_val, y_val)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
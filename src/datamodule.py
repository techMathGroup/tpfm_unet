import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
import numpy as np


class FluidFlowDataset(Dataset):
    def __init__(self, x, y, normalizer=None):
        self.x_data, self.y_data = x, y
        self.normalizer = normalizer

        if self.normalizer:
            # Only normalize y, as x is a lambda field
            # Fit normalizer on x_data if not already fitted
            if self.normalizer.mean is None or self.normalizer.std is None:
                self.normalizer.fit(self.y_data)
            self.y_data = self.normalizer.transform(self.y_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


class FluidFlowDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_file,
                 batch_size=64,
                 val_split=0.2,
                 num_workers=4,
                 max_samples=None,
                 normalize=True
                 ):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.max_samples = max_samples

        self.train_dataset = None
        self.val_dataset = None
        self.val_split = val_split

        self.normalizer = Normalizer() if normalize else None

    def setup(self, stage=None):
        # Load data from a file
        data = np.load(self.data_file)
        x, y = data["inputs"].astype(np.float32), data["outputs"].astype(np.float32)
        if self.max_samples:
            x, y = x[:self.max_samples], y[:self.max_samples]

        dataset_full = FluidFlowDataset(x, y, normalizer=self.normalizer)
        val_size = int(len(dataset_full) * self.val_split)
        train_size = len(dataset_full) - val_size
        self.train_dataset, self.val_dataset = random_split(dataset_full, [train_size, val_size])

        print(f"Dataset sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )


class Normalizer:
    def __init__(self, mean=None, std=None, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def fit(self, data):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        self.mean = np.mean(data, axis=(0, 2, 3), keepdims=True)
        self.std = np.std(data, axis=(0, 2, 3), keepdims=True)
        return self

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fitted before calling transform.")
        return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fitted before calling inverse_transform.")
        return data * (self.std + self.eps) + self.mean

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std, eps=self.eps)

    @classmethod
    def load(cls, path):
        npz = np.load(path)
        return cls(mean=npz['mean'], std=npz['std'], eps=npz['eps'])
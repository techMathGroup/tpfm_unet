import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError

from utils import pad_to_even, center_crop, mask_lambda


class UNet(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 mean,
                 std,
                 lr=1e-3
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['mean', 'std'])

        # Store mean and std as buffers for device management
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        self.enc1 = self._convolution_block(in_channels, 16)
        self.enc2 = self._convolution_block(16, 32)
        self.enc3 = self._convolution_block(32, 64)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self._convolution_block(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._convolution_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._convolution_block(64, 32)

        self.up0 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec0 = self._convolution_block(32, 16)

        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

        # Loss function
        # TODO: add physics loss
        self.loss_fn = nn.MSELoss()
        self.metric = MeanSquaredError()

    def _convolution_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=self.hparams.kernel_size,
                      padding=self.hparams.padding, padding_mode='replicate'),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=self.hparams.kernel_size,
                      padding=self.hparams.padding, padding_mode='replicate'),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def denormalize_output(self, sample, eps=1e-6):
        return sample * (self.std.view(-1, 1, 1) + eps) + self.mean.view(-1, 1, 1)

    def forward(self, x, debug=False, denormalize=False):
        y = x.clone()

        e1 = pad_to_even(self.enc1(x))
        if debug:
            print(f"e1: {e1.shape}")

        e2_in = self.pool(e1)
        e2 = pad_to_even(self.enc2(e2_in))
        if debug:
            print(f"e2: {e2.shape}")

        e3_in = self.pool(e2)
        e3 = pad_to_even(self.enc3(e3_in))
        if debug:
            print(f"e3: {e3.shape}")

        b_in = self.pool(e3)
        b = self.bottleneck(b_in)
        if debug:
            print(f"bottleneck: {b.shape}")

        up2 = self.up2(b)
        up2 = center_crop(up2, e3)
        d2 = self.dec2(torch.cat([up2, e3], dim=1))

        up1 = self.up1(d2)
        up1 = center_crop(up1, e2)
        d1 = self.dec1(torch.cat([up1, e2], dim=1))

        up0 = self.up0(d1)
        up0 = center_crop(up0, e1)
        d0 = self.dec0(torch.cat([up0, e1], dim=1))

        out = self.final(d0)
        out = mask_lambda(out, y)

        if denormalize:
            out = self.denormalize_output(out)

        return out

    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning. Computes the loss and logs it.
        """
        x, y = batch
        # Get model prediction
        y_hat = self(x)
        # Compute loss
        loss = self.loss_fn(y_hat, y)
        # Log loss
        self.log('training_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for PyTorch Lightning. Computes the loss and logs it.
        """
        x, y = batch
        # Get model prediction
        y_hat = self(x)
        # Compute loss
        loss = self.loss_fn(y_hat, y)
        # Log loss
        self.log('validation_loss', loss)
        self.log('validation_mse', self.metric(y_hat, y))

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

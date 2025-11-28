import torch
import torch.nn as nn
import pytorch_lightning as pl
# from torchmetrics import MeanSquaredError


class UNet(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_filters=16,
                 depth=3,
                 kernel_size=3,
                 padding=1,
                 learning_rate=1e-3
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['mean', 'std'])

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        self.kernel_size = kernel_size
        self.padding = padding
        self.learning_rate = learning_rate

        # Store mean and std as buffers for device management
        # self.register_buffer('mean', mean)
        # self.register_buffer('std', std)

        # Build encoder blocks
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for i in range(depth):
            out_channels_enc = base_filters * (2 ** i)
            self.encoders.append(self._convolution_block(prev_channels, out_channels_enc))
            prev_channels = out_channels_enc

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self._convolution_block(base_filters * (2 ** (depth - 1)), base_filters * (2 ** depth))

        # Build decoder blocks
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            in_channels_dec = base_filters * (2 ** (i + 1))
            out_channels_dec = base_filters * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(in_channels_dec, out_channels_dec,
                                                   kernel_size=2, stride=2))
            self.decoders.append(self._convolution_block(in_channels_dec, out_channels_dec))

        # Final convolution
        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

        self.smoothing = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=5,
            padding=2,
            groups=out_channels,
            bias=False,
            padding_mode='replicate'
        )
        self.init_gaussian_kernel(self.smoothing, 5, 1.0)

        # Loss function
        # TODO: add physics loss
        self.mse_loss = nn.MSELoss()
        # self.metric = MeanSquaredError()

    @staticmethod
    def init_gaussian_kernel(layer, k, sigma):
        ax = torch.arange(k) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, k, k)
        with torch.no_grad():
            layer.weight.copy_(kernel.repeat(layer.out_channels, 1, 1, 1))

    def _convolution_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size,
                      padding=self.padding, padding_mode='replicate'),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=self.kernel_size,
                      padding=self.padding, padding_mode='replicate'),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def tv_loss(pred):
        # Penalize only for pressure channel (assumed to be channel index 3)
        p = pred[:, 3:4, :, :]
        dx = p[:, :, 1:, :] - p[:, :, :-1, :]
        dy = p[:, :, :, 1:] - p[:, :, :, :-1]
        return dx.abs().mean() + dy.abs().mean()

    @staticmethod
    def laplacian_loss(pred):
        kernel = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32, device=pred.device
        ).view(1, 1, 3, 3)
        # Apply to pressure channel (assumed to be channel index 3)
        lap = torch.nn.functional.conv2d(pred[:, 3:4, :, :], kernel, padding=1, )
        return lap.pow(2).mean()

    def loss_fn(self, pred, target, tv_weight=0.1, lap_weight=0.05):
        mse = self.mse_loss(pred, target)
        tv = tv_weight * self.tv_loss(pred)
        # lap = lap_weight * self.laplacian_loss(pred)
        return mse + tv

    @staticmethod
    def denormalize_output(sample, mean, std, eps=1e-6):
        return sample * (std.view(-1, 1, 1) + eps) + mean.view(-1, 1, 1)

    @staticmethod
    def mask_lambda(x, y, lambda_index=0):
        mask = 1.0 - y[:, lambda_index, :, :]
        x[:, :3, :, :] *= mask.unsqueeze(1)
        return x

    def forward(self, x, debug=False, denormalize=False, mean=None, std=None):
        inputs = x.clone()

        enc_features = []
        out = x
        for i, encoder in enumerate(self.encoders):
            out = encoder(out)  # For weird dimensions, use out = pad_to_even(encoder(out))
            enc_features.append(out)
            if debug:
                print(f"e{i + 1}: {out.shape}")
            out = self.pool(out)

        out = self.bottleneck(out)
        if debug:
            print(f"bottleneck: {out.shape}")

        for i in range(self.depth):
            upconv = self.upconvs[i]
            decoder = self.decoders[i]
            up = upconv(out)
            skip = enc_features[self.depth - 1 - i]
            # for weird dimensions, use up = center_crop(up, skip)
            if debug:
                print(f"up{i + 1}: {up.shape}, skip{i + 1}: {skip.shape}")
            out = decoder(torch.cat([up, skip], dim=1))

        out = self.final(out)
        out = self.smoothing(out)

        if denormalize and (mean is not None) and (std is not None):
            out = self.denormalize_output(out, mean, std)

        out = self.mask_lambda(out, inputs)

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
        # self.log('validation_mse', self.metric(y_hat, y))

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_loss(self, x, y):
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

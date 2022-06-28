import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Point Cloud Model
# https://github.com/nsavinov/semantic3dnet/blob/master/src/point_cloud_model_definition.lua
# https://github.com/nsavinov/semantic3dnet/blob/master/src/small_train.lua


class PointCloudModel(pl.LightningModule):
    def __init__(self, input_size, n_outputs, number_of_filters) -> None:
        super().__init__()

        self.input_size = input_size
        self.n_outputs = n_outputs
        self.number_of_filters = number_of_filters

        self.criterion = nn.CrossEntropyLoss()

        self.convolutional_model = nn.Sequential(
            nn.Conv3d(
                in_channels=self.input_size,
                out_channels=self.number_of_filters,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(
                in_channels=self.number_of_filters,
                out_channels=self.number_of_filters * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(
                in_channels=self.number_of_filters * 2,
                out_channels=self.number_of_filters * 4,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.fcn_muliplier = 128
        self.full_model = nn.Sequential(
            nn.Linear(
                4 * self.number_of_filters * ((self.input_size / 8) ** 3),
                self.fcn_muliplier * self.number_of_filters,
            ),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.fcn_muliplier * self.number_of_filters, self.n_outputs),
        )

    def forward(self, x):
        x = self.convolutional_model(x)
        x = x.view(-1, 4 * self.number_of_filters * ((self.input_size / 8) ** 3))
        # x = x.view(x.size(0), -1)
        x = self.full_model(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return {"loss": loss}

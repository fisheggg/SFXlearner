import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl

class VanillaNN(pl.LightningModule):
    """
    A vanilla linear layer classifier
    """
    def __init__(self, input_dim: int, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, input):
        out = self.model(input)
        return out

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x[:, 1, :].view(x.shape[0], x.shape[2]) # only take the wet audio
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x[:, 1, :].view(x.shape[0], x.shape[2])
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        y_logits = torch.argmax(y_hat, dim=1)
        accuracy = torch.sum(y == y_logits).item() / (len(y) * 1.0)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)

    def test_step(self, val_batch, batch_idx):
        pass
        x, y = val_batch
        x = x[:, 1, :].view(x.shape[0], x.shape[2])
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        y_logits = torch.argmax(y_hat, dim=1)
        accuracy = torch.sum(y == y_logits).item() / (len(y) * 1.0)
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class VanillaNNWithClean(pl.LightningModule):
    """
    A vanilla linear layer classifier that uses both clean and wet audio
    """
    def __init__(self, input_dim: int, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim * 2
        self.num_classes = num_classes
        self.lr = lr

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, input):
        out = self.model(input)
        return out

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.shape[0], -1) # reshape to combine clean and wet MFCCs
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.shape[0], -1) # reshape to combine clean and wet MFCCs
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        y_logits = torch.argmax(y_hat, dim=1)
        accuracy = torch.sum(y == y_logits).item() / (len(y) * 1.0)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.shape[0], -1) # reshape to combine clean and wet MFCCs
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        y_logits = torch.argmax(y_hat, dim=1)
        accuracy = torch.sum(y == y_logits).item() / (len(y) * 1.0)
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
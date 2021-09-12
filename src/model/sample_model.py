import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl

class VanillaLinear(pl.LightningModule):
    """
    A vanilla linear layer classifier
    """
    def __init__(self, input_dim:int, num_classes:int):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )
    
    def forward(self, input):
        out = self.model(input)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        loss = F.cross_entropy(x, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        loss = F.cross_entropy(x, y)
        self.log('val_loss', loss)
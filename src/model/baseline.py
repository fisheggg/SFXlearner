import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


class BaselineMLP(nn.Module):
    """
    A vanilla linear layer classifier
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], -1)  # reshape to combine clean and wet MFCCs
        return self.model(x)

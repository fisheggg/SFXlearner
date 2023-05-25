import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


class VanillaNN(nn.Module):
    """
    A vanilla linear layer classifier
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.shape[0], -1)  # reshape to combine clean and wet MFCCs
        out = self.model(input)
        return out


class VanillaNNWithClean(nn.Module):
    """
    A vanilla linear layer classifier that uses both clean and wet audio
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim * 2
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.shape[0], -1)  # reshape to combine clean and wet MFCCs
        out = self.model(input)
        return out

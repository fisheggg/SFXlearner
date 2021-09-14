import torch
from torch.random import set_rng_state
import torchaudio
from torch import nn

class MFCCSumTransform(nn.Module):
    """
    returns the sum of MFCCs along time.

    input shape: (..., n_mfcc, time_stamp)
    output shape: (..., n_mfcc)
    """

    def __init__(self, sample_rate, n_mfcc, melkwargs=None):
        super().__init__()
        self.MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs=melkwargs)


    def forward(self, input):
        x = self.MFCC(input)
        x = torch.mean(x, dim=-1)
        return x
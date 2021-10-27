import torch
import torchaudio
import librosa
import numpy as np
import torchaudio.functional as F
# import nnAudio.Spectrogram
from torch.random import set_rng_state
from torch import nn

cqt_filter_fft = librosa.constantq.__cqt_filter_fft

class MFCCSumTransform(nn.Module):
    """
    returns the sum of MFCCs along time.

    input shape: (..., n_mfcc, time_stamp)
    output shape: (..., n_mfcc)
    """

    def __init__(self, sample_rate, n_mfcc, **melkwargs):
        super().__init__()
        self.MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs=melkwargs)


    def forward(self, input):
        x = self.MFCC(input)
        x = torch.mean(x, dim=-1)
        return x


class SpectrogramDBTransform(nn.Module):
    """
    Returns the spectrogram in dB scale.
    input shape: (..., channel, time_stamp)
    output shape: (..., channel, time_stamp, n_fft2)
    """

    def __init__(self, n_fft=2048):
        super().__init__()
        self.specer = torchaudio.transforms.Spectrogram(n_fft)
        self.dBer = torchaudio.transforms.AmplitudeToDB()
    
    def forward(self, input):
        x = self.specer(input)
        x = self.dBer(x)
        return x


class MelSpectrogramDBTransform(nn.Module):
    """
    Returns the Mel spectrogram in dB scale.
    input shape: (..., channel, time_stamp)
    output shape: (..., channel, time_stamp, n_fft2)
    """    
    def __init__(self, sample_rate=44100, n_fft=2048, n_mels=128):
        super().__init__()
        self.specer = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft, n_mels=n_mels)
        self.dBer = torchaudio.transforms.AmplitudeToDB()
    
    def forward(self, input):
        x = self.specer(input)
        x = self.dBer(x)
        return x


# class CQTDBTransform(nn.Module):
#     """
#     Returns the CQT in dB scale. 
#     input shape: (..., channel, time_stamp)
#     output shape: (..., channel, time_stamp, n_bins)
#     """

#     def __init__(self, sr=44100, hop_length=512, fmin=None, n_bins=360, bins_per_octave=60):
#         super().__init__()
#         if fmin is None:
#             fmin = librosa.convert.note_to_hz("E2") # E2 is the lowest note on regular-tuning 6-string guitar
#         self.cqter = nnAudio.Spectrogram.CQT(sr, hop_length, fmin, None, n_bins, bins_per_octave)
#         self.dBer = torchaudio.transforms.AmplitudeToDB()
    
#     def forward(self, input):
#         x = self.cqter(input)
#         x = self.dBer(x)
#         return x
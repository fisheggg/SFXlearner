"""
Modified from: 
K.Choi, G.Fazekas, K.Cho et al.,
Convolutional Recurrent Neural Networks for Music Classification

"""
import torch
import torch.nn as nn
from torch import Tensor

class CRNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=3, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(64, out_channels=128, kernel_size=3, padding=1, dilation=1)
        self.conv4 = nn.Conv2d(128, out_channels=128, kernel_size=(6,1), padding=0, dilation=1)
        self.gru1 = nn.GRU(128, hidden_size=32, num_layers=2, dropout=0.2, batch_first=True)
        self.gru2 = nn.GRU(32, hidden_size=32, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(32, num_classes)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=5, stride=(4,1), padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm1d(32)
        self.elu = nn.ELU(inplace=True)
      
    def forward(self, x: Tensor)->Tensor:
        # input shape: (..., n_channel, 128, 216) with n_mels=128, n_fft=2048, hop_length=1024
        # time resolution for 1 pixel: 23.2 ms       
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool1(x)
        # shape: (..., 32, 64, 108)      
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.maxpool2(x)
        # shape: (..., 64, 22, 36)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.maxpool3(x)
        # shape: (..., 128, 6, 36)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.elu(x)
        # shape: (..., 256, 1, 36)
        x = x[:,:,0,:].permute([0,2,1])
        # shape: (..., 36, 256)
        x, _ = self.gru1(x)
        # shape: (..., 36, 128)
        x = x.permute([0, 2, 1])
        x = self.bn5(x)
        x = self.elu(x)
        x = x.permute([0, 2, 1])
        _, x = self.gru2(x)
        # shape: (1, ..., n_classes)
        return self.fc(x[0])
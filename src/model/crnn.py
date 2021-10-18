"""
Modified from: 
K.Choi, G.Fazekas, K.Cho et al.,
Convolutional Recurrent Neural Networks for Music Classification


"""
from typing import Type, Any, Callable, Union, List, Optional
from PIL.Image import init

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from sklearn.metrics import f1_score

import pytorch_lightning as pl

class CRNN(pl.LightningModule):
    def __init__(self, in_channels: int, num_classes: int, with_clean: bool, lr: float, transform: nn.Module = None, log_class_loss: bool = False):
        super().__init__()
        self.with_clean = with_clean
        self.log_class_loss = log_class_loss
        self.lr = lr
        self.transform = transform
        
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
        if self.transform is not None:
            x = self.transform(x)
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

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        if self.with_clean == False:
            x = x[:, 1, :, :]
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        if self.with_clean == False:
            x = x[:, 1, :, :]
        y_hat = self(x)
        losses = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none')
        loss = torch.mean(losses)
        self.log('val_loss', loss, on_step=True, on_epoch=False)
        self.log('val_micro_f1', f1_score(y.data.cpu(), y_hat.data.cpu() > 0.5, average='micro'), on_step=True, on_epoch=False)
        self.log('val_macro_f1', f1_score(y.data.cpu(), y_hat.data.cpu() > 0.5, average='macro'), on_step=True, on_epoch=False)
        if self.log_class_loss:
            class_losses = torch.mean(losses, axis=0)
            self.log('val_class_loss_0', class_losses[0], on_step=True, on_epoch=False)
            self.log('val_class_loss_1', class_losses[1], on_step=True, on_epoch=False)
            self.log('val_class_loss_2', class_losses[2], on_step=True, on_epoch=False)
            self.log('val_class_loss_3', class_losses[3], on_step=True, on_epoch=False)
            self.log('val_class_loss_4', class_losses[4], on_step=True, on_epoch=False)
            self.log('val_class_loss_5', class_losses[5], on_step=True, on_epoch=False)
            self.log('val_class_loss_6', class_losses[6], on_step=True, on_epoch=False)
            self.log('val_class_loss_7', class_losses[7], on_step=True, on_epoch=False)
            self.log('val_class_loss_8', class_losses[8], on_step=True, on_epoch=False)
            self.log('val_class_loss_9', class_losses[9], on_step=True, on_epoch=False)
            self.log('val_class_loss_10', class_losses[10], on_step=True, on_epoch=False)
            self.log('val_class_loss_11', class_losses[11], on_step=True, on_epoch=False)
            self.log('val_class_loss_12', class_losses[12], on_step=True, on_epoch=False)
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        if self.with_clean == False:
            x = x[:, 1, :, :]
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

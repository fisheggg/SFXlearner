"""The Pytorch lightning wrappers of other models"""
import torch
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from torch.nn import functional as F
from torch import Tensor
class LightningWrapper(pl.LightningModule):
    """
    A wrapper that contains training, validation, and test steps.
    """
    def __init__(self, 
                 model: torch.nn.Module,
                 lr: float,
                 transform: torch.nn.Module, 
                 with_clean: bool=False, 
                 log_class_loss: bool=True):
        super().__init__()
        self.model = model
        self.lr = lr
        self.transform = transform
        self.log_class_loss = log_class_loss
        self.with_clean = with_clean
    
    def forward(self, x: Tensor)->Tensor:
        if self.transform is not None:
            x = self.transform(x)
        # shape: (..., n_channel, 128, 216) with n_mels=128, n_fft=2048, hop_length=1024
        # time resolution for 1 pixel: 23.2 ms
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        if not self.with_clean:
            x = x[:, 1, :].unsqueeze(1)
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        if not self.with_clean:
            x = x[:, 1, :].unsqueeze(1)
        y_hat = self(x)
        losses = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none')
        loss = torch.mean(losses)
        self.log('val_loss', loss)
        self.log('val_micro_f1', f1_score(y.data.cpu(), y_hat.data.cpu() > 0.5, average='micro', zero_division=0))
        self.log('val_macro_f1', f1_score(y.data.cpu(), y_hat.data.cpu() > 0.5, average='macro', zero_division=0))
        if self.log_class_loss:
            class_losses = torch.mean(losses, axis=0)
            for i in range(len(class_losses)):
                self.log(f'val_class_loss_{i}', class_losses[i])
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        if not self.with_clean:
            x = x[:, 1, :].unsqueeze(1)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
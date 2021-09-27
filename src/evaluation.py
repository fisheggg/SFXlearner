import torch
import pytorch_lightning as pl

def multiclass_recall(y, y_hat, n_classes):
    """
    Inputs
    ------
    shape: (N)
    range: [0, n_classes-1]

    Return
    ------
    Tensor of calculated recall
    shape: (n_classes)
    """
    TP, FP = torch.zeros(n_classes)

    for i in range(n_classes):
        pass
   
def multiclass_precision(y, y_hat, n_classes):
    pass

def multiclass_F1_score(y, y_hat, n_classes):
    pass

def confusion_matrix(y, y_hat, n_classes):
    pass

def plot_confusion_matrix(y, y_hat, n_classes):
    pass
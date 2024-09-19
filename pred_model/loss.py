import torch.nn.functional as F
import torch


def mse_loss(output, target):
    return F.mse_loss(output, target)



def mse_with_regularizer_loss(outputs, targets, model, lamda=1.5e-3):
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    mse_loss = torch.sum((outputs - targets) ** 2) / 2
    return mse_loss + reg_loss





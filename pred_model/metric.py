import torch
from utils import math as util
import torchmetrics

## CrowdCNNGRU model metrics
# def MAPE(y, y_, mean_std, mask):
#     y = util.z_inverse(y,mean_std[0],mean_std[1])[mask == True]
#     y_ = util.z_inverse(y_,mean_std[0],mean_std[1])[mask == True]
    
#     return util.MAPE(y,y_)

# def RMSE(y,y_, mean_std, mask):
#     y = util.z_inverse(y, mean_std[0], mean_std[1])[mask == True]
#     y_ = util.z_inverse(y_, mean_std[0], mean_std[1])[mask == True]

#     return util.RMSE(y, y_)

# def MAE(y,y_, mean_std, mask):
#     y = util.z_inverse(y, mean_std[0], mean_std[1])[mask == True]
#     y_ = util.z_inverse(y_, mean_std[0], mean_std[1])[mask == True]

#     return util.MAE(y, y_)

# def MAPE2(y, y_, mean_std, mask):
#     mask = mask[:,-1:,:,:]
#     y = y[:,-1:,:,:]
#     y_ = y_[:,-1:,:,:]
#     y = util.z_inverse(y,mean_std[0],mean_std[1])[mask == True]
#     y_ = util.z_inverse(y_,mean_std[0],mean_std[1])[mask == True]

#     return util.MAPE(y,y_)


# def RMSE2(y,y_, mean_std, mask):
#     mask = mask[:, -1:, :, :]
#     y = y[:, -1:, :, :]
#     y_ = y_[:, -1:, :, :]
#     y = util.z_inverse(y, mean_std[0], mean_std[1])[mask == True]
#     y_ = util.z_inverse(y_, mean_std[0], mean_std[1])[mask == True]

#     return util.RMSE(y,y_)


# def MAE2(y,y_, mean_std, mask):
#     mask = mask[:, -1:, :, :]
#     y = y[:, -1:, :, :]
#     y_ = y_[:, -1:, :, :]
#     y = util.z_inverse(y, mean_std[0], mean_std[1])[mask == True]
#     y_ = util.z_inverse(y_, mean_std[0], mean_std[1])[mask == True]

#     return util.MAE(y,y_)

def RMSE(outputs, targets):
    return torch.sqrt(torch.mean((outputs - targets) ** 2))
    # return torch.sqrt(torchmetrics.functional.mean_squared_error(outputs, targets))

def MAE(outputs, targets):
    return torchmetrics.functional.mean_absolute_error(outputs, targets)

def Accuracy(outputs, targets):
    return 1 - torch.linalg.norm(targets - outputs, "fro") / torch.linalg.norm(targets, "fro")

def R2(outputs,targets):
    return 1 - torch.sum((targets - outputs) ** 2) / torch.sum((targets - torch.mean(outputs)) ** 2)

def Explained_Variance(outputs, targets):
    return 1 - torch.var(targets - outputs) / torch.var(targets)
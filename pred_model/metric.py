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
    outputs = outputs.reshape(-1)
    targets = targets.reshape(-1)
    return torchmetrics.functional.mean_absolute_error(outputs, targets)

def MAPE(outputs, targets):
    return torchmetrics.functional.mean_absolute_percentage_error(outputs, targets)
    # return torchmetrics.functional.mean_absolute_error(outputs, targets)


def Accuracy(outputs, targets):
    #input B, T, N, D or B, N, D or B*N, D
    if len(targets.shape) == 2:
        pass
    elif len(targets.shape) == 3:
        # B, T, D-> B*T, D
        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])
    elif len(targets.shape)  == 4:
        #B, T, N, D -> B* T*N, D
        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])
    else:
        raise ValueError
    # outputs=outputs.squeeze(0)
    # targets = targets.squeeze(0)
    # outputs = outputs.reshape(-1)
    # targets = targets.reshape(-1)
    return 1 - torch.linalg.norm(targets - outputs, "fro") / torch.linalg.norm(targets, "fro")

def R2(outputs,targets):
    return 1 - torch.sum((targets - outputs) ** 2) / torch.sum((targets - torch.mean(outputs)) ** 2)

def Explained_Variance(outputs, targets):
    return 1 - torch.var(targets - outputs) / torch.var(targets)








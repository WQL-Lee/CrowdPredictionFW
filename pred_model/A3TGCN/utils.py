import scipy.sparse as sp
import numpy as np
import torch

def covert_adj_matrix_to_coo(adj_matrix):
    # adj_matrix 是邻接矩阵
    tmp_coo = sp.coo_matrix(adj_matrix)
    values = tmp_coo.data
    indices = np.vstack((tmp_coo.row,tmp_coo.col))
    i = torch.LongTensor(indices)
    v = torch.LongTensor(values)
    edge_index=torch.sparse_coo_tensor(i,v,tmp_coo.shape).float()._indices()
    return edge_index

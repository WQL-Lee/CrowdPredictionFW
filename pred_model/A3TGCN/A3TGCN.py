import torch
from torch_geometric_temporal.nn.recurrent import A3TGCN2 as TGNN
import torch.nn.functional as F
import pandas as pd
import numpy as np
from  pred_model.A3TGCN.utils import covert_adj_matrix_to_coo

class A3TGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, output_dim, batch_size, adj_path):
        super(A3TGCN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = TGNN(in_channels=node_features,  out_channels=hidden_dim, periods=output_dim, batch_size=batch_size) # node_features=2, periods=12
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

        adj = self.load_adj(adj_path)
        self.num_nodes = adj.shape[0]
        self.edge_index = covert_adj_matrix_to_coo(adj)
        self.name = "A3TGCN"


    def load_adj(self, adj_path):
        adj = pd.DataFrame(np.eye(10))
        adj = np.asmatrix(adj)
        return adj

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index) # x [b, 207, 2, 12]  returns h [b, 207, 12]
        h = F.relu(h) 
        h = self.linear(h)
        return h
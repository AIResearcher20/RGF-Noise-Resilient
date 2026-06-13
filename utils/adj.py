import torch
from torch_geometric.utils import to_dense_adj

def build_adj(edge_index, num_nodes):
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    A = A + torch.eye(num_nodes)

    deg = A.sum(dim=1)
    A = A / (deg.unsqueeze(1) + 1e-8)

    return A

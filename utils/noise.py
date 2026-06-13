import torch

def add_feature_noise(X, noise_level):
    return X + torch.randn_like(X) * noise_level


def add_structural_noise(edge_index, noise_level):
    num_edges = edge_index.shape[1]
    mask = torch.rand(num_edges) > noise_level
    return edge_index[:, mask]

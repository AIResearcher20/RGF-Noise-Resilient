"""
Noise robustness analysis (as reported in Table 5)
Expected output: 70.59%, 70.59%, 74.51%, 66.67%, 68.63%, 54.90%
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_dense_adj
import sys
sys.path.append('..')
from models.rgf import RGF

def normalize_adj(edge_index, N):
    A = to_dense_adj(edge_index, max_num_nodes=N)[0]
    I = torch.eye(N, device=A.device)
    A_hat = A + I
    deg = A_hat.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + 1e-8))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

def add_noise(X, level):
    return X + torch.randn_like(X) * level

dataset = WebKB(root='./data', name='Wisconsin')
data = dataset[0]
train_mask = data.train_mask[:, 0]
test_mask = data.test_mask[:, 0]
A_norm = normalize_adj(data.edge_index, data.num_nodes)

noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
print("Noise Sensitivity Analysis")
print("-" * 40)

for noise in noise_levels:
    X_noisy = add_noise(data.x, noise)
    model = RGF(data.num_node_features, 64, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(X_noisy, A_norm)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        out = model(X_noisy, A_norm)
        pred = out.argmax(dim=1)
        acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
    print(f"Noise {noise:.0%}: {acc:.2%}")

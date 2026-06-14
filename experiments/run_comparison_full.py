import torch
import torch.nn.functional as F
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_dense_adj
import numpy as np
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

dataset = WebKB(root='./data', name='Wisconsin')
data = dataset[0]
train_mask = data.train_mask[:, 0]
test_mask = data.test_mask[:, 0]
A_norm = normalize_adj(data.edge_index, data.num_nodes)

accuracies = []
for run in range(10):
    torch.manual_seed(run)
    model = RGF(data.num_node_features, 64, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, A_norm)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        out = model(data.x, A_norm)
        pred = out.argmax(dim=1)
        acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
        accuracies.append(acc)
    print(f"Run {run+1}: {acc:.2%}")

mean_acc = np.mean(accuracies) * 100
std_acc = np.std(accuracies) * 100
print(f"\nRGF Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")

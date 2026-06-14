import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, heads=4):
        super().__init__()
        self.heads = heads
        self.attn = nn.Linear(2 * hidden_dim, 1)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, X, A):
        N = X.size(0)
        H = F.relu(self.fc1(X))
        Hi = H.unsqueeze(1).repeat(1, N, 1)
        Hj = H.unsqueeze(0).repeat(N, 1, 1)
        pair = torch.cat([Hi, Hj], dim=-1)
        e = self.attn(pair).squeeze(-1)
        e = e.masked_fill(A == 0, -1e9)
        alpha = F.softmax(e, dim=1)
        H = alpha @ H
        out = self.fc2(H)
        return out

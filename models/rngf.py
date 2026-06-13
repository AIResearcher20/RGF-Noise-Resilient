import torch
import torch.nn as nn
import torch.nn.functional as F

class RNGF(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, X, A):
        X1 = F.relu(self.lin(X))
        X2 = F.relu(self.lin(A @ X))

        g = torch.sigmoid(self.gate(torch.cat([X1, X2], dim=1)))

        H = g * X1 + (1 - g) * X2

        return self.classifier(H)

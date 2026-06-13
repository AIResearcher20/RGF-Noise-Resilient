import torch
import numpy as np

def train_eval(model, X, A, y, train_mask, val_mask, test_mask, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(X, A)
        loss = loss_fn(out[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(X, A).argmax(dim=1)
            val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()

            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(X, A).argmax(dim=1)
        test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

    return test_acc, best_val

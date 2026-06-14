"""
Evaluation module for RGF model
"""

import torch

def evaluate(model, X, A, y, test_mask):
    """
    Evaluate RGF model on test set
    
    Args:
        model: Trained RGF model
        X: Node features [N, d]
        A: Normalized adjacency matrix [N, N]
        y: Node labels [N]
        test_mask: Boolean mask for test nodes
    
    Returns:
        acc: Classification accuracy on test nodes (as percentage, 0-100)
    """
    model.eval()
    with torch.no_grad():
        out = model(X, A)
        pred = out.argmax(dim=1)
        acc = (pred[test_mask] == y[test_mask]).float().mean().item() * 100
    return acc

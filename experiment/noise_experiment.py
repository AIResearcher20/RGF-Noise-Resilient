import torch
import numpy as np

from utils.noise import add_feature_noise
from train.train_eval import train_eval
from models.rngf import RNGF


def run_noise(config, data):

    X = data.x
    y = data.y

    A = data.A if hasattr(data, "A") else None

    train_mask = data.train_mask[:, 0]
    val_mask = data.val_mask[:, 0]
    test_mask = data.test_mask[:, 0]

    levels = config["noise"]["feature_noise_levels"]

    results = []

    for noise in levels:

        Xn = add_feature_noise(X, noise)

        model = RNGF(
            X.shape[1],
            config["model"]["hidden_dim"],
            int(y.max().item() + 1)
        )

        acc, _ = train_eval(
            model, Xn, A, y,
            train_mask, val_mask, test_mask,
            epochs=config["training"]["epochs"],
            lr=config["training"]["lr"]
        )

        results.append((noise, acc))

        print(f"Noise {noise}: {acc:.4f}")

    return results

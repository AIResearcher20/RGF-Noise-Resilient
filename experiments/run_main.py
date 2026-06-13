import torch
import numpy as np

from models.rngf import RNGF
from models.gcn import GCN
from models.gat import GAT
from models.graphsage import GraphSAGE
from models.mlp import MLP

from train.train_eval import train_eval
from utils.adj import build_adj


def run_main(config, data):

    X = data.x
    y = data.y

    A = build_adj(data.edge_index, data.num_nodes)

    train_mask = data.train_mask[:, 0]
    val_mask = data.val_mask[:, 0]
    test_mask = data.test_mask[:, 0]

    models = {
        "GCN": GCN,
        "GAT": GAT,
        "GraphSAGE": GraphSAGE,
        "MLP": MLP,
        "RGF": RNGF
    }

    results = {}

    for name, model_class in models.items():

        accs = []

        for seed in range(config["experiment"]["runs"]):

            torch.manual_seed(seed)

            model = model_class(
                X.shape[1],
                config["model"]["hidden_dim"],
                int(y.max().item() + 1)
            )

            acc, _ = train_eval(
                model,
                X,
                A,
                y,
                train_mask,
                val_mask,
                test_mask,
                epochs=config["training"]["epochs"],
                lr=config["training"]["lr"]
            )

            accs.append(acc)

        results[name] = {
            "mean": np.mean(accs),
            "std": np.std(accs)
        }

        print(f"{name}: {results[name]['mean']:.4f} ± {results[name]['std']:.4f}")

    return results

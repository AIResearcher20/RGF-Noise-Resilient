from models.gcn import GCN
from models.gat import GAT
from models.rngf import RNGF
from experiments.train import train_eval

def run_all(X, A, y, train_mask, val_mask, test_mask):

    models = {
        "GCN": GCN,
        "GAT": GAT,
        "RNGF": RNGF
    }

    results = {}

    for name, model_class in models.items():

        model = model_class(
            X.shape[1],
            64,
            int(y.max().item()) + 1
        )

        test_acc, val_acc = train_eval(
            model,
            X,
            A,
            y,
            train_mask,
            val_mask,
            test_mask
        )

        results[name] = test_acc
        print(f"{name}: {test_acc:.4f}")

    return results

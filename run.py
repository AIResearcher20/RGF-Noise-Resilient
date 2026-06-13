import torch
import yaml

from experiments.benchmark import run_experiments
from experiments.noise import noise_experiment
from models.gcn import GCN
from models.gat import GAT
from models.rngf import RNGF


def load_config(path="configs/wisconsin.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    config = load_config()

    print("\n🚀 RGF FULL PIPELINE START\n")

    # Load data
    X = torch.load(config["data"]["X"])
    A = torch.load(config["data"]["A"])
    y = torch.load(config["data"]["y"])

    train_mask = torch.load(config["data"]["train_mask"])
    val_mask   = torch.load(config["data"]["val_mask"])
    test_mask  = torch.load(config["data"]["test_mask"])

    models = {
        "GCN": GCN,
        "GAT": GAT,
        "RNGF": RNGF
    }

    results = {}

    print("📊 Benchmark running...\n")

    for name, model_class in models.items():

        model = model_class(
            X.shape[1],
            config["model"]["hidden_dim"],
            int(y.max().item()) + 1
        )

        accs = run_experiments(
            lambda: model,
            X, A, y,
            train_mask, val_mask, test_mask,
            n_runs=config["train"]["runs"]
        )

        results[name] = accs

        print(f"{name} done")

    print("\n📉 Noise experiments...\n")

    noise_results = noise_experiment(
        RNGF,
        X, A,
        None,
        y,
        train_mask, val_mask, test_mask,
        noise_levels=config["noise"]["levels"]
    )

    print("\n====================")
    print("FINAL RESULTS")
    print("====================")

    for k, v in results.items():
        print(k, sum(v)/len(v))

    print("\nNoise:", noise_results)


if __name__ == "__main__":
    main()
